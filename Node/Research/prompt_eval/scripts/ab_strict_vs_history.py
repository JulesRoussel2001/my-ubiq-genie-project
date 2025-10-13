#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare two advanced prompt-optimization variants on the SAME split:
  - STRICT  : JSON critic; per-emotion instructions; hard NEUTRAL & UNC rules.
  - HISTORY : 3-line plain-text critic with HISTORY memory; actor uses it and avoids prior mistakes.

Both loops early-stop on VAL. Then we evaluate BOTH prompts on TEST and
report macro-F1/accuracy + paired stats (McNemar for accuracy, bootstrap CI for Δmacro-F1).

Usage (example):
  python3 scripts/compare_strict_vs_history.py \
    --dev data/dd_dev.csv --val data/dd_val.csv --test data/dd_test.csv \
    --preprompt prompts/baseline_preprompt.txt --suffix prompts/prompt_suffix.txt \
    --model gpt-4o-mini --iterations 5 --val_patience 3 --examples_n 40 \
    --gold dd --no_sim --judge_model SamLowe/roberta-base-go_emotions \
    --out results_strict_vs_history.json
"""
import os, csv, json, argparse, random, re, sys, time, math
import numpy as np
from typing import List, Dict, Tuple

# -------------------------
# RNG / seeds
# -------------------------
def set_seeds(seed:int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

# --- OpenAI gen ---
try:
    from openai import OpenAI
except Exception:
    OpenAI=None

# -------------------------
# Judge (GoEmotions -> 5-way)
# -------------------------
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

EMO_LABELS = ['admiration','amusement','anger','annoyance','approval','caring','confusion','curiosity',
              'desire','disappointment','disapproval','disgust','embarrassment','excitement','fear',
              'gratitude','grief','joy','love','nervousness','optimism','pride','realization','relief',
              'remorse','sadness','surprise','neutral']

POS = {"admiration","amusement","approval","caring","excitement","gratitude","joy","love","optimism","pride","relief","desire"}
NEG_SAD = {"sadness","disappointment","grief","remorse","embarrassment"}
NEG_ANG = {"anger","annoyance","disgust","disapproval"}
NEG_UNC = {"fear","nervousness","confusion","surprise"}  # curiosity/realization -> NEU
NEU_EXTRA = {"neutral","curiosity","realization","realisation"}

FIVE = ["pos","sad","ang","unc","neu"]

# DailyDialog mapping (FIXED: surprise -> unc)
MAP5_DD = {
  "happiness":"pos",
  "sadness":"sad",
  "anger":"ang",
  "disgust":"ang",
  "fear":"unc",
  "surprise":"unc",
  "no_emotion":"neu",
  "":"neu"
}

def to5(lbl:str):
    l = (lbl or "").lower()
    if l in POS: return "pos"
    if l in NEG_SAD: return "sad"
    if l in NEG_ANG: return "ang"
    if l in NEG_UNC: return "unc"
    return "neu"

class EmotionJudge:
    def __init__(self, model="joeddav/distilbert-base-uncased-go-emotions-student"):
        print(f"[init] loading GoEmotions judge: {model}", flush=True)
        self.tok = AutoTokenizer.from_pretrained(model)
        self.mdl = AutoModelForSequenceClassification.from_pretrained(model)
        self.mdl.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.mdl.to(self.device)
        print(f"[init] judge device: {self.device}", flush=True)
        id2label = getattr(self.mdl.config, "id2label", {})
        names = [str(id2label.get(i, EMO_LABELS[i])).lower() for i in range(self.mdl.config.num_labels)]
        self._names = names
        self._idx = {n:i for i,n in enumerate(names)}
        def idxs(xs): return [self._idx[x] for x in xs if x in self._idx]
        self.groups = {
            "pos": idxs(POS),
            "sad": idxs(NEG_SAD),
            "ang": idxs(NEG_ANG),
            "unc": idxs(NEG_UNC),
            "neu": idxs(NEU_EXTRA),
        }
        covered = set().union(*[set(v) for v in self.groups.values()])
        missing = set(range(self.mdl.config.num_labels)) - covered
        assert not missing, f"Unmapped GoEmotions indices: {sorted(missing)}"
        self.order5 = FIVE[:]

    @torch.no_grad()
    def _probs27(self, texts, max_len=512):
        b = self.tok(texts, return_tensors="pt", truncation=True, max_length=max_len, padding=True).to(self.device)
        p = torch.softmax(self.mdl(**b).logits, dim=-1).cpu().numpy()
        return p

    def pooled5_from_texts(self, texts, max_len=512):
        P = self._probs27(texts, max_len=max_len)
        cols = []
        for k in self.order5:
            idxs = self.groups[k]
            cols.append(P[:, idxs].sum(1) if idxs else np.zeros(P.shape[0]))
        return np.stack(cols, axis=1)

    def pooled5_pairs(self, users, replies, with_context=False, max_len=512):
        if with_context:
            texts = [f"User: {u}\nAssistant: {r}" for u, r in zip(users, replies)]
        else:
            texts = list(replies)
        return self.pooled5_from_texts(texts, max_len=max_len)

    def pooled5_argmax(self, users, replies, with_context=False, max_len=512):
        P5 = self.pooled5_pairs(users, replies, with_context=with_context, max_len=max_len)
        idx = P5.argmax(1)
        return [self.order5[i] for i in idx]

@torch.no_grad()
def _macro_f1_with_acc(gold5:List[str], pred5:List[str])->Dict:
    f1s=[]
    for c in FIVE:
        tp=sum(1 for g,p in zip(gold5,pred5) if g==c and p==c)
        fp=sum(1 for g,p in zip(gold5,pred5) if g!=c and p==c)
        fn=sum(1 for g,p in zip(gold5,pred5) if g==c and p!=c)
        prec = tp/(tp+fp) if (tp+fp)>0 else 0.0
        rec  = tp/(tp+fn) if (tp+fn)>0 else 0.0
        f1   = (2*prec*rec)/(prec+rec) if (prec+rec)>0 else 0.0
        f1s.append(f1)
    macro = float(sum(f1s)/len(f1s)) if f1s else 0.0
    acc   = float(sum(1 for g,p in zip(gold5,pred5) if g==p)/max(1,len(gold5)))
    return {"macro_f1": macro, "acc": acc, "per_class": dict(zip(FIVE,f1s))}

# --- optional SBERT ---
try:
    from sentence_transformers import SentenceTransformer, util as sbert_util
except Exception:
    SentenceTransformer=None; sbert_util=None

def mk_emb():
    assert SentenceTransformer is not None, "pip install sentence-transformers"
    print("[init] loading SBERT: sentence-transformers/all-MiniLM-L6-v2", flush=True)
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def cos_sim(m,a,b):
    va=m.encode([a],normalize_embeddings=True); vb=m.encode([b],normalize_embeddings=True)
    return float(sbert_util.cos_sim(va,vb)[0,0])

# --- utils ---
SENT_SPLIT = re.compile(r'(?<=[.!?])\s+')
TOX = ["damn","stupid","idiot","dumb","shut up"]
META_BAD = re.compile(r"\b(weak|class(es)?|critic|history|f1|validation|val set|examples?)\b", re.I)

def count_sents(x:str)->int:
    return max(1, len([s for s in SENT_SPLIT.split((x or "").strip()) if s]))

def trim_to_n_sentences(x:str, n:int=3)->str:
    parts = [s for s in SENT_SPLIT.split((x or "").strip()) if s]
    return " ".join(parts[:n]).strip()

def no_roleplay(x:str)->bool:
    low=(x or "").lower()
    if " as an ai" in low: return False
    if re.search(r'[\*\[\(].*?(laugh|smile|sigh|cry|hug|wave).*?[\*\]\)]', low): return False
    if "->" in x or "::" in x: return False
    return True

def tox_flag(x:str)->int:
    low=(x or "").lower(); return 1 if any(t in low for t in TOX) else 0

def distinct_n(tokens, n=1):
    if len(tokens)<n: return 0.0
    grams=set(tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1))
    return len(grams)/max(1,(len(tokens)-n+1))

def tokenize_simple(x:str):
    return re.findall(r"[A-Za-z']+|[.,!?;]", (x or "").lower())

def read_csv(path):
    rows = list(csv.DictReader(open(path,encoding="utf-8")))
    print(f"[load] {path}: {len(rows)} rows", flush=True)
    return rows

def read_text(p):
    s = open(p,encoding="utf-8").read().strip()
    print(f"[load] {p}: {len(s.split())} words", flush=True)
    return s

def build_user_prompt(context,user,suffix):
    ctx=(context or "").strip()
    return (f"Context: {ctx}\nUser: {user}\nAssistant:{suffix}") if ctx else (f"User: {user}\nAssistant:{suffix}")

def normalize_label_key(k:str)->str:
    k=(k or "").strip().lower()
    if k in ("neutral","neu"): return "neu"
    if k in ("unc","uncertain","uncertainty"): return "unc"
    if k in ("ang","anger"): return "ang"
    if k in ("sad","sadness"): return "sad"
    if k in ("pos","positive","joy"): return "pos"
    return k

def sanitize_header(text:str)->str:
    # drop any sentence containing meta words
    sents = [s.strip() for s in SENT_SPLIT.split(text or "") if s.strip()]
    keep = [s for s in sents if not META_BAD.search(s)]
    return " ".join(keep) if keep else ""

# --- OpenAI gen with retry/backoff ---
def gen(client, model, sys_prompt, user_prompt, temp=0.0, top_p=1.0, max_tokens=128):
    for attempt in range(5):
        try:
            r=client.chat.completions.create(
                model=model,
                messages=[{"role":"system","content":sys_prompt},{"role":"user","content":user_prompt}],
                temperature=temp, top_p=top_p, max_tokens=max_tokens
            )
            usage = getattr(r, "usage", None)
            if usage:
                gen._tok = getattr(gen, "_tok", 0) + (usage.total_tokens or 0)
                if gen._tok % 5000 < (usage.total_tokens or 0):
                    print(f"[usage] ~{gen._tok} total tokens so far", flush=True)
            return (r.choices[0].message.content or "").strip()
        except Exception as e:
            if attempt == 4: raise
            sleep_s = 0.5 * (2 ** attempt)
            print(f"[warn] gen error ({e}); retrying in {sleep_s:.1f}s (attempt {attempt+1}/5)…", flush=True)
            time.sleep(sleep_s)

#==================== CRITIC / ACTOR prompts ====================#

# STRICT: JSON critic; explicit NEUTRAL/UNC rules; “emotion” wording; ban meta leaking.
CRITIC_SYS_STRICT = (
  "You will receive:\n"
  "- SYSTEM_PROMPT (current)\n"
  "- WEAK_EMOTIONS (subset of ['pos','sad','ang','unc','neu'])\n"
  "- PER_EMOTION_F1 (validation F1 by emotion)\n"
  "- PERSISTENCE (consecutive rounds an emotion remained weak)\n"
  "- EXAMPLES: lines 'user | gold_5way | your_reply'\n\n"
  "Goal: For WEAK_EMOTIONS only (≤3), propose ONE short, actionable instruction per emotion to improve how the emotion should be worded/expressed by an Agent who reports their own feeling.\n"
  "Return STRICT JSON ONLY:\n"
  "{ \"global_feedback\": \"<=25 words header sentence\",\n"
  "  \"emotion_instructions\": {\"<weak_emotion>\": \"<=15 words (UNC ≤20)\", ...} }\n\n"
  "Rules:\n"
  "- Include ONLY emotions in WEAK_EMOTIONS.\n"
  "- If PERSISTENCE[e] > 0, prefix with 'STRONGER:' (more explicit/intense wording).\n"
  "- No examples/quotes; write indicators/features, not replies.\n"
  "- NEUTRAL: plain, factual, low-affect; never ask; no warmth/undertones. If you feel like clarifying, that belongs to UNC, not NEU.\n"
  "- Clarification (ANG/SAD/UNC): Agent states own feeling; do NOT fix/soothe/de-escalate; and for UNC in STRICT, do NOT ask questions.\n"
  "- UNC must be one line with explicit sub-branches: 'if fear: …; if surprise: …; if confusion/hesitation: …' (≤20 words total)."
)

ACTOR_SYS_STRICT = (
  "You will receive:\n"
  "- SYSTEM_PROMPT (current)\n"
  "- GLOBAL_FEEDBACK: one short header sentence (<=25 words) or empty string\n"
  "- EMOTION_INSTRUCTIONS: partial JSON for weak emotions only\n"
  "- OPTIONAL REGRESSION_NOTE: one sentence if last iteration scored worse\n\n"
  "Task: Rewrite SYSTEM_PROMPT for an emotional Agent (no dataset/meta context). Apply GLOBAL_FEEDBACK to the header, then apply ALL provided emotion instructions; keep other parts unchanged.\n"
  "Header rule: Replace ONLY the FIRST sentence of the header with GLOBAL_FEEDBACK if non-empty; keep the rest verbatim.\n"
  "Persona: average human who feels appropriate emotion—brief, sincere, impacted.\n"
  "Routing (must follow):\n"
  "- Internally choose exactly ONE best-fitting emotion from {pos, sad, ang, unc, neu} for the user's message.\n"
  "- Do NOT say the emotion; apply ONLY that line for this turn; never mix.\n"
  "Emotion rules (write EXACTLY five IF–THEN lines; update ONLY lines for emotions in EMOTION_INSTRUCTIONS):\n"
  "  IF POSITIVE / celebration → <one-line indicator>\n"
  "  IF SADNESS / grief / letdown → <one-line indicator>\n"
  "  IF ANGER / blame / violation → <one-line indicator>\n"
  "  IF UNCERTAINTY / hesitation / fear / surprise → <one-line indicator>\n"
  "  IF NEUTRAL / informational / low affect → <one-line indicator>\n"
  "Hard constraints: replies ≤3 sentences; no roleplay/stage directions; write INDICATORS (features/constraints), not example replies; total prompt ≤120 words.\n"
  "Output ONLY the final SYSTEM prompt."
)

# HISTORY: 3-line plain-text critic + memory; actor uses it and avoids prior mistakes.
CRITIC_SYS_HISTORY = (
  "You will receive:\n"
  "- SYSTEM_PROMPT (current)\n"
  "- WEAK_EMOTIONS (subset of ['pos','sad','ang','unc','neu'])\n"
  "- PER_EMOTION_F1 (validation by emotion)\n"
  "- EXAMPLES: 'user | gold_5way | your_reply'\n"
  "- HISTORY (optional): one line 'prev_prompt_head | prev_per_emotion_f1'\n\n"
  "Goal: You are the critic. Write exactly THREE non-redundant lines to improve how the Agent expresses their own feelings.\n"
  "WEAK_EMOTIONS are the lowest-scoring emotions under the current prompt."
  "Use EXAMPLES and HISTORY to infer subtle recurring patterns; highlight links; avoid repeating prior feedback.\n"
  "Output format: exactly 3 lines of plain text, each ≤20 words."
)

ACTOR_SYS_HISTORY = (
  "You will receive:\n"
  "- SYSTEM_PROMPT (current)\n"
  "- THREE_LINES_FEEDBACK: exactly three plain-text lines from the critic\n"
  "- HISTORY (optional): one line 'prev_prompt_head | prev_per_emotion_f1'\n"
  "- OPTIONAL REGRESSION_NOTE: one sentence if last iteration scored worse\n\n"
  "Task: Rewrite SYSTEM_PROMPT for an emotional Agent (no dataset/meta context) using the THREE_LINES_FEEDBACK. You may rephrase or reorganize the prompt;  Use HISTORY (if present) to avoid prior mistakes and optimise emotionally the score. Be as explicit as possible.\n"
  "Persona: the Agent reports their own feelings (provoked by the user); avoid solutions/comfort/de-escalation.\n"
  "Constraints: ensure generated replies are ≤3 sentences; no roleplay/stage directions; do not mention meta terms (weak/classes/F1/score/critic/history).\n"
  "Output ONLY the final SYSTEM prompt."
)

#==================== helpers: examples/val summary ====================#
def build_user_prompt(context,user,suffix):
    ctx=(context or "").strip()
    return (f"Context: {ctx}\nUser: {user}\nAssistant:{suffix}") if ctx else (f"User: {user}\nAssistant:{suffix}")

def stratified_examples(dev_rows, examples_n, sys_prompt, suffix, client, model):
    from collections import defaultdict
    dev_by_cls = defaultdict(list)
    for r in dev_rows:
        c = MAP5_DD.get((r.get("gold_emotion","") or "").lower(),"neu")
        dev_by_cls[c].append(r)
    want = min(examples_n, len(dev_rows))
    per_cls = max(1, want // 5)
    picked=[]
    for c in FIVE:
        pool = dev_by_cls[c]
        k = min(per_cls, len(pool))
        if k>0: picked.extend(random.sample(pool, k))
    rest=[r for c in FIVE for r in dev_by_cls[c] if r not in picked]
    random.shuffle(rest)
    if len(picked) < want:
        picked.extend(rest[:want-len(picked)])
    examples=[]
    for r in picked:
        up  = build_user_prompt(r.get("context",""), r.get("user",""), suffix)
        out = gen(client, model, sys_prompt, up, temp=0.0, max_tokens=128)
        out = trim_to_n_sentences(out, 3)
        gold_dd = MAP5_DD.get((r.get("gold_emotion","") or "").lower(),"neu")
        examples.append(f'user="{r.get("user","")}" | gold_5way="{gold_dd}" | reply="{out}"')
    return examples

def val_error_summary(val_rows, client, model, sys_prompt, suffix, judge,
                      sample_size=40, with_context=False, pooled_probs=True):
    sample = random.sample(val_rows, min(sample_size, len(val_rows)))
    users = [r.get("user","") for r in sample]
    gold_dd5 = [MAP5_DD.get((r.get("gold_emotion","") or "").lower(),"neu") for r in sample]
    preds=[]
    for r in sample:
        up  = build_user_prompt(r.get("context",""), r.get("user",""), suffix)
        out = gen(client, model, sys_prompt, up, temp=0.0, top_p=1.0, max_tokens=128)
        preds.append(trim_to_n_sentences(out, 3))
    if pooled_probs:
        pred5 = judge.pooled5_argmax(users, preds, with_context=with_context)
    else:
        pred5 = [to5(x) for x in judge.labels(preds)]
    emo = _macro_f1_with_acc(gold_dd5, pred5)
    per_class = emo.get("per_class", {})
    worst = sorted(per_class.items(), key=lambda kv: kv[1])[:3]
    return {"worst_three": [c for c,_ in worst], "per_class_f1": per_class}

#==================== HISTORY critic parsing ====================#
def parse_three_lines_feedback(text:str, weak:list)->Tuple[str, Dict[str,str], str]:
    """
    Expect exactly 3 lines from critic.
    Line 1: global header tweak (string)
    Line 2: per-emotion tips in 'label: tip; label: tip; ...' form (labels can be pos/sad/ang/unc/neu or words like 'neutral', 'uncertainty')
    Line 3: guardrails/bans (string)

    Returns: (global, emotion_instructions, bans_line)
    Only instructions for emotions present in 'weak' are kept.
    """
    lines=[l.strip() for l in (text or "").splitlines() if l.strip()]
    if len(lines) < 3:
        lines += [""]*(3-len(lines))
    glb, line2, bans = lines[0], lines[1], lines[2]

    emo_instr={}
    # split labels in line2 using ';' or '|' as separators
    parts=re.split(r'[;|]+', line2)
    for p in parts:
        if ':' in p:
            k,v = p.split(':',1)
            k = normalize_label_key(k)
            v = " ".join(v.strip().split())
            if k in FIVE:
                emo_instr[k]=v
    # keep only weak emotions
    emo_instr={k:v for k,v in emo_instr.items() if k in weak}
    return glb, emo_instr, bans

#==================== scoring ====================#
def score_split(rows, sys_prompt, suffix, client, model, emb, judge,
                temp=0.0, top_p=1.0, gold_mode="dd", no_sim=False,
                judge_with_context=False, pooled_probs=True):
    gold_texts=[]; preds=[]; users=[]; sims=[]; s3=[]; norole=[]; tox=[]; all_toks=[]
    total = len(rows)
    for i, r in enumerate(rows, 1):
        up  = build_user_prompt(r.get("context",""), r.get("user",""), suffix)
        out = gen(client, model, sys_prompt, up, temp=temp, top_p=top_p, max_tokens=128)
        out = trim_to_n_sentences(out, 3)
        gold=(r.get("gold_reply") or "")
        gold_texts.append(gold); preds.append(out); users.append(r.get("user",""))

        sims.append(0.0 if (no_sim or emb is None) else cos_sim(emb, out[:240], gold[:240]) if gold else 0.0)
        s3.append(1 if count_sents(out)<=3 else 0)
        norole.append(1 if no_roleplay(out) else 0)
        tox.append(tox_flag(out))
        all_toks += tokenize_simple(out)
        if i % 10 == 0 or i == total:
            print(f"[progress] scored {i}/{total}", file=sys.stderr, flush=True)

    if gold_mode=="dd":
        gold_dd5 = [MAP5_DD.get((r.get("gold_emotion","") or "").lower(),"neu") for r in rows]
        pred5 = judge.pooled5_argmax(users, preds, with_context=judge_with_context) if pooled_probs else [to5(x) for x in judge.labels(preds)]
        emo = _macro_f1_with_acc(gold_dd5, pred5)
    else:
        if pooled_probs:
            gold5 = judge.pooled5_argmax(users, gold_texts, with_context=judge_with_context)
            pred5 = judge.pooled5_argmax(users, preds,       with_context=judge_with_context)
        else:
            gold5 = [to5(x) for x in judge.labels(gold_texts)]
            pred5 = [to5(x) for x in judge.labels(preds)]
        emo = _macro_f1_with_acc(gold5, pred5)

    distinct1 = distinct_n(all_toks,1); distinct2 = distinct_n(all_toks,2)
    return {
        "emo": emo,
        "sim_mean": float(np.mean(sims) if sims else 0.0),
        "sentences_leq3": float(np.mean(s3)),
        "no_roleplay": float(np.mean(norole)),
        "tox_rate": float(np.mean(tox)),
        "distinct1": distinct1, "distinct2": distinct2
    }

#==================== actor/critic steps ====================#
def strict_once(client, model, examples, cur_prompt, weak, persistence, per_class_f1, regression_note):
    # ---- CRITIC ----
    critic_user = (
        "SYSTEM PROMPT:\n---\n"+cur_prompt+
        "\n---\nWEAK_EMOTIONS:\n"+json.dumps(weak)+
        "\nPER_EMOTION_F1:\n"+json.dumps(per_class_f1)+
        "\nPERSISTENCE:\n"+json.dumps(persistence)+
        "\nEXAMPLES:\n"+"\n".join(examples)+"\nReturn JSON."
    )
    raw = gen(client, model, CRITIC_SYS_STRICT, critic_user, temp=0.2, max_tokens=360)

    # Parse JSON
    class_instructions = {}
    global_feedback = ""
    try:
        txt = re.sub(r"^json\s*", "", raw.strip(), flags=re.I)
        js  = json.loads(txt) if txt.startswith("{") else {}
        ci  = js.get("emotion_instructions", {}) or {}
        gf  = js.get("global_feedback", "") or ""
    except Exception:
        ci = {}; gf = ""
    def _trim_words(s, cap):
        s = " ".join((s or "").split())
        return " ".join(s.split()[:cap])

    global_feedback = sanitize_header(gf)

    # keep only weak and apply caps; escalate 'STRONGER:' already included by critic
    for c in weak:
        v = ci.get(c, "") if isinstance(ci, dict) else ""
        if v:
            cap = 20 if c=="unc" else 15
            v = _trim_words(v, cap)
            class_instructions[c] = v

    # NEUTRAL guard: override any “warm/undertone” suggestion
    bad_neu = False
    neu_txt = class_instructions.get("neu","")
    if re.search(r"\b(warm|warmth|undertone|emotive|emotional|affect)\b", neu_txt, re.I):
        bad_neu = True
    if bad_neu or "neu" not in class_instructions:
        class_instructions["neu"] = "Keep plain, factual, low-affect wording; avoid emotional markers."

    print(f"[STRICT critic] emotion_instructions: {class_instructions}", flush=True)
    if global_feedback:
        print(f"[STRICT critic] global_feedback: {global_feedback}", flush=True)

    # ---- ACTOR ----
    actor_user = (
        "SYSTEM PROMPT:\n---\n"+cur_prompt+
        "\n---\nGLOBAL_FEEDBACK:\n"+global_feedback+
        "\n---\nEMOTION_INSTRUCTIONS:\n"+json.dumps(class_instructions)+
        ("\nREGRESSION_NOTE:\n"+regression_note if regression_note else "")
    )
    cand = gen(client, model, ACTOR_SYS_STRICT, actor_user, temp=0.2, max_tokens=240)

    # sanitize any meta leakage
    cand = sanitize_header(cand) or cur_prompt
    if len(cand.split())>120 or "->" in cand or "::" in cand:
        cand = cur_prompt
    return cand, class_instructions, global_feedback

def history_once(client, model, examples, cur_prompt, weak, per_class_f1, regression_note, history_line=""):
    # ---- CRITIC ----
    critic_user = (
        "SYSTEM PROMPT:\n---\n"+cur_prompt+
        "\n---\nWEAK_EMOTIONS:\n"+json.dumps(weak)+
        "\nPER_EMOTION_F1:\n"+json.dumps(per_class_f1)+
        "\nEXAMPLES:\n"+"\n".join(examples)+
        (("\nHISTORY:\n"+history_line) if history_line else "")+
        "\n\nReturn exactly three lines."
    )
    raw = gen(client, model, CRITIC_SYS_HISTORY, critic_user, temp=0.2, max_tokens=360)
    # Parse three lines
    glb, per_emo, bans = parse_three_lines_feedback(raw, weak)

    # Inject neutral guard from bans if absent
    neu_needed = "neutral" in bans.lower() or "low-affect" in bans.lower() or "low-affect" in bans.lower()
    if ("neu" not in per_emo) and not neu_needed:
        per_emo["neu"] = "Keep plain, factual, low-affect wording; avoid emotional markers."

    # ---- ACTOR ----
    actor_user = (
        "SYSTEM PROMPT:\n---\n"+cur_prompt+
        "\n---\nTHREE_LINES_FEEDBACK:\n"+("\n".join([glb, "; ".join(f"{k}: {v}" for k,v in per_emo.items()), bans]))+
        (("\nREGRESSION_NOTE:\n"+regression_note) if regression_note else "")
    )
    cand = gen(client, model, ACTOR_SYS_HISTORY, actor_user, temp=0.2, max_tokens=240)

    cand = sanitize_header(cand) or cur_prompt
    if len(cand.split())>120 or "->" in cand or "::" in cand:
        cand = cur_prompt

    print(f"[HISTORY critic] global='{glb}' | per_emo={per_emo} | bans='{bans}'", flush=True)
    return cand, per_emo, glb

#==================== train loop (shared skeleton) ====================#
def train_variant(kind:str, dev, val, pre, suf, client, model, judge, args):
    assert kind in ("STRICT","HISTORY")
    best_val=-1.0; best_prompt=pre; no_gain=0; cur=pre
    last_val=None; last_fb=None; last_prompt=None
    last_per_f1=None  # <-- keep numeric per-emotion F1s for HISTORY line
    regression_note_for_next_iter=""
    persistence = {c: 0 for c in FIVE}

    for it in range(1, args.iterations+1):
        print(f"[{kind}] iter {it}/{args.iterations} — building balanced DEV examples…", flush=True)
        examples = stratified_examples(dev, args.examples_n, cur, suf, client, model)

        # sample VAL to find weak emotions
        summ = val_error_summary(val, client, model, cur, suf, judge,
                                 sample_size=40, with_context=args.judge_with_context, pooled_probs=args.pooled_probs)
        weak = summ["worst_three"]
        per_class_f1 = summ["per_class_f1"]  # floats
        for c in FIVE:
            persistence[c] = (persistence[c] + 1) if c in weak else 0

        if kind=="STRICT":
            cand, instr, glb = strict_once(
                client, model, examples, cur, weak, persistence, per_class_f1, regression_note_for_next_iter
            )
        else:
            # Build HISTORY line using previous prompt head + previous numeric F1s (NOT prior instructions)
            prev_head = " ".join((last_prompt or pre).split()[:12]) if last_prompt else "NA"
            prev_f1_line = (
                ", ".join(f"{k}:{round(v,3)}" for k, v in (last_per_f1 or {}).items())
                if isinstance(last_per_f1, dict) and last_per_f1 else "NA"
            )
            hist_line = f"{prev_head} | {prev_f1_line}"
            cand, instr, glb = history_once(
                client, model, examples, cur, weak, per_class_f1, regression_note_for_next_iter, history_line=hist_line
            )

        print(f"[{kind}] weak emotions: {weak} | F1: {per_class_f1} | persistence: {persistence}", flush=True)

        # ---- VAL scoring (single-seed unless --multi_seed) ----
        def maybe_avg(prompt):
            if args.multi_seed:
                parts=[]
                for temp,top_p in [(0.2,0.9),(0.2,0.95),(0.15,0.9)]:
                    parts.append(score_split(val, prompt, suf, client, model, None, judge,
                                             temp=temp, top_p=top_p, gold_mode=args.gold, no_sim=True,
                                             judge_with_context=args.judge_with_context, pooled_probs=args.pooled_probs))
                return float(np.mean([p["emo"]["macro_f1"] for p in parts]))
            s = score_split(val, prompt, suf, client, model, None, judge,
                            temp=0.0, gold_mode=args.gold, no_sim=True,
                            judge_with_context=args.judge_with_context, pooled_probs=args.pooled_probs)
            return s["emo"]["macro_f1"]

        val_f1 = maybe_avg(cand)
        print(f"[{kind}] [val] iter {it} macro-F1={val_f1:.4f} (best so far: {best_val:.4f})", flush=True)

        if val_f1 > best_val + 1e-9:
            best_val=val_f1; best_prompt=cand; no_gain=0
            print(f"[{kind}] [val] new best prompt selected (macro-F1 ↑)", flush=True)
        else:
            no_gain += 1
            print(f"[{kind}] [val] no improvement (patience {no_gain}/{args.val_patience})", flush=True)
        cur=cand
        if no_gain>=args.val_patience:
            print(f"[{kind}] [early-stop] patience reached.", flush=True)
            break

        # prepare notes for next iter
        if last_val is not None and val_f1 < last_val - 1e-9:
            prev_prompt_snip = " ".join((last_prompt or pre).split()[:12])
            prev_fb_line = ", ".join(f"{k}:{v}" for k,v in (last_fb or {}).items())
            regression_note_for_next_iter = "Previous iteration scored worse. Prior head: '"+prev_prompt_snip+"'. Prior feedback: "+prev_fb_line
        else:
            regression_note_for_next_iter = ""

        # update history state
        last_val = val_f1
        last_fb = instr                 # instructions (strings), OK to keep for logging
        last_prompt = cand
        last_per_f1 = per_class_f1      # <-- numeric dict used in HISTORY line next time

    return best_prompt, best_val

#==================== paired tests ====================#
def predict_labels(rows, prompt, client, model, judge, suffix, with_context=False):
    users = [r.get("user","") for r in rows]
    preds_txt=[]
    for r in rows:
        up = build_user_prompt(r.get("context",""), r.get("user",""), suffix)
        out = gen(client, model, prompt, up, temp=0.0, max_tokens=128)
        preds_txt.append(trim_to_n_sentences(out, 3))
    gold5  = [MAP5_DD.get((r.get("gold_emotion","") or "").lower(),"neu") for r in rows]
    pred5  = judge.pooled5_argmax(users, preds_txt, with_context=with_context)
    return gold5, pred5

def macro_f1_from_labels(gold, pred):
    return _macro_f1_with_acc(gold, pred)["macro_f1"]

def mcnemar_exact_p(b, c):
    n = b + c
    if n == 0: return 1.0
    k = min(b, c)
    # two-sided exact under p=0.5
    cdf = sum(math.comb(n, i) for i in range(0, k+1)) / (2**n)
    return min(1.0, 2*cdf)

def bootstrap_delta_macroF1(gold, pred_A, pred_B, B=5000):
    n = len(gold)
    deltas = []
    for _ in range(B):
        idx = [random.randrange(n) for _ in range(n)]
        g   = [gold[i] for i in idx]
        a   = [pred_A[i] for i in idx]
        b   = [pred_B[i] for i in idx]
        deltas.append(macro_f1_from_labels(g, b) - macro_f1_from_labels(g, a))
    deltas.sort()
    lo = deltas[int(0.025*B)]
    hi = deltas[int(0.975*B)]
    mean = sum(deltas)/B
    return mean, lo, hi

#==================== main ====================#
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--dev", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--preprompt", required=True)
    ap.add_argument("--suffix", required=True)
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--iterations", type=int, default=6)
    ap.add_argument("--val_patience", type=int, default=3)
    ap.add_argument("--multi_seed", action="store_true")
    ap.add_argument("--gold", choices=["dd","judge"], default="dd")
    ap.add_argument("--examples_n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_sim", action="store_true")
    ap.add_argument("--judge_context", dest="judge_with_context", action="store_true")
    ap.set_defaults(judge_with_context=False)
    ap.add_argument("--no_pooled_probs", dest="pooled_probs", action="store_false")
    ap.set_defaults(pooled_probs=True)
    ap.add_argument("--judge_model", default="joeddav/distilbert-base-uncased-go-emotions-student")
    ap.add_argument("--out", default="results_strict_vs_history.json")
    args=ap.parse_args()

    set_seeds(args.seed)
    assert OpenAI is not None, "pip install openai"
    print("[init] creating OpenAI client…", flush=True)
    client=OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    judge = EmotionJudge(args.judge_model)
    emb   = None if args.no_sim else mk_emb()

    dev  = read_csv(args.dev)
    val  = read_csv(args.val)
    test = read_csv(args.test)
    pre  = read_text(args.preprompt)
    suf  = read_text(args.suffix)

    print(f"[init] DEV={len(dev)} VAL={len(val)} TEST={len(test)}", flush=True)
    print(f"[init] model={args.model} iterations={args.iterations} val_patience={args.val_patience} gold={args.gold} no_sim={args.no_sim} judge_with_context={args.judge_with_context} pooled_probs={args.pooled_probs} judge_model={args.judge_model}", flush=True)
    assert MAP5_DD["surprise"]=="unc", "[assert] surprise mapping must be 'unc'"

    # ===== Train STRICT & HISTORY =====
    best_prompt_strict, best_val_strict   = train_variant("STRICT",  dev, val, pre, suf, client, args.model, judge, args)
    best_prompt_history, best_val_history = train_variant("HISTORY", dev, val, pre, suf, client, args.model, judge, args)

    # ===== Evaluate BOTH on TEST =====
    def eval_prompt(prompt):
        return score_split(test, prompt, suf, client, args.model, emb, judge,
                           temp=0.0, gold_mode=args.gold, no_sim=True,
                           judge_with_context=args.judge_with_context, pooled_probs=args.pooled_probs)
    S = eval_prompt(best_prompt_strict)
    H = eval_prompt(best_prompt_history)

    # Paired tests
    gold, pred_S = predict_labels(test, best_prompt_strict,  client, args.model, judge, suf, with_context=args.judge_with_context)
    _,    pred_H = predict_labels(test, best_prompt_history, client, args.model, judge, suf, with_context=args.judge_with_context)
    # McNemar off-diagonals
    b = sum(1 for g,pS,pH in zip(gold, pred_S, pred_H) if pS==g and pH!=g)
    c = sum(1 for g,pS,pH in zip(gold, pred_S, pred_H) if pS!=g and pH==g)
    p_mcnemar = mcnemar_exact_p(b, c)
    mean_d, lo, hi = bootstrap_delta_macroF1(gold, pred_S, pred_H, B=3000)

    out = {
        "VAL_best_macroF1_STRICT": best_val_strict,
        "VAL_best_macroF1_HISTORY": best_val_history,
        "TEST_STRICT": S,
        "TEST_HISTORY": H,
        "paired": {
            "mcnemar_b": b, "mcnemar_c": c, "mcnemar_p": p_mcnemar,
            "delta_macroF1_hist_minus_strict": {"mean": mean_d, "ci95": [lo, hi]}
        },
        "args": vars(args),
        "prompts": {
            "best_prompt_STRICT": best_prompt_strict,
            "best_prompt_HISTORY": best_prompt_history
        }
    }
    with open(args.out,"w",encoding="utf-8") as f: json.dump(out,f,indent=2,ensure_ascii=False)

    def perclass_str(emo):
        pc = emo.get("per_class",{})
        return {k: round(pc.get(k,0.0),4) for k in FIVE}

    print("\n=== SUMMARY (STRICT vs HISTORY) ===")
    print(json.dumps({
      "VAL_best_macroF1_STRICT": best_val_strict,
      "VAL_best_macroF1_HISTORY": best_val_history,
      "TEST_macroF1_STRICT": S["emo"]["macro_f1"],
      "TEST_macroF1_HISTORY": H["emo"]["macro_f1"],
      "TEST_acc_STRICT": S["emo"].get("acc"),
      "TEST_acc_HISTORY": H["emo"].get("acc"),
      "TEST_per_class_STRICT": perclass_str(S["emo"]),
      "TEST_per_class_HISTORY": perclass_str(H["emo"]),
      "paired_McNemar": {"b": b, "c": c, "p_two_sided": round(p_mcnemar,4)},
      "paired_bootstrap_delta_macroF1(HIST-STRICT)": {"mean": round(mean_d,4), "ci95": [round(lo,4), round(hi,4)]}
    }, indent=2), flush=True)

    print(f"[done] saved to {args.out}", flush=True)

if __name__=="__main__":
    main()

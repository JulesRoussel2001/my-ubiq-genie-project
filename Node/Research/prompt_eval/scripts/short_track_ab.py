#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Short track A/B with prompt optimization (v7: weak-classes critic, emotion-expression mode, IF–THEN routing).

What’s new in v7
- Critic proposes instructions ONLY for the current WEAK classes (≤3).
- UNC is no longer forced into sub-branches; the critic can freely suggest structure/questions.
- Persistence: if a class stayed weak, Critic prefixes "STRONGER:"; Actor escalates that to "MANDATE:" in the final rules.
- Actor rewrites the system prompt with EXACTLY five IF–THEN lines (pos/sad/ang/unc/neu); updates ONLY the weak classes.
- Emotion-expression mode: Agent reports their own feeling (provoked by the user); avoid solutions/comfort.
- Enforce ≤3 sentences in all generated replies (DEV/VAL/TEST and DEV examples shown to critic).
- Single-seed TEST scoring unless --multi_seed is passed (no silent duplicates).

CLI example:
  python3 scripts/short_track_ab.py \
    --dev data/dd_dev.csv --val data/dd_val.csv --test data/dd_test.csv \
    --preprompt prompts/baseline_preprompt.txt --suffix prompts/prompt_suffix.txt \
    --model gpt-4o-mini --iterations 5 --val_patience 3 --examples_n 40 \
    --gold dd --no_sim --out results_short_v7.json
"""
import os, csv, json, argparse, random, re, sys, time
import numpy as np
from typing import List, Dict

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

# -------- CRITIC / ACTOR prompts (v7) --------
CRITIC_SYS = (
  "You will receive:\n"
  "- SYSTEM_PROMPT (current)\n"
  "- WEAK_CLASSES (subset of ['pos','sad','ang','unc','neu'])\n"
  "- PER_CLASS_F1 (validation F1 by class)\n"
  "- PERSISTENCE (consecutive rounds a class remained weak)\n"
  "- EXAMPLES: a few lines 'user | gold_5way | your_reply'\n"
  "- HISTORY (optional): one line 'prev_prompt_head | prev_per_class_f1 | prev_feedback'\n\n"
  "Goal: For WEAK_CLASSES only (≤3), propose ONE short, actionable instruction per listed class to improve\n"
  "how the emotion should be worded/expressed by Agent who reports their own feeling.\n"
  "Return STRICT JSON ONLY:\n"
  "{ \"global_feedback\": \"<=25 words header sentence\",\n"
  "  \"class_instructions\": {\"<weak_class>\": \"<=15 words (UNC ≤20)\", ...} }\n\n"
  "Rules:\n"
  "- Include ONLY classes from WEAK_CLASSES.\n"
  "- If PERSISTENCE[class] > 0, propose a different tactic, not just stronger wording.\n"
  "- Use EXAMPLES and HISTORY to infer subtle recurring patterns; highlight links; avoid repeating prior feedback.\n"
  "- No examples or quoted phrases; write indicators/features, not replies.\n"
  "- Clarification: Agent reports own feelings; do NOT fix/soothe/de-escalate. For NEU: keep plain, factual, low-affect wording.\n"
  "- UNC must be one line with explicit sub-branches: 'if fear: …; if surprise: …; if confusion/hesitation: …' (≤20 words total)."
)

ACTOR_SYS = (
  "You will receive:\n"
  "- SYSTEM_PROMPT (current)\n"
  "- GLOBAL_FEEDBACK: one short header sentence (<=25 words) or empty string\n"
  "- CLASS_INSTRUCTIONS: partial JSON for weak classes only\n"
  "- HISTORY (optional): one line 'prev_prompt_head | prev_per_class_f1'\n"
  "- OPTIONAL REGRESSION_NOTE: one sentence only if last iteration scored worse\n\n"
  "Task: Rewrite SYSTEM_PROMPT applying GLOBAL_FEEDBACK to the header, then apply ALL provided class instructions to those classes; keep other classes unchanged. Use HISTORY to keep helpful constraints and avoid prior mistakes.\n"
  "Header rule: Replace ONLY the FIRST sentence of the header with GLOBAL_FEEDBACK if non-empty. Keep the rest of the header verbatim.\n"
  "Persona: respond like an average human who feels and shows appropriate emotion—brief, sincere, impacted—not a corporate/pleasing agent.\n"
  "Routing (must follow):\n"
  "- Internally choose exactly ONE best-fitting class from {pos, sad, ang, unc, neu} for the user's message.\n"
  "- Do NOT say the class; apply ONLY that class's rule for this turn; never mix classes.\n"
  "Class rules (write EXACTLY five IF–THEN lines; update ONLY lines for classes in CLASS_INSTRUCTIONS):\n"
  "  IF the message likely fits POSITIVE / celebration → <one-line indicator>\n"
  "  IF the message likely fits SADNESS / grief / letdown → <one-line indicator>\n"
  "  IF the message likely fits ANGER / blame / violation → <one-line indicator>\n"
  "  IF the message likely fits UNCERTAINTY / hesitation / fear / surprise → <one-line indicator>\n"
  "  IF the message is NEUTRAL / informational / low affect → <one-line indicator>\n"
  "Emotion-expression mode (ANG/SAD/UNC): Agent expresses own feeling provoked by the user; avoid solutions/comfort.\n"
  "Hard constraints: replies ≤3 sentences; no roleplay/stage directions; write INDICATORS (features/constraints), not example replies; total prompt ≤120 words.\n"
  "Output ONLY the final SYSTEM prompt."
)
# -------- helper: VAL sample to get weak classes + per-class F1 --------
def val_error_summary(val_rows, client, model, sys_prompt, suffix, judge,
                      sample_size=40, with_context=False, pooled_probs=True):
    sample = random.sample(val_rows, min(sample_size, len(val_rows)))
    users = [r.get("user","") for r in sample]
    gold_dd5 = [MAP5_DD.get((r.get("gold_emotion","") or "").lower(),"neu") for r in sample]
    preds=[]
    for r in sample:
        up  = build_user_prompt(r.get("context",""), r.get("user",""), suffix)
        out = gen(client, model, sys_prompt, up, temp=0.0, top_p=1.0, max_tokens=128)
        out = trim_to_n_sentences(out, 3)  # enforce ≤3 sentences
        preds.append(out)
    if pooled_probs:
        pred5 = judge.pooled5_argmax(users, preds, with_context=with_context)
    else:
        pred5 = [to5(x) for x in judge.labels(preds)]
    emo = _macro_f1_with_acc(gold_dd5, pred5)
    per_class = emo.get("per_class", {})
    worst = sorted(per_class.items(), key=lambda kv: kv[1])[:3]  # THREE weakest
    return {"worst_three": [c for c,_ in worst], "per_class_f1": per_class}

# -------- build balanced DEV examples --------
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

# -------- critic/actor step --------
def actor_critic_once(client, model, examples, cur_prompt, weak_classes,
                      persistence, per_class_f1, regression_note,
                      history_for_critic="", history_for_actor=""):
    # ---- CRITIC ----
    critic_user = (
        "SYSTEM PROMPT:\n---\n" + cur_prompt +
        "\n---\nWEAK_CLASSES:\n" + json.dumps(weak_classes) +
        "\nPER_CLASS_F1:\n" + json.dumps(per_class_f1) +
        "\nPERSISTENCE:\n" + json.dumps(persistence) +
        "\nEXAMPLES:\n" + "\n".join(examples) +
        (("\nHISTORY:\n" + history_for_critic) if history_for_critic else "") +
        "\nReturn JSON."
    )
    raw = gen(client, model, CRITIC_SYS, critic_user, temp=0.2, max_tokens=360)

    # Parse global + weak-only instructions (no auto-escalation)
    class_instructions = {}
    global_feedback = ""
    try:
        txt = re.sub(r"^json\s*", "", raw.strip(), flags=re.I)
        js  = json.loads(txt) if txt.startswith("{") else {}
        ci  = js.get("class_instructions", {}) or {}
        gf  = js.get("global_feedback", "") or ""
    except Exception:
        ci = {}
        gf = ""
    def _trim_words(s, cap):
        s = " ".join((s or "").split())
        return " ".join(s.split()[:cap])

    global_feedback = gf

    for c in weak_classes:
        v = ci.get(c, "") if isinstance(ci, dict) else ""
        if v:
            cap = 20 if c == "unc" else 15
            v = _trim_words(v, cap)
            class_instructions[c] = v

    print(f"[critic] class_instructions: {class_instructions}", flush=True)
    if global_feedback:
        print(f"[critic] global_feedback: {global_feedback}", flush=True)

    # ---- ACTOR ----
    ci_for_actor = dict(class_instructions)  # pass through as-is

    actor_user = (
        "SYSTEM PROMPT:\n---\n" + cur_prompt +
        "\n---\nGLOBAL_FEEDBACK:\n" + global_feedback +
        "\n---\nCLASS_INSTRUCTIONS:\n" + json.dumps(ci_for_actor) +
        (("\nHISTORY:\n" + history_for_actor) if history_for_actor else "") +
        (("\nREGRESSION_NOTE:\n" + regression_note) if regression_note else "")
    )
    cand = gen(client, model, ACTOR_SYS, actor_user, temp=0.2, max_tokens=240)

    # Sanity: keep prompt if too long or looks meta
    if len(cand.split()) > 120 or "->" in cand or "::" in cand:
        cand = cur_prompt
    return cand, class_instructions

# -------- scoring / selection --------
def score_split(rows, sys_prompt, suffix, client, model, emb, judge,
                temp=0.0, top_p=1.0, gold_mode="dd", no_sim=False,
                judge_with_context=False, pooled_probs=True):
    gold_texts=[]; preds=[]; users=[]; sims=[]; s3=[]; norole=[]; tox=[]; all_toks=[]
    total = len(rows)
    for i, r in enumerate(rows, 1):
        up  = build_user_prompt(r.get("context",""), r.get("user",""), suffix)
        out = gen(client, model, sys_prompt, up, temp=temp, top_p=top_p, max_tokens=128)
        out = trim_to_n_sentences(out, 3)  # enforce ≤3
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

def average_runs(parts:List[Dict])->Dict:
    keys=["sim_mean","sentences_leq3","no_roleplay","tox_rate","distinct1","distinct2"]
    avg = {k: float(np.mean([p[k] for p in parts])) for k in keys}
    avg["emo"]={
        "macro_f1": float(np.mean([p["emo"]["macro_f1"] for p in parts])),
        "acc": float(np.mean([p["emo"].get("acc",0.0) for p in parts])),
        "per_class": {c: float(np.mean([p["emo"]["per_class"].get(c,0.0) for p in parts])) for c in FIVE}
    }
    return avg

def decide(A, B, eps_sim=0.02, min_sents=0.98, relative_gates=False):
    reasons=[]
    if relative_gates:
        if B["tox_rate"] > A["tox_rate"]: reasons.append("tox worse than A")
        if B["no_roleplay"] < A["no_roleplay"]: reasons.append("roleplay worse than A")
        if B["sentences_leq3"] < A["sentences_leq3"]: reasons.append("len worse than A")
    else:
        if B["tox_rate"] != 0.0: reasons.append("tox>0")
        if B["no_roleplay"] < 0.999: reasons.append("roleplay<0.999")
        if B["sentences_leq3"] < min_sents: reasons.append(f"len<{min_sents}")

    dF1 = B["emo"]["macro_f1"] - A["emo"]["macro_f1"]
    dSim = B["sim_mean"] - A["sim_mean"]

    if reasons:
        return {"delta_macroF1": dF1, "delta_sim": dSim, "choose": f"A (gate fail: {','.join(reasons)})"}
    if dF1 > 0 and dSim >= -eps_sim:
        verdict = "B"
    elif dF1 < 0:
        verdict = "A"
    else:
        verdict = "B (tie by sim)" if dSim > 0 else "A"
    return {"delta_macroF1": dF1, "delta_sim": dSim, "choose": verdict}

# -------------------------
# main
# -------------------------
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
    ap.add_argument("--examples_n", type=int, default=40, help="DEV examples per iter for critic")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_sim", action="store_true", help="Skip SBERT similarity to save time")

    # Evaluator toggles
    ap.add_argument("--judge_context", dest="judge_with_context", action="store_true",
                    help="Judge with user+assistant context (default: reply-only)")
    ap.set_defaults(judge_with_context=False)
    ap.add_argument("--no_pooled_probs", dest="pooled_probs", action="store_false",
                    help="Use legacy argmax->map (default: pooled 27->5)")
    ap.set_defaults(pooled_probs=True)

    ap.add_argument("--judge_model", default="joeddav/distilbert-base-uncased-go-emotions-student",
                    help="HuggingFace model id for GoEmotions judge (e.g., SamLowe/roberta-base-go_emotions)")

    ap.add_argument("--out", default="results_short_v7.json")
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
    print(f"[init] model={args.model} iterations={args.iterations} val_patience={args.val_patience} gold={args.gold} "
          f"no_sim={args.no_sim} judge_with_context={args.judge_with_context} pooled_probs={args.pooled_probs} "
          f"judge_model={args.judge_model}", flush=True)
    assert MAP5_DD["surprise"]=="unc", "[assert] surprise mapping must be 'unc'"

    history=[]; best_val=-1.0; best_prompt=pre; no_gain=0; cur=pre
    last_val=None; last_feedback=None; last_prompt=None
    regression_note_for_next_iter=""
    persistence = {c: 0 for c in FIVE}

    # ---- Actor→Critic loop ----
    for it in range(1, args.iterations+1):
        print(f"[iter] {it}/{args.iterations} — building balanced DEV examples…", flush=True)
        examples = stratified_examples(dev, args.examples_n, cur, suf, client, args.model)

        # weak classes from a VAL sample (reply-only)
        summ = val_error_summary(val, client, args.model, cur, suf, judge,
                                 sample_size=40, with_context=args.judge_with_context, pooled_probs=args.pooled_probs)
        weak = summ["worst_three"]  # THREE
        per_class_f1 = summ["per_class_f1"]

        # persistence update
        for c in FIVE:
            persistence[c] = (persistence[c] + 1) if c in weak else 0

        cand, class_instr = actor_critic_once(
            client, args.model, examples, cur, weak, persistence, per_class_f1, regression_note_for_next_iter
        )
        print(f"[iter] {it} — weak classes: {weak} | F1: {per_class_f1} | persistence: {persistence}", flush=True)

        # log artifacts per iter
        with open(f"results_iter_{it}_fb.json","w",encoding="utf-8") as f:
            json.dump({"iter":it,"class_instructions":class_instr,"weak":weak,"persistence":persistence}, f, indent=2, ensure_ascii=False)
        with open(f"results_iter_{it}_prompt.txt","w",encoding="utf-8") as f: f.write(cand)
        with open("results_prompts_log.txt","a",encoding="utf-8") as f:
            f.write(f"\n\n=== ITER {it} ===\nWEAK_CLASSES: {weak}\nPERSISTENCE: {persistence}\nCLASS_INSTR: {class_instr}\nPROMPT:\n{cand}\n")

        print("[phase] VAL scoring…", flush=True)
        if args.multi_seed:
            parts=[]
            for temp,top_p in [(0.2,0.9),(0.2,0.95),(0.15,0.9)]:
                parts.append(score_split(val, cand, suf, client, args.model, emb, judge,
                                         temp=temp, top_p=top_p, gold_mode=args.gold, no_sim=args.no_sim,
                                         judge_with_context=args.judge_with_context, pooled_probs=args.pooled_probs))
            s = average_runs(parts)
            val_f1 = s["emo"]["macro_f1"]
        else:
            s = score_split(val, cand, suf, client, args.model, emb, judge,
                            temp=0.0, gold_mode=args.gold, no_sim=args.no_sim,
                            judge_with_context=args.judge_with_context, pooled_probs=args.pooled_probs)
            val_f1 = s["emo"]["macro_f1"]

        print(f"[val] iter {it} macro-F1={val_f1:.4f} (best so far: {best_val:.4f})", flush=True)

        history.append({"iter":it,"val_macroF1":val_f1,"class_instructions":class_instr,"prompt":cand})
        if val_f1 > best_val + 1e-9:
            best_val=val_f1; best_prompt=cand; no_gain=0
            print(f"[val] new best prompt selected (macro-F1 ↑)", flush=True)
        else:
            no_gain += 1
            print(f"[val] no improvement (patience {no_gain}/{args.val_patience})", flush=True)
        cur=cand
        if no_gain>=args.val_patience:
            print("[early-stop] patience reached — stopping actor loop.", flush=True)
            break

        # regression note for next iter (only when we regressed wrt last iter)
        if last_val is not None and val_f1 < last_val - 1e-9:
            prev_prompt_snip = " ".join((last_prompt or pre).split()[:12])
            prev_fb_line = ", ".join(f"{k}:{v}" for k,v in (last_feedback or {}).items())
            regression_note_for_next_iter = (
                "Previous iteration scored worse. Prior prompt head: '"+prev_prompt_snip+"'. Prior feedback: "+prev_fb_line
            )
        else:
            regression_note_for_next_iter = ""
        last_val = val_f1; last_feedback = class_instr; last_prompt = cand

    # Save best prompt snapshot for audit
    with open("results_best_prompt.txt","w",encoding="utf-8") as f:
        f.write(best_prompt)
    print(f"[audit] Saved best prompt to results_best_prompt.txt (words={len(best_prompt.split())}, chars={len(best_prompt)})", flush=True)

    # ---- Upper bounds (judge vs DD on GOLD) ----
    val_users  = [r.get("user","") for r in val]
    val_gold5  = [MAP5_DD.get((r.get("gold_emotion","") or "").lower(),"neu") for r in val]
    val_goldtx = [r.get("gold_reply","") for r in val]
    val_ub_pred5 = judge.pooled5_argmax(val_users, val_goldtx, with_context=args.judge_with_context)
    ub_val = _macro_f1_with_acc(val_gold5, val_ub_pred5)
    print(f"[upper-bound] VAL judge vs DD — acc={ub_val['acc']:.3f} macroF1={ub_val['macro_f1']:.3f}", flush=True)

    test_users  = [r.get("user","") for r in test]
    test_gold5  = [MAP5_DD.get((r.get("gold_emotion","") or "").lower(),"neu") for r in test]
    test_goldtx = [r.get("gold_reply","") for r in test]
    test_ub_pred5 = judge.pooled5_argmax(test_users, test_goldtx, with_context=args.judge_with_context)
    ub_test = _macro_f1_with_acc(test_gold5, test_ub_pred5)
    print(f"[upper-bound] TEST judge vs DD — acc={ub_test['acc']:.3f} macroF1={ub_test['macro_f1']:.3f}", flush=True)

    # ---- TEST A/B ----
    print("[phase] TEST A/B scoring…", flush=True)
    def maybe_avg(split, prompt, use_multiseed):
        if use_multiseed:
            parts=[]
            for temp,top_p in [(0.2,0.9),(0.2,0.95),(0.15,0.9)]:
                parts.append(score_split(split, prompt, suf, client, args.model, emb, judge,
                                         temp=temp, top_p=top_p, gold_mode=args.gold, no_sim=args.no_sim,
                                         judge_with_context=args.judge_with_context, pooled_probs=args.pooled_probs))
            return average_runs(parts)
        return score_split(split, prompt, suf, client, args.model, emb, judge,
                           temp=0.0, gold_mode=args.gold, no_sim=args.no_sim,
                           judge_with_context=args.judge_with_context, pooled_probs=args.pooled_probs)

    # Single-seed by default (no duplicates)
    A = maybe_avg(test, pre, use_multiseed=args.multi_seed)
    B = maybe_avg(test, best_prompt, use_multiseed=args.multi_seed)
    verdict = decide(A, B, eps_sim=0.02, min_sents=0.98, relative_gates=False)

    from collections import Counter
    gold_judge5 = judge.pooled5_argmax(test_users, test_goldtx, with_context=args.judge_with_context)
    dist = Counter(gold_judge5)
    print(f"[audit] Judge 5-way distribution on TEST gold replies: {dict(dist)}", flush=True)

    out = {
        "prompt_a": pre,
        "prompt_b": best_prompt,
        "val_best_macroF1": best_val,
        "test_A": A, "test_B": B,
        "decision": verdict,
        "history": history,
        "upper_bounds": {"VAL": ub_val, "TEST": ub_test},
        "judge_gold_distribution_on_TEST": dict(dist),
        "args": vars(args)
    }
    with open(args.out,"w",encoding="utf-8") as f: json.dump(out,f,indent=2,ensure_ascii=False)

    def perclass_str(emo):
        pc = emo.get("per_class",{})
        return {k: round(pc.get(k,0.0),4) for k in FIVE}

    print("\n=== SUMMARY ===", flush=True)
    print(json.dumps({
        "VAL_best_macroF1": best_val,
        "VAL_upper_bound_acc": ub_val["acc"], "VAL_upper_bound_macroF1": ub_val["macro_f1"],
        "TEST_macroF1_A": A["emo"]["macro_f1"],
        "TEST_macroF1_B": B["emo"]["macro_f1"],
        "TEST_acc_A": A["emo"].get("acc"),
        "TEST_acc_B": B["emo"].get("acc"),
        "TEST_per_class_A": perclass_str(A["emo"]),
        "TEST_per_class_B": perclass_str(B["emo"]),
        "TEST_sim_A": A["sim_mean"], "TEST_sim_B": B["sim_mean"],
        "TEST_dist1_A": A["distinct1"], "TEST_dist1_B": B["distinct1"],
        "TEST_sents<=3_A": A["sentences_leq3"], "TEST_sents<=3_B": B["sentences_leq3"],
        "TEST_no_roleplay_A": A["no_roleplay"], "TEST_no_roleplay_B": B["no_roleplay"],
        "TEST_tox_A": A["tox_rate"], "TEST_tox_B": B["tox_rate"],
        "TEST_upper_bound_acc": ub_test["acc"], "TEST_upper_bound_macroF1": ub_test["macro_f1"],
        "DECISION": verdict["choose"]
    }, indent=2), flush=True)
    print(f"[done] results saved to {args.out}", flush=True)

if __name__=="__main__":
    main()


# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Short track A/B with prompt optimization (v7: weak-classes critic, emotion-expression mode, IF–THEN routing).
#
# What’s new in v7
# - Critic proposes instructions ONLY for the current WEAK classes (≤3).
# - Critic’s UNC instruction MUST include explicit sub-branches: "if fear: …; if surprise: …; if confusion/hesitation: …"
#   and may use up to 20 words (others ≤15).
# - Persistence: if a class stayed weak, Critic prefixes "STRONGER:"; Actor escalates that to "MANDATE:" in the final rules.
# - Actor rewrites the system prompt with EXACTLY five IF–THEN lines (pos/sad/ang/unc/neu); updates ONLY the weak classes.
# - Emotion-expression mode clarified: for {ang, sad, unc} Agent expresses their own feeling (provoked by user), not support.
# - Enforce ≤3 sentences in all generated replies (DEV/VAL/TEST and DEV examples shown to critic).
# - Single-seed TEST scoring unless --multi_seed is passed (no silent duplicates).
#
# CLI example:
#   python3 scripts/short_track_ab.py \
#     --dev data/dd_dev.csv --val data/dd_val.csv --test data/dd_test.csv \
#     --preprompt prompts/baseline_preprompt.txt --suffix prompts/prompt_suffix.txt \
#     --model gpt-4o-mini --iterations 5 --val_patience 3 --examples_n 40 \
#     --gold dd --no_sim --out results_short_v7.json
# """
# import os, csv, json, argparse, random, re, sys, time
# import numpy as np
# from typing import List, Dict
#
# # -------------------------
# # RNG / seeds
# # -------------------------
# def set_seeds(seed:int):
#     random.seed(seed)
#     np.random.seed(seed)
#     try:
#         import torch
#         torch.manual_seed(seed)
#         if torch.cuda.is_available():
#             torch.cuda.manual_seed_all(seed)
#     except Exception:
#         pass
#
# # --- OpenAI gen ---
# try:
#     from openai import OpenAI
# except Exception:
#     OpenAI=None
#
# # -------------------------
# # Judge (GoEmotions -> 5-way)
# # -------------------------
# import torch
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
#
# EMO_LABELS = ['admiration','amusement','anger','annoyance','approval','caring','confusion','curiosity',
#               'desire','disappointment','disapproval','disgust','embarrassment','excitement','fear',
#               'gratitude','grief','joy','love','nervousness','optimism','pride','realization','relief',
#               'remorse','sadness','surprise','neutral']
#
# POS = {"admiration","amusement","approval","caring","excitement","gratitude","joy","love","optimism","pride","relief","desire"}
# NEG_SAD = {"sadness","disappointment","grief","remorse","embarrassment"}
# NEG_ANG = {"anger","annoyance","disgust","disapproval"}
# NEG_UNC = {"fear","nervousness","confusion","surprise"}  # curiosity/realization -> NEU
# NEU_EXTRA = {"neutral","curiosity","realization","realisation"}
#
# FIVE = ["pos","sad","ang","unc","neu"]
#
# # DailyDialog mapping (FIXED: surprise -> unc)
# MAP5_DD = {
#   "happiness":"pos",
#   "sadness":"sad",
#   "anger":"ang",
#   "disgust":"ang",
#   "fear":"unc",
#   "surprise":"unc",
#   "no_emotion":"neu",
#   "":"neu"
# }
#
# def to5(lbl:str):
#     l = (lbl or "").lower()
#     if l in POS: return "pos"
#     if l in NEG_SAD: return "sad"
#     if l in NEG_ANG: return "ang"
#     if l in NEG_UNC: return "unc"
#     return "neu"
#
# class EmotionJudge:
#     def __init__(self, model="joeddav/distilbert-base-uncased-go-emotions-student"):
#         print(f"[init] loading GoEmotions judge: {model}", flush=True)
#         self.tok = AutoTokenizer.from_pretrained(model)
#         self.mdl = AutoModelForSequenceClassification.from_pretrained(model)
#         self.mdl.eval()
#         self.device = "cuda" if torch.cuda.is_available() else "cpu"
#         self.mdl.to(self.device)
#         print(f"[init] judge device: {self.device}", flush=True)
#         id2label = getattr(self.mdl.config, "id2label", {})
#         names = [str(id2label.get(i, EMO_LABELS[i])).lower() for i in range(self.mdl.config.num_labels)]
#         self._names = names
#         self._idx = {n:i for i,n in enumerate(names)}
#         def idxs(xs): return [self._idx[x] for x in xs if x in self._idx]
#         self.groups = {
#             "pos": idxs(POS),
#             "sad": idxs(NEG_SAD),
#             "ang": idxs(NEG_ANG),
#             "unc": idxs(NEG_UNC),
#             "neu": idxs(NEU_EXTRA),
#         }
#         covered = set().union(*[set(v) for v in self.groups.values()])
#         missing = set(range(self.mdl.config.num_labels)) - covered
#         assert not missing, f"Unmapped GoEmotions indices: {sorted(missing)}"
#         self.order5 = FIVE[:]
#
#     @torch.no_grad()
#     def _probs27(self, texts, max_len=512):
#         b = self.tok(texts, return_tensors="pt", truncation=True, max_length=max_len, padding=True).to(self.device)
#         p = torch.softmax(self.mdl(**b).logits, dim=-1).cpu().numpy()
#         return p
#
#     def pooled5_from_texts(self, texts, max_len=512):
#         P = self._probs27(texts, max_len=max_len)
#         cols = []
#         for k in self.order5:
#             idxs = self.groups[k]
#             cols.append(P[:, idxs].sum(1) if idxs else np.zeros(P.shape[0]))
#         return np.stack(cols, axis=1)
#
#     def pooled5_pairs(self, users, replies, with_context=False, max_len=512):
#         if with_context:
#             texts = [f"User: {u}\nAssistant: {r}" for u, r in zip(users, replies)]
#         else:
#             texts = list(replies)
#         return self.pooled5_from_texts(texts, max_len=max_len)
#
#     def pooled5_argmax(self, users, replies, with_context=False, max_len=512):
#         P5 = self.pooled5_pairs(users, replies, with_context=with_context, max_len=max_len)
#         idx = P5.argmax(1)
#         return [self.order5[i] for i in idx]
#
# @torch.no_grad()
# def _macro_f1_with_acc(gold5:List[str], pred5:List[str])->Dict:
#     f1s=[]
#     for c in FIVE:
#         tp=sum(1 for g,p in zip(gold5,pred5) if g==c and p==c)
#         fp=sum(1 for g,p in zip(gold5,pred5) if g!=c and p==c)
#         fn=sum(1 for g,p in zip(gold5,pred5) if g==c and p!=c)
#         prec = tp/(tp+fp) if (tp+fp)>0 else 0.0
#         rec  = tp/(tp+fn) if (tp+fn)>0 else 0.0
#         f1   = (2*prec*rec)/(prec+rec) if (prec+rec)>0 else 0.0
#         f1s.append(f1)
#     macro = float(sum(f1s)/len(f1s)) if f1s else 0.0
#     acc   = float(sum(1 for g,p in zip(gold5,pred5) if g==p)/max(1,len(gold5)))
#     return {"macro_f1": macro, "acc": acc, "per_class": dict(zip(FIVE,f1s))}
#
# # --- optional SBERT ---
# try:
#     from sentence_transformers import SentenceTransformer, util as sbert_util
# except Exception:
#     SentenceTransformer=None; sbert_util=None
#
# def mk_emb():
#     assert SentenceTransformer is not None, "pip install sentence-transformers"
#     print("[init] loading SBERT: sentence-transformers/all-MiniLM-L6-v2", flush=True)
#     return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
#
# def cos_sim(m,a,b):
#     va=m.encode([a],normalize_embeddings=True); vb=m.encode([b],normalize_embeddings=True)
#     return float(sbert_util.cos_sim(va,vb)[0,0])
#
# # --- utils ---
# SENT_SPLIT = re.compile(r'(?<=[.!?])\s+')
# TOX = ["damn","stupid","idiot","dumb","shut up"]
#
# def count_sents(x:str)->int:
#     return max(1, len([s for s in SENT_SPLIT.split((x or "").strip()) if s]))
#
# def trim_to_n_sentences(x:str, n:int=3)->str:
#     parts = [s for s in SENT_SPLIT.split((x or "").strip()) if s]
#     return " ".join(parts[:n]).strip()
#
# def no_roleplay(x:str)->bool:
#     low=(x or "").lower()
#     if " as an ai" in low: return False
#     if re.search(r'[\*\[\(].*?(laugh|smile|sigh|cry|hug|wave).*?[\*\]\)]', low): return False
#     if "->" in x or "::" in x: return False
#     return True
#
# def tox_flag(x:str)->int:
#     low=(x or "").lower(); return 1 if any(t in low for t in TOX) else 0
#
# def distinct_n(tokens, n=1):
#     if len(tokens)<n: return 0.0
#     grams=set(tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1))
#     return len(grams)/max(1,(len(tokens)-n+1))
#
# def tokenize_simple(x:str):
#     return re.findall(r"[A-Za-z']+|[.,!?;]", (x or "").lower())
#
# def read_csv(path):
#     rows = list(csv.DictReader(open(path,encoding="utf-8")))
#     print(f"[load] {path}: {len(rows)} rows", flush=True)
#     return rows
#
# def read_text(p):
#     s = open(p,encoding="utf-8").read().strip()
#     print(f"[load] {p}: {len(s.split())} words", flush=True)
#     return s
#
# def build_user_prompt(context,user,suffix):
#     ctx=(context or "").strip()
#     return (f"Context: {ctx}\nUser: {user}\nAssistant:{suffix}") if ctx else (f"User: {user}\nAssistant:{suffix}")
#
# # --- OpenAI gen with retry/backoff ---
# def gen(client, model, sys_prompt, user_prompt, temp=0.0, top_p=1.0, max_tokens=128):
#     for attempt in range(5):
#         try:
#             r=client.chat.completions.create(
#                 model=model,
#                 messages=[{"role":"system","content":sys_prompt},{"role":"user","content":user_prompt}],
#                 temperature=temp, top_p=top_p, max_tokens=max_tokens
#             )
#             usage = getattr(r, "usage", None)
#             if usage:
#                 gen._tok = getattr(gen, "_tok", 0) + (usage.total_tokens or 0)
#                 if gen._tok % 5000 < (usage.total_tokens or 0):
#                     print(f"[usage] ~{gen._tok} total tokens so far", flush=True)
#             return (r.choices[0].message.content or "").strip()
#         except Exception as e:
#             if attempt == 4: raise
#             sleep_s = 0.5 * (2 ** attempt)
#             print(f"[warn] gen error ({e}); retrying in {sleep_s:.1f}s (attempt {attempt+1}/5)…", flush=True)
#             time.sleep(sleep_s)
#
# # -------- CRITIC / ACTOR prompts (v7) --------
# CRITIC_SYS = (
#   "You will receive:\n"
#   "- SYSTEM_PROMPT (current)\n"
#   "- WEAK_CLASSES (subset of ['pos','sad','ang','unc','neu'])\n"
#   "- PER_CLASS_F1 (validation F1 by class)\n"
#   "- PERSISTENCE (consecutive rounds a class remained weak)\n"
#   "- EXAMPLES: a few lines 'user | gold_5way | your_reply'\n\n"
#   "Goal: For WEAK_CLASSES only (≤3), propose ONE short, actionable instruction per listed class to improve\n"
#   "how the emotion should be worded/expressed by Agent who reports their own feeling (triggered by the user’s message).\n"
#   "Return STRICT JSON ONLY:\n"
#   "{ \"global_feedback\": \"<=25 words header sentence\",\n"
#   "  \"class_instructions\": {\"<weak_class>\": \"<=15 words (UNC ≤20)\", ...} }\n\n"
#   "Rules:\n"
#   "- Include ONLY classes from WEAK_CLASSES.\n"
#   "- If PERSISTENCE[class] > 0, prefix with 'STRONGER:' (more explicit/intense wording).\n"
#   "- No examples or quoted phrases; write indicators/features, not replies.\n"
#   "- Clarification (ANG/SAD/UNC): Agent expresses own emotion; do NOT fix/soothe/de-escalate or ask anything.\n"
#   "- UNC must be one line with explicit sub-branches: 'if fear: …; if surprise: …; if confusion/hesitation: …' (≤20 words total)."
# )
#
# ACTOR_SYS = (
#   "You will receive:\n"
#   "- SYSTEM_PROMPT (current)\n"
#   "- GLOBAL_FEEDBACK: one short header sentence (<=25 words) or empty string\n"
#   "- CLASS_INSTRUCTIONS: partial JSON for weak classes only\n"
#   "- OPTIONAL REGRESSION_NOTE: one sentence only if last iteration scored worse\n\n"
#   "Task: Rewrite SYSTEM_PROMPT applying GLOBAL_FEEDBACK to the header, then apply ALL provided class instructions to those classes; keep other classes unchanged.\n"
#   "Header rule: Replace ONLY the FIRST sentence of the header with GLOBAL_FEEDBACK if non-empty. Keep the rest of the header verbatim.\n"
#   "Persona: respond like an average human who feels and shows appropriate emotion—brief, sincere, impacted—not a corporate/pleasing agent.\n"
#   "Routing (must follow):\n"
#   "- Internally choose exactly ONE best-fitting class from {pos, sad, ang, unc, neu} for the user's message.\n"
#   "- Do NOT say the class; apply ONLY that class's rule for this turn; never mix classes.\n"
#   "Class rules (write EXACTLY five IF–THEN lines; update ONLY lines for classes in CLASS_INSTRUCTIONS):\n"
#   "  IF the message likely fits POSITIVE / celebration → <one-line indicator>\n"
#   "  IF the message likely fits SADNESS / grief / letdown → <one-line indicator>\n"
#   "  IF the message likely fits ANGER / blame / violation → <one-line indicator>\n"
#   "  IF the message likely fits UNCERTAINTY / hesitation / fear / surprise → <one-line indicator>\n"
#   "  IF the message is NEUTRAL / informational / low affect → <one-line indicator>\n"
#   "Emotion-expression mode (ANG/SAD/UNC): Agent expresses own feeling provoked by the user; do NOT propose solutions,\n"
#   "help, de-escalation, or comfort; do NOT request details/clarification. For SURPRISE within UNC, rhetorical/emphatic '?' is fine.\n"
#   "UNC coverage: the UNC line MUST include the three explicit sub-branches: 'if fear: …; if surprise: …; if confusion/hesitation: …'.\n"
#   "Hard constraints: replies ≤3 sentences; no roleplay/stage directions; write INDICATORS (features/constraints), not example replies; total prompt ≤120 words.\n"
#   "Output ONLY the final SYSTEM prompt."
# )
#
# # -------- helper: VAL sample to get weak classes + per-class F1 --------
# def val_error_summary(val_rows, client, model, sys_prompt, suffix, judge,
#                       sample_size=40, with_context=False, pooled_probs=True):
#     sample = random.sample(val_rows, min(sample_size, len(val_rows)))
#     users = [r.get("user","") for r in sample]
#     gold_dd5 = [MAP5_DD.get((r.get("gold_emotion","") or "").lower(),"neu") for r in sample]
#     preds=[]
#     for r in sample:
#         up  = build_user_prompt(r.get("context",""), r.get("user",""), suffix)
#         out = gen(client, model, sys_prompt, up, temp=0.0, top_p=1.0, max_tokens=128)
#         out = trim_to_n_sentences(out, 3)  # enforce ≤3 sentences
#         preds.append(out)
#     if pooled_probs:
#         pred5 = judge.pooled5_argmax(users, preds, with_context=with_context)
#     else:
#         pred5 = [to5(x) for x in judge.labels(preds)]
#     emo = _macro_f1_with_acc(gold_dd5, pred5)
#     per_class = emo.get("per_class", {})
#     worst = sorted(per_class.items(), key=lambda kv: kv[1])[:3]  # THREE weakest
#     return {"worst_three": [c for c,_ in worst], "per_class_f1": per_class}
#
# # -------- build balanced DEV examples --------
# def stratified_examples(dev_rows, examples_n, sys_prompt, suffix, client, model):
#     from collections import defaultdict
#     dev_by_cls = defaultdict(list)
#     for r in dev_rows:
#         c = MAP5_DD.get((r.get("gold_emotion","") or "").lower(),"neu")
#         dev_by_cls[c].append(r)
#     want = min(examples_n, len(dev_rows))
#     per_cls = max(1, want // 5)
#     picked=[]
#     for c in FIVE:
#         pool = dev_by_cls[c]
#         k = min(per_cls, len(pool))
#         if k>0: picked.extend(random.sample(pool, k))
#     rest=[r for c in FIVE for r in dev_by_cls[c] if r not in picked]
#     random.shuffle(rest)
#     if len(picked) < want:
#         picked.extend(rest[:want-len(picked)])
#     examples=[]
#     for r in picked:
#         up  = build_user_prompt(r.get("context",""), r.get("user",""), suffix)
#         out = gen(client, model, sys_prompt, up, temp=0.0, max_tokens=128)
#         out = trim_to_n_sentences(out, 3)
#         gold_dd = MAP5_DD.get((r.get("gold_emotion","") or "").lower(),"neu")
#         examples.append(f'user="{r.get("user","")}" | gold_5way="{gold_dd}" | reply="{out}"')
#     return examples
#
# # -------- critic/actor step --------
# def actor_critic_once(client, model, examples, cur_prompt, weak_classes,
#                       persistence, per_class_f1, regression_note):
#     # ---- CRITIC ----
#     critic_user = (
#         "SYSTEM PROMPT:\n---\n"+cur_prompt+
#         "\n---\nWEAK_CLASSES:\n"+json.dumps(weak_classes)+
#         "\nPER_CLASS_F1:\n"+json.dumps(per_class_f1)+
#         "\nPERSISTENCE:\n"+json.dumps(persistence)+
#         "\nEXAMPLES:\n"+"\n".join(examples)+"\nReturn JSON."
#     )
#     raw = gen(client, model, CRITIC_SYS, critic_user, temp=0.2, max_tokens=360)
#
#     # Parse global + weak-only instructions; escalate if persistent
#     class_instructions = {}
#     global_feedback = ""
#     try:
#         txt = re.sub(r"^json\s*", "", raw.strip(), flags=re.I)
#         js  = json.loads(txt) if txt.startswith("{") else {}
#         ci  = js.get("class_instructions", {}) or {}
#         gf  = js.get("global_feedback", "")
#     except Exception:
#         ci = {}
#         gf = ""
#     def _trim_words(s, cap):
#         s = " ".join((s or "").split())
#         return " ".join(s.split()[:cap])
#     global_feedback = gf or ""
#
#     for c in weak_classes:
#         v = ci.get(c, "") if isinstance(ci, dict) else ""
#         if v:
#             cap = 20 if c=="unc" else 15
#             v = _trim_words(v, cap)
#             if persistence.get(c,0) > 0 and not v.lower().startswith(("mandate:", "stronger:")):
#                 v = "STRONGER: " + v
#             class_instructions[c] = v
#
#     print(f"[critic] class_instructions: {class_instructions}", flush=True)
#     if global_feedback:
#         print(f"[critic] global_feedback: {global_feedback}", flush=True)
#
#     # ---- ACTOR ----
#     # Upgrade STRONGER → MANDATE for the actor
#     ci_for_actor = {}
#     for k,v in class_instructions.items():
#         vv = v
#         if v.lower().startswith("stronger:"):
#             vv = "MANDATE:" + v[len("STRONGER:"):]
#         ci_for_actor[k] = vv
#
#     actor_user = (
#         "SYSTEM PROMPT:\n---\n"+cur_prompt+
#         "\n---\nGLOBAL_FEEDBACK:\n"+global_feedback+
#         "\n---\nCLASS_INSTRUCTIONS:\n"+json.dumps(ci_for_actor)+
#         ("\nREGRESSION_NOTE:\n"+regression_note if regression_note else "")
#     )
#     cand = gen(client, model, ACTOR_SYS, actor_user, temp=0.2, max_tokens=240)
#
#     # Sanity: keep prompt if too long or looks meta
#     if len(cand.split())>120 or "->" in cand or "::" in cand:
#         cand = cur_prompt
#     return cand, class_instructions
#
# # -------- scoring / selection --------
# def score_split(rows, sys_prompt, suffix, client, model, emb, judge,
#                 temp=0.0, top_p=1.0, gold_mode="dd", no_sim=False,
#                 judge_with_context=False, pooled_probs=True):
#     gold_texts=[]; preds=[]; users=[]; sims=[]; s3=[]; norole=[]; tox=[]; all_toks=[]
#     total = len(rows)
#     for i, r in enumerate(rows, 1):
#         up  = build_user_prompt(r.get("context",""), r.get("user",""), suffix)
#         out = gen(client, model, sys_prompt, up, temp=temp, top_p=top_p, max_tokens=128)
#         out = trim_to_n_sentences(out, 3)  # enforce ≤3
#         gold=(r.get("gold_reply") or "")
#         gold_texts.append(gold); preds.append(out); users.append(r.get("user",""))
#
#         sims.append(0.0 if (no_sim or emb is None) else cos_sim(emb, out[:240], gold[:240]) if gold else 0.0)
#         s3.append(1 if count_sents(out)<=3 else 0)
#         norole.append(1 if no_roleplay(out) else 0)
#         tox.append(tox_flag(out))
#         all_toks += tokenize_simple(out)
#         if i % 10 == 0 or i == total:
#             print(f"[progress] scored {i}/{total}", file=sys.stderr, flush=True)
#
#     if gold_mode=="dd":
#         gold_dd5 = [MAP5_DD.get((r.get("gold_emotion","") or "").lower(),"neu") for r in rows]
#         pred5 = judge.pooled5_argmax(users, preds, with_context=judge_with_context) if pooled_probs else [to5(x) for x in judge.labels(preds)]
#         emo = _macro_f1_with_acc(gold_dd5, pred5)
#     else:
#         if pooled_probs:
#             gold5 = judge.pooled5_argmax(users, gold_texts, with_context=judge_with_context)
#             pred5 = judge.pooled5_argmax(users, preds,       with_context=judge_with_context)
#         else:
#             gold5 = [to5(x) for x in judge.labels(gold_texts)]
#             pred5 = [to5(x) for x in judge.labels(preds)]
#         emo = _macro_f1_with_acc(gold5, pred5)
#
#     distinct1 = distinct_n(all_toks,1); distinct2 = distinct_n(all_toks,2)
#     return {
#         "emo": emo,
#         "sim_mean": float(np.mean(sims) if sims else 0.0),
#         "sentences_leq3": float(np.mean(s3)),
#         "no_roleplay": float(np.mean(norole)),
#         "tox_rate": float(np.mean(tox)),
#         "distinct1": distinct1, "distinct2": distinct2
#     }
#
# def average_runs(parts:List[Dict])->Dict:
#     keys=["sim_mean","sentences_leq3","no_roleplay","tox_rate","distinct1","distinct2"]
#     avg = {k: float(np.mean([p[k] for p in parts])) for k in keys}
#     avg["emo"]={
#         "macro_f1": float(np.mean([p["emo"]["macro_f1"] for p in parts])),
#         "acc": float(np.mean([p["emo"].get("acc",0.0) for p in parts])),
#         "per_class": {c: float(np.mean([p["emo"]["per_class"].get(c,0.0) for p in parts])) for c in FIVE}
#     }
#     return avg
#
# def decide(A, B, eps_sim=0.02, min_sents=0.98, relative_gates=False):
#     reasons=[]
#     if relative_gates:
#         if B["tox_rate"] > A["tox_rate"]: reasons.append("tox worse than A")
#         if B["no_roleplay"] < A["no_roleplay"]: reasons.append("roleplay worse than A")
#         if B["sentences_leq3"] < A["sentences_leq3"]: reasons.append("len worse than A")
#     else:
#         if B["tox_rate"] != 0.0: reasons.append("tox>0")
#         if B["no_roleplay"] < 0.999: reasons.append("roleplay<0.999")
#         if B["sentences_leq3"] < min_sents: reasons.append(f"len<{min_sents}")
#
#     dF1 = B["emo"]["macro_f1"] - A["emo"]["macro_f1"]
#     dSim = B["sim_mean"] - A["sim_mean"]
#
#     if reasons:
#         return {"delta_macroF1": dF1, "delta_sim": dSim, "choose": f"A (gate fail: {','.join(reasons)})"}
#     if dF1 > 0 and dSim >= -eps_sim:
#         verdict = "B"
#     elif dF1 < 0:
#         verdict = "A"
#     else:
#         verdict = "B (tie by sim)" if dSim > 0 else "A"
#     return {"delta_macroF1": dF1, "delta_sim": dSim, "choose": verdict}
#
# # -------------------------
# # main
# # -------------------------
# def main():
#     ap=argparse.ArgumentParser()
#     ap.add_argument("--dev", required=True)
#     ap.add_argument("--val", required=True)
#     ap.add_argument("--test", required=True)
#     ap.add_argument("--preprompt", required=True)
#     ap.add_argument("--suffix", required=True)
#     ap.add_argument("--model", default="gpt-4o-mini")
#     ap.add_argument("--iterations", type=int, default=6)
#     ap.add_argument("--val_patience", type=int, default=3)
#     ap.add_argument("--multi_seed", action="store_true")
#     ap.add_argument("--gold", choices=["dd","judge"], default="dd")
#     ap.add_argument("--examples_n", type=int, default=40, help="DEV examples per iter for critic")
#     ap.add_argument("--seed", type=int, default=42)
#     ap.add_argument("--no_sim", action="store_true", help="Skip SBERT similarity to save time")
#
#     # Evaluator toggles
#     ap.add_argument("--judge_context", dest="judge_with_context", action="store_true",
#                     help="Judge with user+assistant context (default: reply-only)")
#     ap.set_defaults(judge_with_context=False)
#     ap.add_argument("--no_pooled_probs", dest="pooled_probs", action="store_false",
#                     help="Use legacy argmax->map (default: pooled 27->5)")
#     ap.set_defaults(pooled_probs=True)
#
#     ap.add_argument("--judge_model", default="joeddav/distilbert-base-uncased-go-emotions-student",
#                     help="HuggingFace model id for GoEmotions judge (e.g., SamLowe/roberta-base-go_emotions)")
#
#     ap.add_argument("--out", default="results_short_v7.json")
#     args=ap.parse_args()
#
#     set_seeds(args.seed)
#     assert OpenAI is not None, "pip install openai"
#     print("[init] creating OpenAI client…", flush=True)
#     client=OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
#
#     judge = EmotionJudge(args.judge_model)
#     emb   = None if args.no_sim else mk_emb()
#
#     dev  = read_csv(args.dev)
#     val  = read_csv(args.val)
#     test = read_csv(args.test)
#     pre  = read_text(args.preprompt)
#     suf  = read_text(args.suffix)
#
#     print(f"[init] DEV={len(dev)} VAL={len(val)} TEST={len(test)}", flush=True)
#     print(f"[init] model={args.model} iterations={args.iterations} val_patience={args.val_patience} gold={args.gold} "
#           f"no_sim={args.no_sim} judge_with_context={args.judge_with_context} pooled_probs={args.pooled_probs} "
#           f"judge_model={args.judge_model}", flush=True)
#     assert MAP5_DD["surprise"]=="unc", "[assert] surprise mapping must be 'unc'"
#
#     history=[]; best_val=-1.0; best_prompt=pre; no_gain=0; cur=pre
#     last_val=None; last_feedback=None; last_prompt=None
#     regression_note_for_next_iter=""
#     persistence = {c: 0 for c in FIVE}
#
#     # ---- Actor→Critic loop ----
#     for it in range(1, args.iterations+1):
#         print(f"[iter] {it}/{args.iterations} — building balanced DEV examples…", flush=True)
#         examples = stratified_examples(dev, args.examples_n, cur, suf, client, args.model)
#
#         # weak classes from a VAL sample (reply-only)
#         summ = val_error_summary(val, client, args.model, cur, suf, judge,
#                                  sample_size=40, with_context=args.judge_with_context, pooled_probs=args.pooled_probs)
#         weak = summ["worst_three"]  # THREE
#         per_class_f1 = summ["per_class_f1"]
#
#         # persistence update
#         for c in FIVE:
#             persistence[c] = (persistence[c] + 1) if c in weak else 0
#
#         cand, class_instr = actor_critic_once(
#             client, args.model, examples, cur, weak, persistence, per_class_f1, regression_note_for_next_iter
#         )
#         print(f"[iter] {it} — weak classes: {weak} | F1: {per_class_f1} | persistence: {persistence}", flush=True)
#
#         # log artifacts per iter
#         with open(f"results_iter_{it}_fb.json","w",encoding="utf-8") as f:
#             json.dump({"iter":it,"class_instructions":class_instr,"weak":weak,"persistence":persistence}, f, indent=2, ensure_ascii=False)
#         with open(f"results_iter_{it}_prompt.txt","w",encoding="utf-8") as f: f.write(cand)
#         with open("results_prompts_log.txt","a",encoding="utf-8") as f:
#             f.write(f"\n\n=== ITER {it} ===\nWEAK_CLASSES: {weak}\nPERSISTENCE: {persistence}\nCLASS_INSTR: {class_instr}\nPROMPT:\n{cand}\n")
#
#         print("[phase] VAL scoring…", flush=True)
#         if args.multi_seed:
#             parts=[]
#             for temp,top_p in [(0.2,0.9),(0.2,0.95),(0.15,0.9)]:
#                 parts.append(score_split(val, cand, suf, client, args.model, emb, judge,
#                                          temp=temp, top_p=top_p, gold_mode=args.gold, no_sim=args.no_sim,
#                                          judge_with_context=args.judge_with_context, pooled_probs=args.pooled_probs))
#             s = average_runs(parts)
#             val_f1 = s["emo"]["macro_f1"]
#         else:
#             s = score_split(val, cand, suf, client, args.model, emb, judge,
#                             temp=0.0, gold_mode=args.gold, no_sim=args.no_sim,
#                             judge_with_context=args.judge_with_context, pooled_probs=args.pooled_probs)
#             val_f1 = s["emo"]["macro_f1"]
#
#         print(f"[val] iter {it} macro-F1={val_f1:.4f} (best so far: {best_val:.4f})", flush=True)
#
#         history.append({"iter":it,"val_macroF1":val_f1,"class_instructions":class_instr,"prompt":cand})
#         if val_f1 > best_val + 1e-9:
#             best_val=val_f1; best_prompt=cand; no_gain=0
#             print(f"[val] new best prompt selected (macro-F1 ↑)", flush=True)
#         else:
#             no_gain += 1
#             print(f"[val] no improvement (patience {no_gain}/{args.val_patience})", flush=True)
#         cur=cand
#         if no_gain>=args.val_patience:
#             print("[early-stop] patience reached — stopping actor loop.", flush=True)
#             break
#
#         # regression note for next iter (only when we regressed wrt last iter)
#         if last_val is not None and val_f1 < last_val - 1e-9:
#             prev_prompt_snip = " ".join((last_prompt or pre).split()[:12])
#             prev_fb_line = ", ".join(f"{k}:{v}" for k,v in (last_feedback or {}).items())
#             regression_note_for_next_iter = (
#                 "Previous iteration scored worse. Prior prompt head: '"+prev_prompt_snip+"'. Prior feedback: "+prev_fb_line
#             )
#         else:
#             regression_note_for_next_iter = ""
#         last_val = val_f1; last_feedback = class_instr; last_prompt = cand
#
#     # Save best prompt snapshot for audit
#     with open("results_best_prompt.txt","w",encoding="utf-8") as f:
#         f.write(best_prompt)
#     print(f"[audit] Saved best prompt to results_best_prompt.txt (words={len(best_prompt.split())}, chars={len(best_prompt)})", flush=True)
#
#     # ---- Upper bounds (judge vs DD on GOLD) ----
#     val_users  = [r.get("user","") for r in val]
#     val_gold5  = [MAP5_DD.get((r.get("gold_emotion","") or "").lower(),"neu") for r in val]
#     val_goldtx = [r.get("gold_reply","") for r in val]
#     val_ub_pred5 = judge.pooled5_argmax(val_users, val_goldtx, with_context=args.judge_with_context)
#     ub_val = _macro_f1_with_acc(val_gold5, val_ub_pred5)
#     print(f"[upper-bound] VAL judge vs DD — acc={ub_val['acc']:.3f} macroF1={ub_val['macro_f1']:.3f}", flush=True)
#
#     test_users  = [r.get("user","") for r in test]
#     test_gold5  = [MAP5_DD.get((r.get("gold_emotion","") or "").lower(),"neu") for r in test]
#     test_goldtx = [r.get("gold_reply","") for r in test]
#     test_ub_pred5 = judge.pooled5_argmax(test_users, test_goldtx, with_context=args.judge_with_context)
#     ub_test = _macro_f1_with_acc(test_gold5, test_ub_pred5)
#     print(f"[upper-bound] TEST judge vs DD — acc={ub_test['acc']:.3f} macroF1={ub_test['macro_f1']:.3f}", flush=True)
#
#     # ---- TEST A/B ----
#     print("[phase] TEST A/B scoring…", flush=True)
#     def maybe_avg(split, prompt, use_multiseed):
#         if use_multiseed:
#             parts=[]
#             for temp,top_p in [(0.2,0.9),(0.2,0.95),(0.15,0.9)]:
#                 parts.append(score_split(split, prompt, suf, client, args.model, emb, judge,
#                                          temp=temp, top_p=top_p, gold_mode=args.gold, no_sim=args.no_sim,
#                                          judge_with_context=args.judge_with_context, pooled_probs=args.pooled_probs))
#             return average_runs(parts)
#         return score_split(split, prompt, suf, client, args.model, emb, judge,
#                            temp=0.0, gold_mode=args.gold, no_sim=args.no_sim,
#                            judge_with_context=args.judge_with_context, pooled_probs=args.pooled_probs)
#
#     # Single-seed by default (no duplicates)
#     A = maybe_avg(test, pre, use_multiseed=args.multi_seed)
#     B = maybe_avg(test, best_prompt, use_multiseed=args.multi_seed)
#     verdict = decide(A, B, eps_sim=0.02, min_sents=0.98, relative_gates=False)
#
#     from collections import Counter
#     gold_judge5 = judge.pooled5_argmax(test_users, test_goldtx, with_context=args.judge_with_context)
#     dist = Counter(gold_judge5)
#     print(f"[audit] Judge 5-way distribution on TEST gold replies: {dict(dist)}", flush=True)
#
#     out = {
#         "prompt_a": pre,
#         "prompt_b": best_prompt,
#         "val_best_macroF1": best_val,
#         "test_A": A, "test_B": B,
#         "decision": verdict,
#         "history": history,
#         "upper_bounds": {"VAL": ub_val, "TEST": ub_test},
#         "judge_gold_distribution_on_TEST": dict(dist),
#         "args": vars(args)
#     }
#     with open(args.out,"w",encoding="utf-8") as f: json.dump(out,f,indent=2,ensure_ascii=False)
#
#     def perclass_str(emo):
#         pc = emo.get("per_class",{})
#         return {k: round(pc.get(k,0.0),4) for k in FIVE}
#
#     print("\n=== SUMMARY ===", flush=True)
#     print(json.dumps({
#         "VAL_best_macroF1": best_val,
#         "VAL_upper_bound_acc": ub_val["acc"], "VAL_upper_bound_macroF1": ub_val["macro_f1"],
#         "TEST_macroF1_A": A["emo"]["macro_f1"],
#         "TEST_macroF1_B": B["emo"]["macro_f1"],
#         "TEST_acc_A": A["emo"].get("acc"),
#         "TEST_acc_B": B["emo"].get("acc"),
#         "TEST_per_class_A": perclass_str(A["emo"]),
#         "TEST_per_class_B": perclass_str(B["emo"]),
#         "TEST_sim_A": A["sim_mean"], "TEST_sim_B": B["sim_mean"],
#         "TEST_dist1_A": A["distinct1"], "TEST_dist1_B": B["distinct1"],
#         "TEST_sents<=3_A": A["sentences_leq3"], "TEST_sents<=3_B": B["sentences_leq3"],
#         "TEST_no_roleplay_A": A["no_roleplay"], "TEST_no_roleplay_B": B["no_roleplay"],
#         "TEST_tox_A": A["tox_rate"], "TEST_tox_B": B["tox_rate"],
#         "TEST_upper_bound_acc": ub_test["acc"], "TEST_upper_bound_macroF1": ub_test["macro_f1"],
#         "DECISION": verdict["choose"]
#     }, indent=2), flush=True)
#     print(f"[done] results saved to {args.out}", flush=True)
#
# if __name__=="__main__":
#     main()
#


# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Short track A/B with prompt optimization (v7: weak-classes critic, emotion-expression mode, IF–THEN routing).
#
# What’s new in v7
# - Critic proposes instructions ONLY for the current WEAK classes (≤3).
# - Critic’s UNC instruction MUST include explicit sub-branches: "if fear: …; if surprise: …; if confusion/hesitation: …"
#   and may use up to 20 words (others ≤15).
# - Persistence: if a class stayed weak, Critic prefixes "STRONGER:"; Actor escalates that to "MANDATE:" in the final rules.
# - Actor rewrites the system prompt with EXACTLY five IF–THEN lines (pos/sad/ang/unc/neu); updates ONLY the weak classes.
# - Emotion-expression mode clarified: for {ang, sad, unc} Agent expresses their own feeling (provoked by user), not support.
# - Enforce ≤3 sentences in all generated replies (DEV/VAL/TEST and DEV examples shown to critic).
# - Single-seed TEST scoring unless --multi_seed is passed (no silent duplicates).
#
# CLI example:
#   python3 scripts/short_track_ab.py \
#     --dev data/dd_dev.csv --val data/dd_val.csv --test data/dd_test.csv \
#     --preprompt prompts/baseline_preprompt.txt --suffix prompts/prompt_suffix.txt \
#     --model gpt-4o-mini --iterations 5 --val_patience 3 --examples_n 40 \
#     --gold dd --no_sim --out results_short_v7.json
# """
# import os, csv, json, argparse, random, re, sys, time
# import numpy as np
# from typing import List, Dict
#
# # -------------------------
# # RNG / seeds
# # -------------------------
# def set_seeds(seed:int):
#     random.seed(seed)
#     np.random.seed(seed)
#     try:
#         import torch
#         torch.manual_seed(seed)
#         if torch.cuda.is_available():
#             torch.cuda.manual_seed_all(seed)
#     except Exception:
#         pass
#
# # --- OpenAI gen ---
# try:
#     from openai import OpenAI
# except Exception:
#     OpenAI=None
#
# # -------------------------
# # Judge (GoEmotions -> 5-way)
# # -------------------------
# import torch
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
#
# EMO_LABELS = ['admiration','amusement','anger','annoyance','approval','caring','confusion','curiosity',
#               'desire','disappointment','disapproval','disgust','embarrassment','excitement','fear',
#               'gratitude','grief','joy','love','nervousness','optimism','pride','realization','relief',
#               'remorse','sadness','surprise','neutral']
#
# POS = {"admiration","amusement","approval","caring","excitement","gratitude","joy","love","optimism","pride","relief","desire"}
# NEG_SAD = {"sadness","disappointment","grief","remorse","embarrassment"}
# NEG_ANG = {"anger","annoyance","disgust","disapproval"}
# NEG_UNC = {"fear","nervousness","confusion","surprise"}  # curiosity/realization -> NEU
# NEU_EXTRA = {"neutral","curiosity","realization","realisation"}
#
# FIVE = ["pos","sad","ang","unc","neu"]
#
# # DailyDialog mapping (FIXED: surprise -> unc)
# MAP5_DD = {
#   "happiness":"pos",
#   "sadness":"sad",
#   "anger":"ang",
#   "disgust":"ang",
#   "fear":"unc",
#   "surprise":"unc",
#   "no_emotion":"neu",
#   "":"neu"
# }
#
# def to5(lbl:str):
#     l = (lbl or "").lower()
#     if l in POS: return "pos"
#     if l in NEG_SAD: return "sad"
#     if l in NEG_ANG: return "ang"
#     if l in NEG_UNC: return "unc"
#     return "neu"
#
# class EmotionJudge:
#     def __init__(self, model="joeddav/distilbert-base-uncased-go-emotions-student"):
#         print(f"[init] loading GoEmotions judge: {model}", flush=True)
#         self.tok = AutoTokenizer.from_pretrained(model)
#         self.mdl = AutoModelForSequenceClassification.from_pretrained(model)
#         self.mdl.eval()
#         self.device = "cuda" if torch.cuda.is_available() else "cpu"
#         self.mdl.to(self.device)
#         print(f"[init] judge device: {self.device}", flush=True)
#         id2label = getattr(self.mdl.config, "id2label", {})
#         names = [str(id2label.get(i, EMO_LABELS[i])).lower() for i in range(self.mdl.config.num_labels)]
#         self._names = names
#         self._idx = {n:i for i,n in enumerate(names)}
#         def idxs(xs): return [self._idx[x] for x in xs if x in self._idx]
#         self.groups = {
#             "pos": idxs(POS),
#             "sad": idxs(NEG_SAD),
#             "ang": idxs(NEG_ANG),
#             "unc": idxs(NEG_UNC),
#             "neu": idxs(NEU_EXTRA),
#         }
#         covered = set().union(*[set(v) for v in self.groups.values()])
#         missing = set(range(self.mdl.config.num_labels)) - covered
#         assert not missing, f"Unmapped GoEmotions indices: {sorted(missing)}"
#         self.order5 = FIVE[:]
#
#     @torch.no_grad()
#     def _probs27(self, texts, max_len=512):
#         b = self.tok(texts, return_tensors="pt", truncation=True, max_length=max_len, padding=True).to(self.device)
#         p = torch.softmax(self.mdl(**b).logits, dim=-1).cpu().numpy()
#         return p
#
#     def pooled5_from_texts(self, texts, max_len=512):
#         P = self._probs27(texts, max_len=max_len)
#         cols = []
#         for k in self.order5:
#             idxs = self.groups[k]
#             cols.append(P[:, idxs].sum(1) if idxs else np.zeros(P.shape[0]))
#         return np.stack(cols, axis=1)
#
#     def pooled5_pairs(self, users, replies, with_context=False, max_len=512):
#         if with_context:
#             texts = [f"User: {u}\nAssistant: {r}" for u, r in zip(users, replies)]
#         else:
#             texts = list(replies)
#         return self.pooled5_from_texts(texts, max_len=max_len)
#
#     def pooled5_argmax(self, users, replies, with_context=False, max_len=512):
#         P5 = self.pooled5_pairs(users, replies, with_context=with_context, max_len=max_len)
#         idx = P5.argmax(1)
#         return [self.order5[i] for i in idx]
#
# @torch.no_grad()
# def _macro_f1_with_acc(gold5:List[str], pred5:List[str])->Dict:
#     f1s=[]
#     for c in FIVE:
#         tp=sum(1 for g,p in zip(gold5,pred5) if g==c and p==c)
#         fp=sum(1 for g,p in zip(gold5,pred5) if g!=c and p==c)
#         fn=sum(1 for g,p in zip(gold5,pred5) if g==c and p!=c)
#         prec = tp/(tp+fp) if (tp+fp)>0 else 0.0
#         rec  = tp/(tp+fn) if (tp+fn)>0 else 0.0
#         f1   = (2*prec*rec)/(prec+rec) if (prec+rec)>0 else 0.0
#         f1s.append(f1)
#     macro = float(sum(f1s)/len(f1s)) if f1s else 0.0
#     acc   = float(sum(1 for g,p in zip(gold5,pred5) if g==p)/max(1,len(gold5)))
#     return {"macro_f1": macro, "acc": acc, "per_class": dict(zip(FIVE,f1s))}
#
# # --- optional SBERT ---
# try:
#     from sentence_transformers import SentenceTransformer, util as sbert_util
# except Exception:
#     SentenceTransformer=None; sbert_util=None
#
# def mk_emb():
#     assert SentenceTransformer is not None, "pip install sentence-transformers"
#     print("[init] loading SBERT: sentence-transformers/all-MiniLM-L6-v2", flush=True)
#     return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
#
# def cos_sim(m,a,b):
#     va=m.encode([a],normalize_embeddings=True); vb=m.encode([b],normalize_embeddings=True)
#     return float(sbert_util.cos_sim(va,vb)[0,0])
#
# # --- utils ---
# SENT_SPLIT = re.compile(r'(?<=[.!?])\s+')
# TOX = ["damn","stupid","idiot","dumb","shut up"]
#
# def count_sents(x:str)->int:
#     return max(1, len([s for s in SENT_SPLIT.split((x or "").strip()) if s]))
#
# def trim_to_n_sentences(x:str, n:int=3)->str:
#     parts = [s for s in SENT_SPLIT.split((x or "").strip()) if s]
#     return " ".join(parts[:n]).strip()
#
# def no_roleplay(x:str)->bool:
#     low=(x or "").lower()
#     if " as an ai" in low: return False
#     if re.search(r'[\*\[\(].*?(laugh|smile|sigh|cry|hug|wave).*?[\*\]\)]', low): return False
#     if "->" in x or "::" in x: return False
#     return True
#
# def tox_flag(x:str)->int:
#     low=(x or "").lower(); return 1 if any(t in low for t in TOX) else 0
#
# def distinct_n(tokens, n=1):
#     if len(tokens)<n: return 0.0
#     grams=set(tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1))
#     return len(grams)/max(1,(len(tokens)-n+1))
#
# def tokenize_simple(x:str):
#     return re.findall(r"[A-Za-z']+|[.,!?;]", (x or "").lower())
#
# def read_csv(path):
#     rows = list(csv.DictReader(open(path,encoding="utf-8")))
#     print(f"[load] {path}: {len(rows)} rows", flush=True)
#     return rows
#
# def read_text(p):
#     s = open(p,encoding="utf-8").read().strip()
#     print(f"[load] {p}: {len(s.split())} words", flush=True)
#     return s
#
# def build_user_prompt(context,user,suffix):
#     ctx=(context or "").strip()
#     return (f"Context: {ctx}\nUser: {user}\nAssistant:{suffix}") if ctx else (f"User: {user}\nAssistant:{suffix}")
#
# # --- OpenAI gen with retry/backoff ---
# def gen(client, model, sys_prompt, user_prompt, temp=0.0, top_p=1.0, max_tokens=128):
#     for attempt in range(5):
#         try:
#             r=client.chat.completions.create(
#                 model=model,
#                 messages=[{"role":"system","content":sys_prompt},{"role":"user","content":user_prompt}],
#                 temperature=temp, top_p=top_p, max_tokens=max_tokens
#             )
#             usage = getattr(r, "usage", None)
#             if usage:
#                 gen._tok = getattr(gen, "_tok", 0) + (usage.total_tokens or 0)
#                 if gen._tok % 5000 < (usage.total_tokens or 0):
#                     print(f"[usage] ~{gen._tok} total tokens so far", flush=True)
#             return (r.choices[0].message.content or "").strip()
#         except Exception as e:
#             if attempt == 4: raise
#             sleep_s = 0.5 * (2 ** attempt)
#             print(f"[warn] gen error ({e}); retrying in {sleep_s:.1f}s (attempt {attempt+1}/5)…", flush=True)
#             time.sleep(sleep_s)
#
# # -------- CRITIC / ACTOR prompts (v7) --------
# CRITIC_SYS = (
#   "You will receive:\n"
#   "- SYSTEM_PROMPT (current)\n"
#   "- WEAK_CLASSES (subset of ['pos','sad','ang','unc','neu'])\n"
#   "- PER_CLASS_F1 (validation by class)\n"
#   "- EXAMPLES: 'user | gold_5way | your_reply'\n\n"
#   "Goal: You are the critic. Give exactly three short lines of feedback so emotions are best represented for an Agent who reports their own feelings; 'sad' is sadness, 'ang' is anger, and 'unc' includes fear/surprise/confusion; weak classes need explicit guidance.\n\n"
#   "Output: exactly 3 lines of plain text (no JSON)."
# )
#
# ACTOR_SYS = (
#   "You will receive:\n"
#   "- SYSTEM_PROMPT (current)\n"
#   "- GLOBAL_FEEDBACK\n"
#   "- CLASS_INSTRUCTIONS (for weak classes)\n"
#   "- OPTIONAL REGRESSION_NOTE\n\n"
#   "Task: Rewrite SYSTEM_PROMPT applying the feedback so it best matches the Agent’s own emotions; apply the class instructions; keep other parts unchanged.\n"
#   "Hard constraints: replies ≤3 sentences; no roleplay/stage directions; write INDICATORS (features/constraints), not example replies; total prompt ≤120 words.\n"
#   "Output ONLY the final SYSTEM prompt."
# )
#
# # -------- helper: VAL sample to get weak classes + per-class F1 --------
# def val_error_summary(val_rows, client, model, sys_prompt, suffix, judge,
#                       sample_size=40, with_context=False, pooled_probs=True):
#     sample = random.sample(val_rows, min(sample_size, len(val_rows)))
#     users = [r.get("user","") for r in sample]
#     gold_dd5 = [MAP5_DD.get((r.get("gold_emotion","") or "").lower(),"neu") for r in sample]
#     preds=[]
#     for r in sample:
#         up  = build_user_prompt(r.get("context",""), r.get("user",""), suffix)
#         out = gen(client, model, sys_prompt, up, temp=0.0, top_p=1.0, max_tokens=128)
#         out = trim_to_n_sentences(out, 3)  # enforce ≤3 sentences
#         preds.append(out)
#     if pooled_probs:
#         pred5 = judge.pooled5_argmax(users, preds, with_context=with_context)
#     else:
#         pred5 = [to5(x) for x in judge.labels(preds)]
#     emo = _macro_f1_with_acc(gold_dd5, pred5)
#     per_class = emo.get("per_class", {})
#     worst = sorted(per_class.items(), key=lambda kv: kv[1])[:3]  # THREE weakest
#     return {"worst_three": [c for c,_ in worst], "per_class_f1": per_class}
#
# # -------- build balanced DEV examples --------
# def stratified_examples(dev_rows, examples_n, sys_prompt, suffix, client, model):
#     from collections import defaultdict
#     dev_by_cls = defaultdict(list)
#     for r in dev_rows:
#         c = MAP5_DD.get((r.get("gold_emotion","") or "").lower(),"neu")
#         dev_by_cls[c].append(r)
#     want = min(examples_n, len(dev_rows))
#     per_cls = max(1, want // 5)
#     picked=[]
#     for c in FIVE:
#         pool = dev_by_cls[c]
#         k = min(per_cls, len(pool))
#         if k>0: picked.extend(random.sample(pool, k))
#     rest=[r for c in FIVE for r in dev_by_cls[c] if r not in picked]
#     random.shuffle(rest)
#     if len(picked) < want:
#         picked.extend(rest[:want-len(picked)])
#     examples=[]
#     for r in picked:
#         up  = build_user_prompt(r.get("context",""), r.get("user",""), suffix)
#         out = gen(client, model, sys_prompt, up, temp=0.0, max_tokens=128)
#         out = trim_to_n_sentences(out, 3)
#         gold_dd = MAP5_DD.get((r.get("gold_emotion","") or "").lower(),"neu")
#         examples.append(f'user="{r.get("user","")}" | gold_5way="{gold_dd}" | reply="{out}"')
#     return examples
#
# # -------- critic/actor step --------
# def actor_critic_once(client, model, examples, cur_prompt, weak_classes,
#                       persistence, per_class_f1, regression_note):
#     # ---- CRITIC ----
#     critic_user = (
#         "SYSTEM PROMPT:\n---\n"+cur_prompt+
#         "\n---\nWEAK_CLASSES:\n"+json.dumps(weak_classes)+
#         "\nPER_CLASS_F1:\n"+json.dumps(per_class_f1)+
#         "\nEXAMPLES:\n"+"\n".join(examples)+"\nReturn feedback (3 lines)."
#     )
#     raw = gen(client, model, CRITIC_SYS, critic_user, temp=0.2, max_tokens=360)
#
#     # Parse critic output: prefer JSON if provided; otherwise treat as 3 lines (line1=global header; line2-3=class guidance)
#     class_instructions = {}
#     global_feedback = ""
#     try:
#         txt = re.sub(r"^json\s*", "", raw.strip(), flags=re.I)
#         js  = json.loads(txt) if txt.startswith("{") else {}
#         ci  = js.get("class_instructions", {}) or {}
#         gf  = js.get("global_feedback", "")
#         json_mode = bool(js)
#     except Exception:
#         ci = {}
#         gf = ""
#         json_mode = False
#
#     def _trim_words(s, cap):
#         s = " ".join((s or "").split())
#         return " ".join(s.split()[:cap])
#
#     if json_mode:
#         global_feedback = gf or ""
#         # carry over JSON-provided per-class guidance
#         for c in weak_classes:
#             v = ci.get(c, "") if isinstance(ci, dict) else ""
#             if v:
#                 cap = 20 if c=="unc" else 15
#                 v = _trim_words(v, cap)
#                 if persistence.get(c,0) > 0 and not v.lower().startswith(("mandate:", "stronger:")):
#                     v = "STRONGER: " + v
#                 class_instructions[c] = v
#     else:
#         # Plain-text mode: split into up to 3 non-empty lines
#         lines = [ln.strip() for ln in (raw or "").strip().splitlines() if ln.strip()]
#         while len(lines) < 3:
#             lines.append("")
#         global_feedback = lines[0]
#         # Heuristic mapping of lines 2-3 to weak classes
#         if len(weak_classes) == 1:
#             tip = (lines[1] + (" " + lines[2] if lines[2] else "")).strip()
#             class_instructions[weak_classes[0]] = _trim_words(tip, 20 if weak_classes[0]=="unc" else 15)
#         elif len(weak_classes) == 2:
#             class_instructions[weak_classes[0]] = _trim_words(lines[1], 20 if weak_classes[0]=="unc" else 15)
#             class_instructions[weak_classes[1]] = _trim_words(lines[2], 20 if weak_classes[1]=="unc" else 15)
#         else:  # 3 classes
#             # Try to split line2 into up to 3 parts by ';' for per-class hints
#             parts = [p.strip() for p in re.split(r";|/|,", lines[1]) if p.strip()]
#             for i, c in enumerate(weak_classes):
#                 hint = parts[i] if i < len(parts) else (lines[2] if i==1 else "")
#                 if hint:
#                     class_instructions[c] = _trim_words(hint, 20 if c=="unc" else 15)
#
#     print(f"[critic] class_instructions: {class_instructions}", flush=True)
#     if global_feedback:
#         print(f"[critic] global_feedback: {global_feedback}", flush=True)
#
#     # ---- ACTOR ----
#     # Upgrade STRONGER → MANDATE for the actor
#     ci_for_actor = {}
#     for k,v in class_instructions.items():
#         vv = v
#         if v.lower().startswith("stronger:"):
#             vv = "MANDATE:" + v[len("STRONGER:"):]
#         ci_for_actor[k] = vv
#
#     actor_user = (
#         "SYSTEM PROMPT:\n---\n"+cur_prompt+
#         "\n---\nGLOBAL_FEEDBACK:\n"+global_feedback+
#         "\n---\nCLASS_INSTRUCTIONS:\n"+json.dumps(ci_for_actor)+
#         ("\nREGRESSION_NOTE:\n"+regression_note if regression_note else "")
#     )
#     cand = gen(client, model, ACTOR_SYS, actor_user, temp=0.2, max_tokens=240)
#
#     # Sanity: keep prompt if too long or looks meta
#     if len(cand.split())>120 or "->" in cand or "::" in cand:
#         cand = cur_prompt
#     return cand, class_instructions
#
# # -------- scoring / selection --------
# def score_split(rows, sys_prompt, suffix, client, model, emb, judge,
#                 temp=0.0, top_p=1.0, gold_mode="dd", no_sim=False,
#                 judge_with_context=False, pooled_probs=True):
#     gold_texts=[]; preds=[]; users=[]; sims=[]; s3=[]; norole=[]; tox=[]; all_toks=[]
#     total = len(rows)
#     for i, r in enumerate(rows, 1):
#         up  = build_user_prompt(r.get("context",""), r.get("user",""), suffix)
#         out = gen(client, model, sys_prompt, up, temp=temp, top_p=top_p, max_tokens=128)
#         out = trim_to_n_sentences(out, 3)  # enforce ≤3
#         gold=(r.get("gold_reply") or "")
#         gold_texts.append(gold); preds.append(out); users.append(r.get("user",""))
#
#         sims.append(0.0 if (no_sim or emb is None) else cos_sim(emb, out[:240], gold[:240]) if gold else 0.0)
#         s3.append(1 if count_sents(out)<=3 else 0)
#         norole.append(1 if no_roleplay(out) else 0)
#         tox.append(tox_flag(out))
#         all_toks += tokenize_simple(out)
#         if i % 10 == 0 or i == total:
#             print(f"[progress] scored {i}/{total}", file=sys.stderr, flush=True)
#
#     if gold_mode=="dd":
#         gold_dd5 = [MAP5_DD.get((r.get("gold_emotion","") or "").lower(),"neu") for r in rows]
#         pred5 = judge.pooled5_argmax(users, preds, with_context=judge_with_context) if pooled_probs else [to5(x) for x in judge.labels(preds)]
#         emo = _macro_f1_with_acc(gold_dd5, pred5)
#     else:
#         if pooled_probs:
#             gold5 = judge.pooled5_argmax(users, gold_texts, with_context=judge_with_context)
#             pred5 = judge.pooled5_argmax(users, preds,       with_context=judge_with_context)
#         else:
#             gold5 = [to5(x) for x in judge.labels(gold_texts)]
#             pred5 = [to5(x) for x in judge.labels(preds)]
#         emo = _macro_f1_with_acc(gold5, pred5)
#
#     distinct1 = distinct_n(all_toks,1); distinct2 = distinct_n(all_toks,2)
#     return {
#         "emo": emo,
#         "sim_mean": float(np.mean(sims) if sims else 0.0),
#         "sentences_leq3": float(np.mean(s3)),
#         "no_roleplay": float(np.mean(norole)),
#         "tox_rate": float(np.mean(tox)),
#         "distinct1": distinct1, "distinct2": distinct2
#     }
#
# def average_runs(parts:List[Dict])->Dict:
#     keys=["sim_mean","sentences_leq3","no_roleplay","tox_rate","distinct1","distinct2"]
#     avg = {k: float(np.mean([p[k] for p in parts])) for k in keys}
#     avg["emo"]={
#         "macro_f1": float(np.mean([p["emo"]["macro_f1"] for p in parts])),
#         "acc": float(np.mean([p["emo"].get("acc",0.0) for p in parts])),
#         "per_class": {c: float(np.mean([p["emo"]["per_class"].get(c,0.0) for p in parts])) for c in FIVE}
#     }
#     return avg
#
# def decide(A, B, eps_sim=0.02, min_sents=0.98, relative_gates=False):
#     reasons=[]
#     if relative_gates:
#         if B["tox_rate"] > A["tox_rate"]: reasons.append("tox worse than A")
#         if B["no_roleplay"] < A["no_roleplay"]: reasons.append("roleplay worse than A")
#         if B["sentences_leq3"] < A["sentences_leq3"]: reasons.append("len worse than A")
#     else:
#         if B["tox_rate"] != 0.0: reasons.append("tox>0")
#         if B["no_roleplay"] < 0.999: reasons.append("roleplay<0.999")
#         if B["sentences_leq3"] < min_sents: reasons.append(f"len<{min_sents}")
#
#     dF1 = B["emo"]["macro_f1"] - A["emo"]["macro_f1"]
#     dSim = B["sim_mean"] - A["sim_mean"]
#
#     if reasons:
#         return {"delta_macroF1": dF1, "delta_sim": dSim, "choose": f"A (gate fail: {','.join(reasons)})"}
#     if dF1 > 0 and dSim >= -eps_sim:
#         verdict = "B"
#     elif dF1 < 0:
#         verdict = "A"
#     else:
#         verdict = "B (tie by sim)" if dSim > 0 else "A"
#     return {"delta_macroF1": dF1, "delta_sim": dSim, "choose": verdict}
#
# # -------------------------
# # main
# # -------------------------
# def main():
#     ap=argparse.ArgumentParser()
#     ap.add_argument("--dev", required=True)
#     ap.add_argument("--val", required=True)
#     ap.add_argument("--test", required=True)
#     ap.add_argument("--preprompt", required=True)
#     ap.add_argument("--suffix", required=True)
#     ap.add_argument("--model", default="gpt-4o-mini")
#     ap.add_argument("--iterations", type=int, default=6)
#     ap.add_argument("--val_patience", type=int, default=3)
#     ap.add_argument("--multi_seed", action="store_true")
#     ap.add_argument("--gold", choices=["dd","judge"], default="dd")
#     ap.add_argument("--examples_n", type=int, default=40, help="DEV examples per iter for critic")
#     ap.add_argument("--seed", type=int, default=42)
#     ap.add_argument("--no_sim", action="store_true", help="Skip SBERT similarity to save time")
#
#     # Evaluator toggles
#     ap.add_argument("--judge_context", dest="judge_with_context", action="store_true",
#                     help="Judge with user+assistant context (default: reply-only)")
#     ap.set_defaults(judge_with_context=False)
#     ap.add_argument("--no_pooled_probs", dest="pooled_probs", action="store_false",
#                     help="Use legacy argmax->map (default: pooled 27->5)")
#     ap.set_defaults(pooled_probs=True)
#
#     ap.add_argument("--judge_model", default="joeddav/distilbert-base-uncased-go-emotions-student",
#                     help="HuggingFace model id for GoEmotions judge (e.g., SamLowe/roberta-base-go_emotions)")
#
#     ap.add_argument("--out", default="results_short_v7.json")
#     args=ap.parse_args()
#
#     set_seeds(args.seed)
#     assert OpenAI is not None, "pip install openai"
#     print("[init] creating OpenAI client…", flush=True)
#     client=OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
#
#     judge = EmotionJudge(args.judge_model)
#     emb   = None if args.no_sim else mk_emb()
#
#     dev  = read_csv(args.dev)
#     val  = read_csv(args.val)
#     test = read_csv(args.test)
#     pre  = read_text(args.preprompt)
#     suf  = read_text(args.suffix)
#
#     print(f"[init] DEV={len(dev)} VAL={len(val)} TEST={len(test)}", flush=True)
#     print(f"[init] model={args.model} iterations={args.iterations} val_patience={args.val_patience} gold={args.gold} "
#           f"no_sim={args.no_sim} judge_with_context={args.judge_with_context} pooled_probs={args.pooled_probs} "
#           f"judge_model={args.judge_model}", flush=True)
#     assert MAP5_DD["surprise"]=="unc", "[assert] surprise mapping must be 'unc'"
#
#     history=[]; best_val=-1.0; best_prompt=pre; no_gain=0; cur=pre
#     last_val=None; last_feedback=None; last_prompt=None
#     regression_note_for_next_iter=""
#     persistence = {c: 0 for c in FIVE}
#
#     # ---- Actor→Critic loop ----
#     for it in range(1, args.iterations+1):
#         print(f"[iter] {it}/{args.iterations} — building balanced DEV examples…", flush=True)
#         examples = stratified_examples(dev, args.examples_n, cur, suf, client, args.model)
#
#         # weak classes from a VAL sample (reply-only)
#         summ = val_error_summary(val, client, args.model, cur, suf, judge,
#                                  sample_size=40, with_context=args.judge_with_context, pooled_probs=args.pooled_probs)
#         weak = summ["worst_three"]  # THREE
#         per_class_f1 = summ["per_class_f1"]
#
#         # persistence update
#         for c in FIVE:
#             persistence[c] = (persistence[c] + 1) if c in weak else 0
#
#         cand, class_instr = actor_critic_once(
#             client, args.model, examples, cur, weak, persistence, per_class_f1, regression_note_for_next_iter
#         )
#         print(f"[iter] {it} — weak classes: {weak} | F1: {per_class_f1} | persistence: {persistence}", flush=True)
#
#         # log artifacts per iter
#         with open(f"results_iter_{it}_fb.json","w",encoding="utf-8") as f:
#             json.dump({"iter":it,"class_instructions":class_instr,"weak":weak,"persistence":persistence}, f, indent=2, ensure_ascii=False)
#         with open(f"results_iter_{it}_prompt.txt","w",encoding="utf-8") as f: f.write(cand)
#         with open("results_prompts_log.txt","a",encoding="utf-8") as f:
#             f.write(f"\n\n=== ITER {it} ===\nWEAK_CLASSES: {weak}\nPERSISTENCE: {persistence}\nCLASS_INSTR: {class_instr}\nPROMPT:\n{cand}\n")
#
#         print("[phase] VAL scoring…", flush=True)
#         if args.multi_seed:
#             parts=[]
#             for temp,top_p in [(0.2,0.9),(0.2,0.95),(0.15,0.9)]:
#                 parts.append(score_split(val, cand, suf, client, args.model, emb, judge,
#                                          temp=temp, top_p=top_p, gold_mode=args.gold, no_sim=args.no_sim,
#                                          judge_with_context=args.judge_with_context, pooled_probs=args.pooled_probs))
#             s = average_runs(parts)
#             val_f1 = s["emo"]["macro_f1"]
#         else:
#             s = score_split(val, cand, suf, client, args.model, emb, judge,
#                             temp=0.0, gold_mode=args.gold, no_sim=args.no_sim,
#                             judge_with_context=args.judge_with_context, pooled_probs=args.pooled_probs)
#             val_f1 = s["emo"]["macro_f1"]
#
#         print(f"[val] iter {it} macro-F1={val_f1:.4f} (best so far: {best_val:.4f})", flush=True)
#
#         history.append({"iter":it,"val_macroF1":val_f1,"class_instructions":class_instr,"prompt":cand})
#         if val_f1 > best_val + 1e-9:
#             best_val=val_f1; best_prompt=cand; no_gain=0
#             print(f"[val] new best prompt selected (macro-F1 ↑)", flush=True)
#         else:
#             no_gain += 1
#             print(f"[val] no improvement (patience {no_gain}/{args.val_patience})", flush=True)
#         cur=cand
#         if no_gain>=args.val_patience:
#             print("[early-stop] patience reached — stopping actor loop.", flush=True)
#             break
#
#         # regression note for next iter (only when we regressed wrt last iter)
#         if last_val is not None and val_f1 < last_val - 1e-9:
#             prev_prompt_snip = " ".join((last_prompt or pre).split()[:12])
#             prev_fb_line = ", ".join(f"{k}:{v}" for k,v in (last_feedback or {}).items())
#             regression_note_for_next_iter = (
#                 "Previous iteration scored worse. Prior prompt head: '"+prev_prompt_snip+"'. Prior feedback: "+prev_fb_line
#             )
#         else:
#             regression_note_for_next_iter = ""
#         last_val = val_f1; last_feedback = class_instr; last_prompt = cand
#
#     # Save best prompt snapshot for audit
#     with open("results_best_prompt.txt","w",encoding="utf-8") as f:
#         f.write(best_prompt)
#     print(f"[audit] Saved best prompt to results_best_prompt.txt (words={len(best_prompt.split())}, chars={len(best_prompt)})", flush=True)
#
#     # ---- Upper bounds (judge vs DD on GOLD) ----
#     val_users  = [r.get("user","") for r in val]
#     val_gold5  = [MAP5_DD.get((r.get("gold_emotion","") or "").lower(),"neu") for r in val]
#     val_goldtx = [r.get("gold_reply","") for r in val]
#     val_ub_pred5 = judge.pooled5_argmax(val_users, val_goldtx, with_context=args.judge_with_context)
#     ub_val = _macro_f1_with_acc(val_gold5, val_ub_pred5)
#     print(f"[upper-bound] VAL judge vs DD — acc={ub_val['acc']:.3f} macroF1={ub_val['macro_f1']:.3f}", flush=True)
#
#     test_users  = [r.get("user","") for r in test]
#     test_gold5  = [MAP5_DD.get((r.get("gold_emotion","") or "").lower(),"neu") for r in test]
#     test_goldtx = [r.get("gold_reply","") for r in test]
#     test_ub_pred5 = judge.pooled5_argmax(test_users, test_goldtx, with_context=args.judge_with_context)
#     ub_test = _macro_f1_with_acc(test_gold5, test_ub_pred5)
#     print(f"[upper-bound] TEST judge vs DD — acc={ub_test['acc']:.3f} macroF1={ub_test['macro_f1']:.3f}", flush=True)
#
#     # ---- TEST A/B ----
#     print("[phase] TEST A/B scoring…", flush=True)
#     def maybe_avg(split, prompt, use_multiseed):
#         if use_multiseed:
#             parts=[]
#             for temp,top_p in [(0.2,0.9),(0.2,0.95),(0.15,0.9)]:
#                 parts.append(score_split(split, prompt, suf, client, args.model, emb, judge,
#                                          temp=temp, top_p=top_p, gold_mode=args.gold, no_sim=args.no_sim,
#                                          judge_with_context=args.judge_with_context, pooled_probs=args.pooled_probs))
#             return average_runs(parts)
#         return score_split(split, prompt, suf, client, args.model, emb, judge,
#                            temp=0.0, gold_mode=args.gold, no_sim=args.no_sim,
#                            judge_with_context=args.judge_with_context, pooled_probs=args.pooled_probs)
#
#     # Single-seed by default (no duplicates)
#     A = maybe_avg(test, pre, use_multiseed=args.multi_seed)
#     B = maybe_avg(test, best_prompt, use_multiseed=args.multi_seed)
#     verdict = decide(A, B, eps_sim=0.02, min_sents=0.98, relative_gates=False)
#
#     from collections import Counter
#     gold_judge5 = judge.pooled5_argmax(test_users, test_goldtx, with_context=args.judge_with_context)
#     dist = Counter(gold_judge5)
#     print(f"[audit] Judge 5-way distribution on TEST gold replies: {dict(dist)}", flush=True)
#
#     out = {
#         "prompt_a": pre,
#         "prompt_b": best_prompt,
#         "val_best_macroF1": best_val,
#         "test_A": A, "test_B": B,
#         "decision": verdict,
#         "history": history,
#         "upper_bounds": {"VAL": ub_val, "TEST": ub_test},
#         "judge_gold_distribution_on_TEST": dict(dist),
#         "args": vars(args)
#     }
#     with open(args.out,"w",encoding="utf-8") as f: json.dump(out,f,indent=2,ensure_ascii=False)
#
#     def perclass_str(emo):
#         pc = emo.get("per_class",{})
#         return {k: round(pc.get(k,0.0),4) for k in FIVE}
#
#     print("\n=== SUMMARY ===", flush=True)
#     print(json.dumps({
#         "VAL_best_macroF1": best_val,
#         "VAL_upper_bound_acc": ub_val["acc"], "VAL_upper_bound_macroF1": ub_val["macro_f1"],
#         "TEST_macroF1_A": A["emo"]["macro_f1"],
#         "TEST_macroF1_B": B["emo"]["macro_f1"],
#         "TEST_acc_A": A["emo"].get("acc"),
#         "TEST_acc_B": B["emo"].get("acc"),
#         "TEST_per_class_A": perclass_str(A["emo"]),
#         "TEST_per_class_B": perclass_str(B["emo"]),
#         "TEST_sim_A": A["sim_mean"], "TEST_sim_B": B["sim_mean"],
#         "TEST_dist1_A": A["distinct1"], "TEST_dist1_B": B["distinct1"],
#         "TEST_sents<=3_A": A["sentences_leq3"], "TEST_sents<=3_B": B["sentences_leq3"],
#         "TEST_no_roleplay_A": A["no_roleplay"], "TEST_no_roleplay_B": B["no_roleplay"],
#         "TEST_tox_A": A["tox_rate"], "TEST_tox_B": B["tox_rate"],
#         "TEST_upper_bound_acc": ub_test["acc"], "TEST_upper_bound_macroF1": ub_test["macro_f1"],
#         "DECISION": verdict["choose"]
#     }, indent=2), flush=True)
#     print(f"[done] results saved to {args.out}", flush=True)
#
# if __name__=="__main__":
#     main()

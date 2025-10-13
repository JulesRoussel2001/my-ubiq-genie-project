#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stratified splitter for DailyDialog pairs.

Key points:
- 5-way mapping (pos/sad/ang/unc/neu) with **surprise -> unc** (fixed).
- Stratify by DD labels (default) or by GoEmotions judge (--by_judge).
- Mapping tweak for judge buckets: curiosity/realization → NEU (not UNC).
- Optional dialog-act sub-bucketing (--by_act).
- When --by_judge, also print judge-class counts to avoid confusion.
- Writes *_dev.csv, *_val.csv, *_test.csv and prints class distributions.

Examples
--------
DD labels:
  python3 scripts/stratified_splitter.py \
    --in_csv data/dd_pairs.csv --out_prefix data/dd \
    --dev_per_class 40 --val_per_class 80 --test_per_class 160

Judge labels:
  python3 scripts/stratified_splitter.py \
    --in_csv data/dd_pairs.csv --out_prefix data/dd \
    --dev_per_class 40 --val_per_class 80 --test_per_class 160 --by_judge
"""
import csv, argparse, os, random
from collections import defaultdict, Counter

# --- 5-way mapping from DailyDialog gold_emotion (FIXED: surprise -> unc) ---
MAP5_DD = {
  "happiness":"pos",
  "sadness":"sad",
  "anger":"ang",
  "disgust":"ang",
  "fear":"unc",
  "surprise":"unc",     # <<< FIXED (used to be "neu")
  "no_emotion":"neu",
  "":"neu"
}
FIVE = ["pos","sad","ang","unc","neu"]

# Optional: GoEmotions -> 5-way (for --by_judge)
GO_POS = {"admiration","amusement","approval","caring","excitement","gratitude","joy","love","optimism","pride","relief","desire"}
GO_SAD = {"sadness","disappointment","grief","remorse","embarrassment"}
GO_ANG = {"anger","annoyance","disgust","disapproval"}
GO_UNC = {"fear","nervousness","confusion","surprise"}  # curiosity/realization moved out
GO_NEU = {"neutral","curiosity","realization","realisation"}  # UK spelling too

def go_to5(lbl:str):
    l = (lbl or "neutral").lower()
    if l in GO_POS: return "pos"
    if l in GO_SAD: return "sad"
    if l in GO_ANG: return "ang"
    if l in GO_UNC: return "unc"
    if l in GO_NEU: return "neu"
    return "neu"

def _read(path):
    with open(path,encoding="utf-8") as f:
        return list(csv.DictReader(f))

def _write(path,rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    flds=["context","user","gold_reply","gold_emotion","dialog_act"]
    with open(path,"w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=flds); w.writeheader(); w.writerows(rows)

def _emo5_dd(r):
    return MAP5_DD.get((r.get("gold_emotion","") or "").lower(),"neu")

def _act(r):
    return (r.get("dialog_act","") or "none").lower()

def stratify_by_dd(rows, by_act=False):
    buckets=defaultdict(list)
    for r in rows:
        cls = _emo5_dd(r)
        key = f"{cls}|{_act(r)}" if by_act else cls
        buckets[key].append(r)
    return buckets

def stratify_by_judge(rows, judge_model="joeddav/distilbert-base-uncased-go-emotions-student", by_act=False):
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        EMO_LABELS = ['admiration','amusement','anger','annoyance','approval','caring','confusion','curiosity',
                      'desire','disappointment','disapproval','disgust','embarrassment','excitement','fear',
                      'gratitude','grief','joy','love','nervousness','optimism','pride','realization','relief',
                      'remorse','sadness','surprise','neutral']
        tok = AutoTokenizer.from_pretrained(judge_model)
        mdl = AutoModelForSequenceClassification.from_pretrained(judge_model)
        mdl.eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        mdl.to(device)

        def judge(texts):
            with torch.no_grad():
                b = tok(texts, return_tensors="pt", truncation=True, max_length=512, padding=True).to(device)
                idx = mdl(**b).logits.softmax(dim=-1).argmax(dim=-1).cpu().numpy().tolist()
                return [EMO_LABELS[i].lower() for i in idx]

        golds = [r.get("gold_reply","") for r in rows]
        labels = judge(golds)
        buckets=defaultdict(list)
        for r,lbl in zip(rows, labels):
            cls5 = go_to5(lbl)
            key = f"{cls5}|{_act(r)}" if by_act else cls5
            buckets[key].append(r)
        print("CLASS (by_judge):", dict(Counter(go_to5(l) for l in labels)))
        return buckets
    except Exception as e:
        print(f"[WARN] judge-based stratification failed ({e}); falling back to DD labels.")
        return stratify_by_dd(rows, by_act=by_act)

def rr_take_per_class(buckets, per_class, seed=42):
    random.seed(seed)
    # regroup by class
    by_cls = {c: defaultdict(list) for c in FIVE}
    for key, rows in buckets.items():
        parts = key.split("|")
        cls = parts[0]
        sub = parts[1] if len(parts)>1 else "_all"
        for r in rows:
            by_cls[cls][sub].append(r)
    # shuffle each sub-bucket
    for c in FIVE:
        for sk in by_cls[c]:
            random.shuffle(by_cls[c][sk])

    picked = []
    for c in FIVE:
        need = per_class
        subs = list(by_cls[c].keys())
        i = 0
        while need > 0 and any(by_cls[c][sk] for sk in subs):
            sk = subs[i % len(subs)]
            if by_cls[c][sk]:
                picked.append(by_cls[c][sk].pop())
                need -= 1
            i += 1
    # remove picked from original buckets
    picked_ids = set(id(r) for r in picked)
    for key in list(buckets.keys()):
        buckets[key] = [r for r in buckets[key] if id(r) not in picked_ids]
        if not buckets[key]:
            buckets.pop(key, None)
    return picked

def dist_by_dd(rows):
    return dict(Counter(MAP5_DD.get((r.get("gold_emotion","") or "").lower(),"neu") for r in rows))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True)
    ap.add_argument("--out_prefix", default="data/dd")
    ap.add_argument("--dev_per_class",  type=int, default=20)
    ap.add_argument("--val_per_class",  type=int, default=40)
    ap.add_argument("--test_per_class", type=int, default=80)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--by_act", action="store_true")
    ap.add_argument("--by_judge", action="store_true", help="use GoEmotions judge for 5-way buckets")
    ap.add_argument("--judge_model", default="joeddav/distilbert-base-uncased-go-emotions-student",
                    help="HuggingFace model id for GoEmotions judge")
    args=ap.parse_args()

    random.seed(args.seed)
    rows = _read(args.in_csv)

    # build buckets
    if args.by_judge:
        buckets = stratify_by_judge(rows, judge_model=args.judge_model, by_act=args.by_act)
    else:
        buckets = stratify_by_dd(rows, by_act=args.by_act)

    # DEV / VAL / TEST
    dev  = rr_take_per_class(buckets, args.dev_per_class,  seed=args.seed)
    val  = rr_take_per_class(buckets, args.val_per_class,  seed=args.seed+1)
    test = rr_take_per_class(buckets, args.test_per_class, seed=args.seed+2)

    # write
    _write(f"{args.out_prefix}_dev.csv",  dev)
    _write(f"{args.out_prefix}_val.csv",  val)
    _write(f"{args.out_prefix}_test.csv", test)

    print("DEV  :", dist_by_dd(dev))
    print("VAL  :", dist_by_dd(val))
    print("TEST :", dist_by_dd(test))
    assert MAP5_DD["surprise"]=="unc", "[assert] surprise mapping must be 'unc'"

if __name__=="__main__":
    main()

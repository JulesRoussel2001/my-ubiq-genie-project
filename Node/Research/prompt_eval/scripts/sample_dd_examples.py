import csv, argparse, random, textwrap
from collections import defaultdict

def load_rows(p):
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))

def show(rows, per_class):
    want = {"anger","sadness","fear","surprise"}  # raw DD labels
    buckets = defaultdict(list)
    for r in rows:
        emo = (r.get("gold_emotion","") or "").lower()
        if emo in want:
            buckets[emo].append(r)
    for emo in ["anger","sadness","fear","surprise"]:
        pool = buckets.get(emo, [])
        print(f"\n=== {emo.upper()} (showing up to {per_class}, pool={len(pool)}) ===")
        for r in random.sample(pool, min(per_class, len(pool))):
            user = (r.get("user","") or "").strip()
            gold = (r.get("gold_reply","") or "").strip()
            print("- USER:", textwrap.shorten(user, width=140, placeholder="…"))
            print("  REPLY:", textwrap.shorten(gold, width=140, placeholder="…"))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--per_class", type=int, default=5)
    args = ap.parse_args()
    rows = load_rows(args.csv)
    show(rows, args.per_class)

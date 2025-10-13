#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert DailyDialog raw files to a pair-wise CSV with columns:
  context (empty by default), user, gold_reply, gold_emotion, dialog_act

Inputs (from DailyDialog):
- dialogues_text.txt
- dialogues_act.txt
- dialogues_emotion.txt
"""
import csv, argparse, os

EMO = {0:"no_emotion",1:"anger",2:"disgust",3:"fear",4:"happiness",5:"sadness",6:"surprise"}
ACT = {0:"none",1:"inform",2:"question",3:"directive",4:"commissive"}

def _lines(p): return [ln.rstrip("\n") for ln in open(p,encoding="utf-8")]
def _utts(line): return [p.strip() for p in line.split("__eou__") if p.strip()]
def _ints(line): return [int(x) for x in line.split()] if line.strip() else []

def convert(text_path, act_path, emo_path, out_csv):
    texts, acts, emos = _lines(text_path), _lines(act_path), _lines(emo_path)
    assert len(texts)==len(acts)==len(emos), "Mismatched file lengths."
    rows=[]
    for t,a,e in zip(texts, acts, emos):
        utts=_utts(t); aseq=_ints(a); eseq=_ints(e)
        if len(aseq)<len(utts): aseq += [0]*(len(utts)-len(aseq))
        if len(eseq)<len(utts): eseq += [0]*(len(utts)-len(eseq))
        aseq,eseq = aseq[:len(utts)], eseq[:len(utts)]
        for i in range(len(utts)-1):
            rows.append({
                "context": "",
                "user": utts[i],
                "gold_reply": utts[i+1],
                "gold_emotion": EMO.get(eseq[i+1],"no_emotion"),
                "dialog_act": ACT.get(aseq[i],"none")
            })
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv,"w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=["context","user","gold_reply","gold_emotion","dialog_act"])
        w.writeheader(); w.writerows(rows)
    print(f"Wrote {len(rows)} rows -> {out_csv}")

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--text", required=True)
    ap.add_argument("--act", required=True)
    ap.add_argument("--emotion", required=True)
    ap.add_argument("--out", default="data/dd_pairs.csv")
    args=ap.parse_args()
    convert(args.text,args.act,args.emotion,args.out)

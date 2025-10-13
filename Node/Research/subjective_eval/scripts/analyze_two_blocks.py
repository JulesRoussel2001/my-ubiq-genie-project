#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_two_blocks.py  —  FINAL (dynamic M-levels + tone slices)

Usage:
  python analyze_two_blocks.py <animation_csv> <final_csv> [items_animation.csv] [items_final.csv] [out.json]

Inputs (CSV columns must match):
  participant_id, feature_evaluated, condition, item_id, rating_1to5
- Animation block: condition in {M0, M1, M2, M3, M4, ...}
- Final A/B block: condition in {BASELINE, FINAL}
- Item banks: must include columns item_id, reverse (0/1), subscale

What it does
- Reverse-scores via items tables (reverse=1 => score = 6 - rating)
- Animation: condition means + bootstrap 95% CI, and ALL stepwise paired contrasts
  across whatever M-levels exist (M0->M1, M1->M2, ..., M(k-1)->Mk), plus extras M1->M3 and M1->M4 if present
- Final A/B: overall paired difference (FINAL - BASELINE) + tone-focused slices:
  * explicit tone:   subscales = {voice_naturalness, voice_fit}
  * broad tone:      subscales = explicit + {congruence, emotional_fit, emotional_involvement, coherence}

Output
- Prints JSON to console
- Writes JSON to the given out.json path (default: outputs/two_blocks_results.json) and ensures the directory exists
"""
import os, sys, json
import numpy as np
import pandas as pd

# -------------------- helpers --------------------
def bootstrap_ci(values, B=5000, alpha=0.05, seed=7):
    v = np.array(values, dtype=float)
    if len(v) == 0:
        return (float('nan'), float('nan'), float('nan'))
    rng = np.random.default_rng(seed)
    boots = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(B)]
    lo = float(np.percentile(boots, 100*alpha/2))
    hi = float(np.percentile(boots, 100*(1 - alpha/2)))
    return (float(v.mean()), lo, hi)

def sign_test_paired(diffs):
    # Two-sided exact sign test on paired differences
    s = [1 if d>0 else (-1 if d<0 else 0) for d in diffs]
    k = sum(1 for x in s if x>0)
    n = sum(1 for x in s if x!=0)
    if n == 0:
        return 1.0
    from math import comb
    tail = sum(comb(n, i) for i in range(max(k, n-k), n+1)) / (2**n)
    return float(min(1.0, 2*tail))

def cohen_dz(diffs):
    d = np.array(diffs, dtype=float)
    if len(d) < 2:
        return float('nan')
    sd = d.std(ddof=1)
    return float('nan') if sd == 0 else float(d.mean() / sd)

def paired_perm_p(diffs, R=10000, seed=11):
    rng = np.random.default_rng(seed)
    d = np.array(diffs, dtype=float)
    if len(d) == 0:
        return 1.0
    obs = abs(d.mean())
    ge = 0
    for _ in range(R):
        flips = rng.choice([1, -1], size=len(d))
        if abs((d*flips).mean()) >= obs:
            ge += 1
    return float((ge + 1) / (R + 1))

def apply_reverse(df, items_df):
    # reverse=1 => score = 6 - rating (Likert 1..5)
    rev_map = dict(zip(items_df["item_id"], items_df["reverse"]))
    df = df.copy()
    df["rating"] = pd.to_numeric(df["rating_1to5"], errors="coerce")
    df = df.dropna(subset=["rating"])
    df["score"] = df.apply(
        lambda r: (6 - r["rating"]) if int(rev_map.get(r["item_id"], 0)) == 1 else r["rating"],
        axis=1
    )
    return df

def sort_levels(levels):
    # Sort like M0, M1, M2, ... and put any non-M* labels at the end
    def keyfunc(s):
        if isinstance(s, str) and s.startswith("M"):
            num = s[1:]
            if num.isdigit():
                return (0, int(num))
        return (1, s)
    return sorted(levels, key=keyfunc)

def paired_summary_from_df(df_subset):
    """Compute paired FINAL-BASELINE stats from a dataframe with columns:
       participant_id, condition, score
    """
    means = df_subset.groupby(["participant_id","condition"], as_index=False)["score"].mean()
    diffs = []
    for pid, gp in means.groupby("participant_id"):
        row = gp.set_index("condition")["score"]
        if {"BASELINE","FINAL"}.issubset(row.index):
            diffs.append(float(row["FINAL"] - row["BASELINE"]))
    mu, lo, hi = bootstrap_ci(diffs)
    return {
        "n_pairs": len(diffs),
        "mean_diff": mu,
        "ci95": [lo, hi],
        "p_sign": sign_test_paired(diffs),
        "cohen_dz": cohen_dz(diffs)
    }

def short_comment(stats, label):
    mu = stats["mean_diff"]; lo, hi = stats["ci95"]
    p = stats["p_sign"]
    sig = "significant" if (p < 0.05 and lo > 0) else "not significant"
    return f"{label}: Δ={mu:.2f} [{lo:.2f}, {hi:.2f}], sign test p={p:.3g} ⇒ {sig} uplift."

# -------------------- analyses --------------------
def analyze_animation(csv_path, items_path):
    items = pd.read_csv(csv_path if csv_path.endswith('items_animation.csv') else items_path)  # backward safety
    if 'reverse' not in items.columns or 'item_id' not in items.columns:
        items = pd.read_csv(items_path)  # ensure item bank, not ratings

    df = pd.read_csv(csv_path)
    df = apply_reverse(df, items)
    # participant means per condition
    means = df.groupby(["participant_id","condition"], as_index=False)["score"].mean()

    # Condition summaries (means + CI), dynamically across levels
    levels = sort_levels(means["condition"].unique().tolist())
    cond_summary = {}
    for lv in levels:
        g = means.loc[means["condition"]==lv, "score"].tolist()
        m, lo, hi = bootstrap_ci(g)
        cond_summary[lv] = {"mean": m, "ci95":[lo, hi], "n": int(len(g))}

    # Build stepwise contrasts: Mx -> M(x+1)
    pairs = []
    for i in range(len(levels)-1):
        a, b = levels[i], levels[i+1]
        if a.startswith("M") and b.startswith("M"):
            pairs.append((a,b))
    # Optional extras for reporting
    if set(["M1","M3"]).issubset(levels): pairs.append(("M1","M3"))
    if set(["M1","M4"]).issubset(levels): pairs.append(("M1","M4"))

    out = []
    for a,b in pairs:
        diffs = []
        for pid, gp in means.groupby("participant_id"):
            row = gp.set_index("condition")["score"]
            if a in row and b in row:
                diffs.append(float(row[b] - row[a]))
        mu, lo, hi = bootstrap_ci(diffs)
        out.append({
            "contrast": f"{a}->{b}",
            "n_pairs": len(diffs),
            "mean_diff": mu,
            "ci95": [lo, hi],
            "p_perm": paired_perm_p(diffs),
            "p_sign": sign_test_paired(diffs),
            "cohen_dz": cohen_dz(diffs)
        })

    return {"conditions": cond_summary, "paired_contrasts": out}

def analyze_final(csv_path, items_path):
    items = pd.read_csv(items_path)
    df = pd.read_csv(csv_path)
    df = apply_reverse(df, items)
    # merge subscale for slicing
    df = df.merge(items[["item_id","subscale"]], on="item_id", how="left")

    # Overall
    overall_stats = paired_summary_from_df(df[["participant_id","condition","score"]])

    # Tone slices
    tone_explicit = {"voice_naturalness","voice_fit"}
    tone_broad = tone_explicit | {"congruence","emotional_fit","emotional_involvement","coherence"}

    exp_df = df[df["subscale"].isin(tone_explicit)][["participant_id","condition","score"]]
    broad_df = df[df["subscale"].isin(tone_broad)][["participant_id","condition","score"]]

    exp_stats = paired_summary_from_df(exp_df) if len(exp_df) else {"n_pairs":0,"mean_diff":float('nan'),"ci95":[float('nan'),float('nan')],"p_sign":1.0,"cohen_dz":float('nan')}
    brd_stats = paired_summary_from_df(broad_df) if len(broad_df) else {"n_pairs":0,"mean_diff":float('nan'),"ci95":[float('nan'),float('nan')],"p_sign":1.0,"cohen_dz":float('nan')}

    commentary = {
        "explicit": short_comment(exp_stats, "Tone (explicit)"),
        "broad": short_comment(brd_stats, "Tone (broad)")
    }

    return {
        "paired_difference": overall_stats,
        "tone": {
            "explicit": exp_stats,
            "broad": brd_stats,
            "commentary": commentary
        }
    }

# -------------------- main --------------------
def main(anim_csv, final_csv, items_anim_csv, items_final_csv, out_json):
    res = {
        "animation": analyze_animation(anim_csv, items_anim_csv),
        "final_ab": analyze_final(final_csv, items_final_csv)
    }
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(res, f, indent=2)
    print("Wrote", out_json)
    print(json.dumps(res, indent=2))

if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) < 2:
        print("Usage: python analyze_two_blocks.py <anim.csv> <final.csv> [items_anim.csv] [items_final.csv] [out.json]")
        sys.exit(1)

    anim_csv = args[0]
    final_csv = args[1]
    items_anim_csv = args[2] if len(args) > 2 else "items/items_animation.csv"
    items_final_csv = args[3] if len(args) > 3 else "items/items_final.csv"
    out_json = args[4] if len(args) > 4 else "outputs/two_blocks_results.json"

    main(anim_csv, final_csv, items_anim_csv, items_final_csv, out_json)

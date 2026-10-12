#!/usr/bin/env python3
"""
Zero-shot TF-IDF baseline: train TF-IDF + Ridge on BBC Track A news values,
evaluate on the CNN test split with the same groups and metrics as every other
method (eval_results/cnn/cnn_split.json).

This is the like-for-like counterpart to ResonantBERT v3 (also trained on BBC only).
The in-domain TF-IDF baseline (trained on CNN) is kept as a reference.

Writes eval_results/cnn_v3/tfidf_bbc_to_cnn.json and adds a column to
eval_results/cnn_v3/test_scores_cnn_v3.csv if that file exists.
"""

import json, os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from scipy.stats import spearmanr, ttest_rel, wilcoxon

from evaluate_v3_cnn import (NEWS_VALUES, SEED, CNN_CSV, BASELINE_DIR, OUTPUT_DIR,
                             evaluate_scores, bootstrap_ci)

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BBC_CSV  = os.path.join(REPO_DIR, "bbc_articles", "track_a_all.csv")
METHOD   = "TF-IDF + Ridge (BBC→CNN)"
VAL_FRAC = 0.10

def text(h, b): return h + " " + b[:2000]          # same text recipe as the CNN TF-IDF baseline

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---- BBC training data ----
    bbc = pd.read_csv(BBC_CSV, usecols=["id", "title", "content"] + NEWS_VALUES,
                      encoding="utf-8", encoding_errors="replace")
    bbc = bbc.dropna(subset=["title", "content"])
    bbc = bbc[bbc["content"].str.strip().str.len() > 20].reset_index(drop=True)
    bbc["gold"] = bbc[NEWS_VALUES].sum(axis=1).astype(float)
    perm = np.random.RandomState(SEED).permutation(len(bbc))
    n_val = int(round(VAL_FRAC * len(bbc)))
    val, train = bbc.iloc[perm[:n_val]], bbc.iloc[perm[n_val:]]
    print(f"BBC Track A: train={len(train)} val={len(val)}")

    # ---- CNN test (identical split/groups) ----
    with open(os.path.join(BASELINE_DIR, "cnn_split.json"), encoding="utf-8") as f:
        split = json.load(f)
    base = pd.read_csv(os.path.join(BASELINE_DIR, "test_scores_cnn_baselines.csv"), encoding="utf-8")
    assert base["id"].tolist() == split["test_ids"]
    cnn = pd.read_csv(CNN_CSV, encoding="utf-8").set_index("id")
    test_df = base.copy()
    test_df["headline"] = cnn.loc[test_df["id"], "title"].astype(str).values
    test_df["body"]     = cnn.loc[test_df["id"], "content"].astype(str).values
    row = {i: r for r, i in enumerate(test_df["id"])}
    test_groups = [np.array([row[i] for i in g]) for g in split["test_groups"]]

    # ---- fit (same hyper-parameters as the CNN TF-IDF baseline; alpha picked on BBC val) ----
    tfidf = TfidfVectorizer(max_features=20000, stop_words="english", ngram_range=(1, 2),
                            sublinear_tf=True, min_df=2)
    X_train = tfidf.fit_transform([text(h, b) for h, b in zip(train["title"], train["content"])])
    X_val   = tfidf.transform([text(h, b) for h, b in zip(val["title"], val["content"])])
    best_alpha, best_rho = None, -np.inf
    for alpha in [0.3, 1.0, 3.0, 10.0]:
        rho = spearmanr(Ridge(alpha=alpha).fit(X_train, train["gold"]).predict(X_val), val["gold"]).correlation
        print(f"  alpha={alpha:<5} BBC-val Spearman={rho:.4f}")
        if rho > best_rho: best_alpha, best_rho = alpha, rho
    ridge = Ridge(alpha=best_alpha).fit(X_train, train["gold"])
    print(f"  chosen alpha={best_alpha}")

    scores = ridge.predict(tfidf.transform([text(h, b) for h, b in zip(test_df["headline"], test_df["body"])]))
    res = evaluate_scores(scores, test_df, test_groups)

    # ---- compare with the in-domain TF-IDF and, if available, ResonantBERT v3 ----
    ref = {"TF-IDF + Ridge (CNN, in-domain)": test_df["TF-IDF + Ridge"].to_numpy(dtype=float)}
    v3_csv = os.path.join(OUTPUT_DIR, "test_scores_cnn_v3.csv")
    if os.path.exists(v3_csv):
        v3 = pd.read_csv(v3_csv, encoding="utf-8")
        assert v3["id"].tolist() == test_df["id"].tolist()
        col = "BERTv3-step1881.pt [value_sum]"
        if col in v3.columns: ref[f"ResonantBERT v3 step1881 (value_sum)"] = v3[col].to_numpy(dtype=float)
        v3[METHOD] = scores
        v3.to_csv(v3_csv, index=False)

    print("\n" + "=" * 100)
    print(f"{'Method':<42} {'NDCG@5':>7} {'NDCG@10':>8} {'Kendall τ':>10} {'Pairwise':>9} {'Spearman':>9}")
    print("-" * 100)
    comparisons = {}
    for name, s in [(METHOD, scores), *ref.items()]:
        r = res if name == METHOD else evaluate_scores(s, test_df, test_groups)
        print(f"{name:<42} {r['ndcg@5']:>7.4f} {r['ndcg@10']:>8.4f} {r['kendall_tau']:>+10.4f} "
              f"{r['pairwise_acc']:>9.4f} {r['spearman_global']:>+9.4f}")
        if name != METHOD:
            a, b = np.array(r["per_group_ndcg10"]), np.array(res["per_group_ndcg10"])
            _, p_t = ttest_rel(a, b)
            try:    _, p_w = wilcoxon(a, b)
            except ValueError: p_w = float("nan")
            comparisons[name] = {"delta_ndcg10_minus_bbc_tfidf": float((a - b).mean()),
                                 "ci95": bootstrap_ci(a - b), "p_ttest": float(p_t), "p_wilcoxon": float(p_w)}
    print("=" * 100)
    for name, c in comparisons.items():
        print(f"{name} − {METHOD}: Δ NDCG@10 = {c['delta_ndcg10_minus_bbc_tfidf']:+.4f} "
              f"[{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}]  t-test p = {c['p_ttest']:.2e}  Wilcoxon p = {c['p_wilcoxon']:.2e}")
    print("\nSpearman with each CNN news value:", {k: round(v, 3) for k, v in res["spearman_by_news_value"].items()})

    with open(os.path.join(OUTPUT_DIR, "tfidf_bbc_to_cnn.json"), "w", encoding="utf-8") as f:
        json.dump({"method": METHOD, "alpha": best_alpha, "bbc_val_spearman": float(best_rho),
                   "n_train": len(train), "metrics": {k: v for k, v in res.items() if k != "per_group_ndcg10"},
                   "comparisons": comparisons}, f, indent=2)
    print(f"\nSaved: {os.path.join(OUTPUT_DIR, 'tfidf_bbc_to_cnn.json')}")

if __name__ == "__main__":
    main()

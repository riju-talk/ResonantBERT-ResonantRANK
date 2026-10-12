#!/usr/bin/env python3
"""
Aggregate the ablation study: mean ± sd over seeds per variant, next to the
evaluation-only baselines, with paired significance (per-group NDCG@10 averaged
over seeds) of each variant vs. `full` and vs. the best TF-IDF baseline.

Reads  eval_results/ablations/<split>/<variant>_seed*/metrics.json + test_scores.csv
       eval_results/ablations/<split>/baselines/{metrics.json, test_scores.csv}
Writes eval_results/ablations/<split>/{ablation_summary.csv, table_ablation.tex, fig_ablation.png/pdf}
"""

import glob, json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel, wilcoxon

from evaluate_v3_cnn import evaluate_scores, bootstrap_ci
from ablation_study import cnn_test, VARIANTS

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "eval_results", "ablations")
KEYS = ["ndcg@5", "ndcg@10", "kendall_tau", "pairwise_acc", "spearman_global"]
LABEL = {"full": "ResonantBERT v3 (full)", "no_rank": "− ranking losses", "no_interaction": "− factor interaction",
         "no_token": "− token-level heads", "plain_coral": "Plain ModernBERT, 7 CORAL heads",
         "plain_mse": "Plain ModernBERT, 1 regression"}
C_OURS, C_ABL, C_BASE, C_INK, C_INK2, C_GRID, C_SURF = "#2a78d6", "#9ec3ee", "#a3a29c", "#0b0b0b", "#52514e", "#e8e7e3", "#fcfcfb"
stars = lambda p: "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))

def paired(a, b):
    _, p_t = ttest_rel(a, b)
    try:    _, p_w = wilcoxon(a, b)
    except ValueError: p_w = float("nan")
    return {"delta": float((a - b).mean()), "ci": bootstrap_ci(a - b), "p_t": float(p_t), "p_w": float(p_w)}

def report(split):
    d = os.path.join(ROOT, split)
    test, groups = cnn_test(split)
    rows, pg = [], {}

    bl = pd.read_csv(os.path.join(d, "baselines", "test_scores.csv"), encoding="utf-8")
    assert bl["id"].tolist() == test["id"].tolist()
    base_methods = [c for c in bl.columns if c not in set(test.columns) | {"gold", "section"}]
    for m in base_methods:
        r = evaluate_scores(bl[m].to_numpy(dtype=float), test, groups)
        pg[m] = np.array(r["per_group_ndcg10"])
        rows.append({"method": m, "kind": "baseline", "n_seeds": 1, **{k: r[k] for k in KEYS}, **{k + "_sd": 0.0 for k in KEYS}})

    for v in VARIANTS:
        runs = sorted(glob.glob(os.path.join(d, f"{v}_seed*", "metrics.json")))
        if not runs: continue
        per_seed, groups_ndcg = [], []
        for p in runs:
            m = json.load(open(p, encoding="utf-8"))
            head = m["primary_head"]
            sc = pd.read_csv(os.path.join(os.path.dirname(p), "test_scores.csv"), encoding="utf-8")
            r = evaluate_scores(sc[head].to_numpy(dtype=float), test, groups)
            per_seed.append({k: r[k] for k in KEYS}); groups_ndcg.append(r["per_group_ndcg10"])
        pg[LABEL[v]] = np.mean(groups_ndcg, 0)
        dfm = pd.DataFrame(per_seed)
        rows.append({"method": LABEL[v], "kind": "ours" if v == "full" else "ablation", "n_seeds": len(runs),
                     **dfm.mean().to_dict(), **{k + "_sd": float(dfm[k].std(ddof=1)) if len(runs) > 1 else float("nan") for k in KEYS}})

    summary = pd.DataFrame(rows).sort_values("ndcg@10", ascending=False).reset_index(drop=True)
    tfidf = [m for m in base_methods if m.startswith("TF-IDF")]
    best_tfidf = max(tfidf, key=lambda m: summary.set_index("method").loc[m, "ndcg@10"]) if tfidf else None
    full = LABEL["full"] if LABEL["full"] in pg else None
    for i, r in summary.iterrows():
        if full and r["method"] != full:
            s = paired(pg[full], pg[r["method"]])
            summary.loc[i, "delta_full_minus_this"] = s["delta"]; summary.loc[i, "p_vs_full"] = s["p_t"]
        if best_tfidf and r["method"] != best_tfidf and r["kind"] != "baseline":
            s = paired(pg[r["method"]], pg[best_tfidf])
            summary.loc[i, "delta_vs_best_tfidf"] = s["delta"]; summary.loc[i, "p_vs_best_tfidf"] = s["p_t"]
    summary.to_csv(os.path.join(d, "ablation_summary.csv"), index=False)

    pd.set_option("display.width", 220)
    show = summary[["method", "n_seeds"] + [c for k in ["ndcg@10", "kendall_tau", "spearman_global"] for c in (k, k + "_sd")] +
                   [c for c in ["delta_full_minus_this", "p_vs_full", "delta_vs_best_tfidf", "p_vs_best_tfidf"] if c in summary]]
    print(f"\n===== {split} split ({len(test)} articles, {len(groups)} groups; best TF-IDF = {best_tfidf}) =====")
    print(show.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # LaTeX
    fmt = lambda r, k: f"{r[k]:.3f}" + (f" $\\pm$ {r[k + '_sd']:.3f}" if r["n_seeds"] > 1 else "")
    tex = ["\\begin{table}[t]", "\\centering", "\\small", "\\begin{tabular}{lccc}", "\\toprule",
           "\\textbf{Model} & \\textbf{NDCG@10} & \\textbf{Kendall $\\tau$} & \\textbf{Spearman $\\rho$} \\\\", "\\midrule"]
    for _, r in summary.iterrows():
        mark = ""
        if "p_vs_full" in r and pd.notna(r.get("p_vs_full")) and stars(r["p_vs_full"]) != "ns":
            mark = f"$^{{{stars(r['p_vs_full'])}}}$"
        name = r["method"].replace("_", "\\_").replace("&", "\\&").replace("−", "$-$")
        name = f"\\textbf{{{name}}}" if r["kind"] == "ours" else name
        tex.append(f"{name}{mark} & {fmt(r, 'ndcg@10')} & {fmt(r, 'kendall_tau')} & {fmt(r, 'spearman_global')} \\\\")
    tex += ["\\bottomrule", "\\end{tabular}",
            f"\\caption{{Ablations, {split} split: trained on BBC, tested zero-shot on CNN ({len(groups)} groups). "
            "Mean $\\pm$ sd over seeds. Stars: paired t-test vs.\\ full model on seed-averaged per-group NDCG@10.}}",
            f"\\label{{tab:ablation_{split}}}", "\\end{table}"]
    open(os.path.join(d, "table_ablation.tex"), "w", encoding="utf-8").write("\n".join(tex))

    # Figure: Kendall tau and Spearman per method, seed sd as error bars
    s = summary.sort_values("kendall_tau")
    color = lambda k: C_OURS if k == "ours" else (C_ABL if k == "ablation" else C_BASE)
    fig, axes = plt.subplots(1, 2, figsize=(14, 0.45 * len(s) + 1.5), sharey=True)
    for ax, (k, t) in zip(axes, [("kendall_tau", "Kendall τ"), ("spearman_global", "Spearman ρ (global)")]):
        ax.barh(range(len(s)), s[k], xerr=s[k + "_sd"].fillna(0), color=[color(x) for x in s["kind"]],
                edgecolor=C_SURF, linewidth=2, error_kw=dict(ecolor=C_INK2, capsize=3))
        for y, (v, sd) in enumerate(zip(s[k], s[k + "_sd"].fillna(0))):
            ax.text(v + sd + 0.01, y, f"{v:.3f}", va="center", fontsize=8, color=C_INK)
        ax.set_title(t, fontweight="bold"); ax.grid(axis="x", color=C_GRID); ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False); ax.set_xlim(min(0, s[k].min()) - 0.02, s[k].max() * 1.18)
    axes[0].set_yticks(range(len(s)), s["method"])
    fig.suptitle(f"Ablations ({split} split): BBC-trained, CNN zero-shot — mean ± sd over seeds", fontweight="bold")
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(d, f"fig_ablation.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {d}/ablation_summary.csv, table_ablation.tex, fig_ablation.png")

if __name__ == "__main__":
    for split in ["random", "temporal"]:
        if os.path.exists(os.path.join(ROOT, split, "baselines", "test_scores.csv")):
            report(split)

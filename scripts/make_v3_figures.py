#!/usr/bin/env python3
"""
Recreate the cnn_sota comparison figures/tables for ResonantBERT v3.

Inputs (run these first):
  python evaluate_v3_cnn.py      -> eval_results/cnn_v3/test_scores_cnn_v3.csv
  python tfidf_bbc_to_cnn.py     -> adds the "TF-IDF + Ridge (BBC→CNN)" column (optional)

Outputs: eval_results/cnn_v3/figures/fig1..fig8 (png + pdf), table_cnn_v3_main.tex,
results_cnn_v3_figures.json.

Usage:
  python make_v3_figures.py                         # main model = BERTv3-step1881.pt
  python make_v3_figures.py --ckpt BERTv3-step1000.pt
"""

import argparse, json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import spearmanr, ttest_rel, wilcoxon

from evaluate_v3_cnn import (NEWS_VALUES, SEED, BASELINE_DIR, OUTPUT_DIR, CKPT_DIR,
                             evaluate_scores, bootstrap_ci)

FIG_DIR = os.path.join(OUTPUT_DIR, "figures")

# Display names: baselines keep their meaning, training data made explicit
RENAME = {"TF-IDF + Ridge": "TF-IDF + Ridge (CNN-trained)",
          "TF-IDF + Ridge (BBC→CNN)": "TF-IDF + Ridge (BBC-trained)",
          "LLM Zero-Shot (Qwen2.5-1.5B-Instruct)": "LLM Zero-Shot (Qwen2.5-1.5B)"}
OURS     = "ResonantBERT v3"
ABL_CON  = "ResonantBERT v3 (concept head)"
ABL_RANK = "ResonantBERT v3 (+RankingTower)"

# ---------------- style (same as cnn_sota) ----------------
C_OURS, C_ABL, C_BASE = "#2a78d6", "#9ec3ee", "#a3a29c"
C_INK, C_INK2, C_GRID, C_SURF, C_WARM = "#0b0b0b", "#52514e", "#e8e7e3", "#fcfcfb", "#eb6834"
mpl.rcParams.update({
    "figure.facecolor": C_SURF, "axes.facecolor": C_SURF, "savefig.facecolor": C_SURF,
    "axes.edgecolor": C_GRID, "axes.labelcolor": C_INK2, "axes.titlecolor": C_INK,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.labelsize": 10,
    "xtick.color": C_INK2, "ytick.color": C_INK2, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": C_GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "font.family": "sans-serif", "legend.frameon": False, "legend.fontsize": 9,
})
DIVERGING = LinearSegmentedColormap.from_list("div", [C_OURS, "#e8e7e3", C_WARM])
color_of = lambda m: C_OURS if m == OURS else (C_ABL if m.startswith(OURS) else C_BASE)
stars = lambda p: "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))

def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG_DIR, f"{name}.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {name}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="BERTv3-step1881.pt", help="checkpoint used as the main model")
    args = ap.parse_args()
    os.makedirs(FIG_DIR, exist_ok=True)

    # ---------------- data ----------------
    with open(os.path.join(BASELINE_DIR, "cnn_split.json"), encoding="utf-8") as f:
        split = json.load(f)
    df = pd.read_csv(os.path.join(OUTPUT_DIR, "test_scores_cnn_v3.csv"), encoding="utf-8")
    assert df["id"].tolist() == split["test_ids"], "v3 scores and split out of sync"
    TARGET = split["target"]
    # Frozen-encoder probe from ablation_baselines.py (same CNN test split), if it has been run
    frozen = "Frozen ModernBERT + Ridge (BBC-trained)"
    abl = os.path.join(os.path.dirname(OUTPUT_DIR), "ablations", "random", "baselines", "test_scores.csv")
    if os.path.exists(abl) and frozen not in df.columns:
        fz = pd.read_csv(abl, encoding="utf-8", usecols=["id", frozen]).set_index("id")
        df[frozen] = fz.loc[df["id"], frozen].to_numpy()
    row = {i: r for r, i in enumerate(df["id"])}
    test_groups = [np.array([row[i] for i in g]) for g in split["test_groups"]]

    base_cols = [c for c in ["Random", "Rule-Based", "Sentence-BERT", "LLM Zero-Shot (Qwen2.5-1.5B-Instruct)",
                             "Frozen ModernBERT + Ridge (BBC-trained)", "TF-IDF + Ridge (BBC→CNN)",
                             "TF-IDF + Ridge"] if c in df.columns]
    ck = args.ckpt
    cols = {RENAME.get(c, c): c for c in base_cols}
    cols[OURS] = f"{ck} [value_sum]"
    cols[ABL_CON] = f"{ck} [concept]"
    # A ranker evaluated on this checkpoint (or on its identical 'final' copy for step1881)
    rank_col = next((c for c in df.columns if "[ranker:" in c and
                     (c.startswith(ck) or (ck == "BERTv3-step1881.pt" and c.startswith("BERTv3-final-step1881")))), None)
    if rank_col: cols[ABL_RANK] = rank_col
    for m, c in cols.items():
        assert c in df.columns, f"missing column {c!r}; re-run evaluate_v3_cnn.py"
    METHODS = list(cols)
    BASELINES = [m for m in METHODS if not m.startswith(OURS)]
    print(f"Main model: {ck} [value_sum] | methods: {METHODS}")

    results = {m: evaluate_scores(df[c].to_numpy(dtype=float), df, test_groups) for m, c in cols.items()}
    pg = {m: np.array(results[m]["per_group_ndcg10"]) for m in METHODS}

    sig = {}
    for m in METHODS:
        if m == OURS: continue
        a, b = pg[OURS], pg[m]
        diff = a - b
        _, p_t = ttest_rel(a, b)
        try:    _, p_w = wilcoxon(a, b)
        except ValueError: p_w = float("nan")
        sig[m] = {"delta": float(diff.mean()), "ci": bootstrap_ci(diff), "p_t": float(p_t), "p_w": float(p_w),
                  "d": float(diff.mean() / (diff.std(ddof=1) + 1e-12)),
                  "win": float((diff > 0).mean()), "tie": float((diff == 0).mean()), "loss": float((diff < 0).mean())}

    # ---------------- tables (stdout) ----------------
    print("\n" + "=" * 112)
    print(f"{'Method':<36} {'NDCG@5':>8} {'NDCG@10':>8} {'95% CI':>17} {'Kendall τ':>10} {'Pairwise':>9} {'Spearman':>9}")
    print("-" * 112)
    for m in sorted(METHODS, key=lambda m: -results[m]["ndcg@10"]):
        r = results[m]; lo, hi = r["ndcg@10_ci"]
        print(f"{m:<36} {r['ndcg@5']:>8.4f} {r['ndcg@10']:>8.4f}   [{lo:.4f}, {hi:.4f}] {r['kendall_tau']:>+10.4f} "
              f"{r['pairwise_acc']:>9.4f} {r['spearman_global']:>+9.4f}")
    print("=" * 112)
    print(f"\n{OURS} vs. others (paired per-group NDCG@10):")
    for m, s in sorted(sig.items(), key=lambda kv: -kv[1]["delta"]):
        print(f"  {m:<36} Δ={s['delta']:+.4f} [{s['ci'][0]:+.4f}, {s['ci'][1]:+.4f}]  d={s['d']:+.3f}  "
              f"p_t={s['p_t']:.1e}  p_w={s['p_w']:.1e} {stars(s['p_t']):>3}  "
              f"W/T/L={100*s['win']:.0f}/{100*s['tie']:.0f}/{100*s['loss']:.0f}%")

    ORDER = sorted(METHODS, key=lambda m: results[m]["ndcg@10"])

    # ---------------- Fig 1: NDCG@10 with 95% CI ----------------
    fig, ax = plt.subplots(figsize=(8.5, 0.5 * len(ORDER) + 1.2))
    y = np.arange(len(ORDER))
    vals = [results[m]["ndcg@10"] for m in ORDER]
    err = np.array([[results[m]["ndcg@10"] - results[m]["ndcg@10_ci"][0],
                     results[m]["ndcg@10_ci"][1] - results[m]["ndcg@10"]] for m in ORDER]).T
    ax.barh(y, vals, height=0.62, color=[color_of(m) for m in ORDER], edgecolor=C_SURF, linewidth=2)
    ax.errorbar(vals, y, xerr=err, fmt="none", ecolor=C_INK2, elinewidth=1.2, capsize=3)
    for yi, m in enumerate(ORDER):
        ax.text(results[m]["ndcg@10_ci"][1] + 0.003, yi, f"{results[m]['ndcg@10']:.3f}", va="center",
                fontsize=9, color=C_INK, fontweight="bold" if m == OURS else "normal")
    ax.set_yticks(y, ORDER); ax.grid(axis="y", visible=False)
    ax.set_xlim(max(0, min(results[m]["ndcg@10_ci"][0] for m in ORDER) - 0.03), 1.0)
    ax.set_xlabel("NDCG@10 (mean over test groups, 95% bootstrap CI)")
    ax.set_title(f"CNN news-value ranking: {OURS} (BBC-trained) vs. baselines")
    save(fig, "fig1_ndcg10_comparison")

    # ---------------- Fig 2: all metrics, small multiples ----------------
    panels = [("ndcg@5", "NDCG@5"), ("ndcg@10", "NDCG@10"), ("kendall_tau", "Kendall τ"),
              ("pairwise_acc", "Pairwise accuracy"), ("spearman_global", "Spearman ρ (global)")]
    fig, axes = plt.subplots(1, len(panels), figsize=(18, 0.5 * len(ORDER) + 1.4), sharey=True)
    for ax, (key, title) in zip(axes, panels):
        v = [results[m][key] for m in ORDER]
        ax.barh(np.arange(len(ORDER)), v, height=0.62, color=[color_of(m) for m in ORDER],
                edgecolor=C_SURF, linewidth=2)
        ax.axvline(0, color=C_INK2, linewidth=0.8)
        ax.set_title(title); ax.grid(axis="y", visible=False)
        span = max(abs(x) for x in v) or 1
        for yi, x in enumerate(v):
            ax.text(x + (0.02 if x >= 0 else -0.02) * span, yi, f"{x:.3f}", va="center",
                    ha="left" if x >= 0 else "right", fontsize=8, color=C_INK)
        ax.set_xlim(min(0, min(v)) - 0.3 * span, max(0, max(v)) + 0.3 * span)
    axes[0].set_yticks(np.arange(len(ORDER)), ORDER)
    fig.suptitle("All metrics on the CNN test set", fontweight="bold", color=C_INK)
    plt.tight_layout()
    save(fig, "fig2_all_metrics")

    # ---------------- Fig 3: paired Δ forest ----------------
    others = sorted(sig, key=lambda m: sig[m]["delta"])
    fig, ax = plt.subplots(figsize=(8.5, 0.48 * len(others) + 1.4))
    for yi, m in enumerate(others):
        s = sig[m]
        ax.plot(s["ci"], [yi, yi], color=C_OURS, linewidth=2, solid_capstyle="round")
        ax.plot(s["delta"], yi, "o", markersize=8, color=C_OURS, markeredgecolor=C_SURF, markeredgewidth=2)
        ax.text(max(s["ci"][1], 0) + 0.003, yi, f"{s['delta']:+.3f} {stars(s['p_t'])}", va="center",
                fontsize=9, color=C_INK)
    ax.axvline(0, color=C_INK2, linewidth=1, linestyle="--")
    ax.set_yticks(range(len(others)), others); ax.grid(axis="y", visible=False)
    ax.set_xlim(min(0, min(sig[m]["ci"][0] for m in others)) - 0.01, max(sig[m]["ci"][1] for m in others) + 0.04)
    ax.set_xlabel(f"Δ NDCG@10 ({OURS} − method), paired over {len(test_groups)} groups, 95% CI")
    ax.set_title(f"{OURS} wins right of 0, loses left")
    save(fig, "fig3_paired_delta_forest")

    # ---------------- Fig 4: per-group distribution ----------------
    od = ORDER[::-1]
    fig, ax = plt.subplots(figsize=(max(10, 1.35 * len(od)), 4.8))
    bp = ax.boxplot([pg[m] for m in od], tick_labels=od, patch_artist=True, widths=0.55, showfliers=False,
                    medianprops=dict(color=C_INK, linewidth=1.5),
                    whiskerprops=dict(color=C_INK2), capprops=dict(color=C_INK2))
    for patch, m in zip(bp["boxes"], od):
        patch.set_facecolor(color_of(m)); patch.set_edgecolor(C_INK2)
    ax.plot(range(1, len(od) + 1), [pg[m].mean() for m in od], "D", color=C_SURF,
            markeredgecolor=C_INK, markersize=6)
    ax.set_ylabel("NDCG@10 per group"); ax.grid(axis="x", visible=False)
    ax.set_title("Per-group NDCG@10 (box = IQR, line = median, ◇ = mean)")
    plt.xticks(rotation=18, ha="right")
    save(fig, "fig4_per_group_distribution")

    # ---------------- Fig 5: Spearman with each news value ----------------
    dim = pd.DataFrame({m: {**results[m]["spearman_by_news_value"], TARGET: results[m]["spearman_global"]}
                        for m in METHODS}).T.loc[ORDER[::-1]]
    vmax = max(0.3, np.abs(dim.values).max())
    fig, ax = plt.subplots(figsize=(11.5, 0.55 * len(dim) + 1.6))
    im = ax.imshow(dim.values, cmap=DIVERGING, vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(dim.shape[1]), [c.replace("_", " ") for c in dim.columns], rotation=25, ha="right")
    ax.set_yticks(range(dim.shape[0]), dim.index); ax.grid(False)
    for i in range(dim.shape[0]):
        for j in range(dim.shape[1]):
            ax.text(j, i, f"{dim.values[i, j]:+.2f}", ha="center", va="center", fontsize=8.5, color=C_INK)
    ax.axvline(dim.shape[1] - 1.5, color=C_SURF, linewidth=3)
    cb = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02); cb.set_label("Spearman ρ"); cb.outline.set_visible(False)
    ax.set_title("Which news values does each method's score track?")
    save(fig, "fig5_news_value_correlation")

    # ---------------- Fig 6: score vs gold, ours vs best baseline ----------------
    best_base = max(BASELINES, key=lambda m: results[m]["ndcg@10"])
    gold = df["gold"].to_numpy(dtype=float)
    jitter = np.random.default_rng(SEED).uniform(-0.25, 0.25, len(gold))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    for ax, m in zip(axes, [best_base, OURS]):
        pct = pd.Series(df[cols[m]].to_numpy(dtype=float)).rank(pct=True).to_numpy()
        ax.scatter(gold + jitter, pct, s=8, alpha=0.25, color=color_of(m), linewidths=0)
        bins = np.unique(gold)
        ax.plot(bins, [pct[gold == b].mean() for b in bins], color=C_INK, linewidth=2, label="mean per gold value")
        ax.set_title(f"{m}  (ρ = {results[m]['spearman_global']:+.3f})", fontsize=11)
        ax.set_xlabel(f"Gold score ({TARGET} of news values)"); ax.legend(loc="upper left")
    axes[0].set_ylabel("Predicted score (percentile)")
    fig.suptitle("Predicted vs. gold on every test article", fontweight="bold", color=C_INK)
    plt.tight_layout()
    save(fig, "fig6_score_vs_gold")

    # ---------------- Fig 7: per-section dumbbell ----------------
    sec = pd.DataFrame({"section": [df.loc[r[0], "section"] for r in test_groups],
                        "ours": pg[OURS], "base": pg[best_base]})
    sec = sec.groupby("section").agg(ours=("ours", "mean"), base=("base", "mean"), n=("ours", "size"))
    sec = sec[sec["n"] >= 3].sort_values("ours")
    fig, ax = plt.subplots(figsize=(8.5, 0.34 * len(sec) + 1.5))
    yy = np.arange(len(sec))
    ax.hlines(yy, sec["base"], sec["ours"], color=C_GRID, linewidth=3)
    ax.plot(sec["base"], yy, "o", markersize=8, color=C_BASE, markeredgecolor=C_SURF, markeredgewidth=2, label=best_base)
    ax.plot(sec["ours"], yy, "o", markersize=8, color=C_OURS, markeredgecolor=C_SURF, markeredgewidth=2, label=OURS)
    ax.set_yticks(yy, [f"{s}  (n={n})" for s, n in zip(sec.index, sec["n"])]); ax.grid(axis="y", visible=False)
    ax.set_xlabel("Mean NDCG@10 over the section's test groups")
    ax.set_title(f"Per-section NDCG@10: {OURS} vs. {best_base}", fontsize=11)
    ax.legend(loc="lower right")
    save(fig, "fig7_per_section")

    # ---------------- Fig 8: learned value weights + per-value transfer matrix ----------------
    import torch
    state = torch.load(os.path.join(CKPT_DIR, ck), map_location="cpu", weights_only=False)["model_state"]
    w = torch.softmax(state["scoring.value_weights"].float(), 0).numpy()
    names = [v.replace("_", " ") for v in NEWS_VALUES]
    pred = np.stack([df[f"{ck}|pred_{v}"].to_numpy(dtype=float) for v in NEWS_VALUES], 1)
    goldv = df[NEWS_VALUES].to_numpy(dtype=float)
    M = np.array([[spearmanr(pred[:, i], goldv[:, j]).correlation for j in range(7)] for i in range(7)])

    fig, axes = plt.subplots(1, 2, figsize=(15, 4.6), gridspec_kw={"width_ratios": [1, 1.5]})
    ax = axes[0]
    o = np.argsort(w)
    ax.barh(np.arange(7), w[o], height=0.62, color=C_OURS, edgecolor=C_SURF, linewidth=2)
    ax.axvline(1 / 7, color=C_INK2, linestyle="--", linewidth=1)
    ax.text(1 / 7, 6.55, " uniform (1/7)", color=C_INK2, fontsize=8, va="bottom")
    for yi, x in enumerate(w[o]):
        ax.text(x + 0.003, yi, f"{x:.3f}", va="center", fontsize=8, color=C_INK)
    ax.set_xlim(0, max(w.max(), 1 / 7) * 1.18)
    ax.set_yticks(np.arange(7), [names[i] for i in o]); ax.grid(axis="y", visible=False)
    ax.set_xlabel("softmax weight"); ax.set_title("Learned news-value weights (scoring head)")

    ax = axes[1]
    vmax = max(0.3, np.nanmax(np.abs(M)))
    im = ax.imshow(M, cmap=DIVERGING, vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(7), names, rotation=25, ha="right"); ax.set_yticks(range(7), [f"pred {n}" for n in names])
    ax.grid(False)
    for i in range(7):
        for j in range(7):
            ax.text(j, i, f"{M[i, j]:+.2f}", ha="center", va="center", fontsize=8.5, color=C_INK,
                    fontweight="bold" if i == j else "normal")
    ax.set_xlabel("CNN gold news value")
    cb = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02); cb.set_label("Spearman ρ"); cb.outline.set_visible(False)
    ax.set_title("BBC-trained value heads vs. CNN labels (diagonal = transfer)")
    plt.tight_layout()
    save(fig, "fig8_value_head_analysis")

    # ---------------- exports ----------------
    esc = lambda s: s.replace("_", "\\_").replace("&", "\\&").replace("→", "$\\to$")
    keys = ["ndcg@5", "ndcg@10", "kendall_tau", "pairwise_acc", "spearman_global"]
    best = {k: max(results[m][k] for m in METHODS) for k in keys}
    latex  = "\\begin{table}[t]\n\\centering\n\\small\n\\begin{tabular}{llccccc}\n\\toprule\n"
    latex += ("\\textbf{Model} & \\textbf{Train} & \\textbf{NDCG@5} & \\textbf{NDCG@10} & \\textbf{Kendall $\\tau$} & "
              "\\textbf{Pair. Acc.} & \\textbf{Spearman $\\rho$} \\\\\n\\midrule\n")
    train_of = lambda m: "CNN" if "CNN-trained" in m else ("BBC" if ("BBC-trained" in m or m.startswith(OURS)) else "--")
    for m in sorted(METHODS, key=lambda m: -results[m]["ndcg@10"]):
        r = results[m]
        cell = lambda k: ("\\textbf{%.3f}" % r[k]) if r[k] >= best[k] else "%.3f" % r[k]
        mark = f"$^{{{stars(sig[m]['p_t'])}}}$" if m in sig and stars(sig[m]["p_t"]) != "ns" else ""
        name = f"\\textbf{{{esc(m)}}}" if m == OURS else esc(m) + mark
        latex += f"{name} & {train_of(m)} & " + " & ".join(cell(k) for k in keys) + " \\\\\n"
    latex += "\\bottomrule\n\\end{tabular}\n"
    latex += (f"\\caption{{News-value ranking on the CNN test split ({len(test_groups)} groups, {len(df)} articles; "
              f"target: {TARGET} of 7 news values). Train = labelled data the method was fit on. "
              f"Stars: paired t-test on per-group NDCG@10 vs.\\ {OURS} "
              f"($^{{*}}p<.05$, $^{{**}}p<.01$, $^{{***}}p<.001$).}}\n\\label{{tab:cnn_v3_main}}\n\\end{{table}}")
    with open(os.path.join(OUTPUT_DIR, "table_cnn_v3_main.tex"), "w", encoding="utf-8") as f:
        f.write(latex)
    with open(os.path.join(OUTPUT_DIR, "results_cnn_v3_figures.json"), "w", encoding="utf-8") as f:
        json.dump({"main_checkpoint": ck, "columns": cols,
                   "metrics": {m: {k: v for k, v in results[m].items() if k != "per_group_ndcg10"} for m in METHODS},
                   "significance_vs_ours": sig,
                   "value_weights": dict(zip(NEWS_VALUES, w.round(5).tolist())),
                   "value_transfer_matrix": {"rows_pred": NEWS_VALUES, "cols_gold": NEWS_VALUES,
                                             "spearman": np.round(M, 4).tolist()}}, f, indent=2)
    print(f"\nSaved figures to {FIG_DIR}\nSaved table_cnn_v3_main.tex and results_cnn_v3_figures.json")

if __name__ == "__main__":
    main()

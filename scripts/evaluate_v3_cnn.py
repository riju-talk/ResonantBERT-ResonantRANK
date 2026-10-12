#!/usr/bin/env python3
"""
Evaluate every ResonantBERT v3 checkpoint on the CNN Track A test split, using
exactly the groups, gold score and metrics the baselines were measured with.

Reads (written by the CNN baseline notebook):
  eval_results/cnn/cnn_split.json               test ids + ranked test lists
  eval_results/cnn/test_scores_cnn_baselines.csv baseline scores per test article

Scores each checkpoint three ways:
  concept   - the encoder's concept_score (what the listwise losses trained)
  value_sum - sum of the 7 predicted news-value scores (same construct as the gold)
  ranker    - RankingTower on the viral embedding, for checkpoints that have a ranker

Writes eval_results/cnn_v3/: summary table, per-article scores, per-value fit,
paired significance vs. the best baseline, and a checkpoint-selection figure.

Usage:
  python evaluate_v3_cnn.py
  python evaluate_v3_cnn.py --only BERTv3-step1254.pt BERTv3-step1881.pt
"""

import argparse, gc, glob, json, os, re, types
import numpy as np
import pandas as pd
import torch

# transformers 5.x imports a symbol newer torch removed
if not hasattr(torch, "_utils"):
    torch._utils = types.ModuleType("torch._utils")
if not hasattr(torch._utils, "_get_device_index"):
    def _get_device_index(device, optional=False, allow_cpu=False):
        if device is None:
            if optional: return None
            raise ValueError("device is None")
        if isinstance(device, str): device = torch.device(device)
        if isinstance(device, torch.device):
            if device.type == "cpu":
                if allow_cpu: return -1
                raise RuntimeError("CPU not allowed")
            return device.index if device.index is not None else 0
        return device
    torch._utils._get_device_index = _get_device_index

import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM, AutoTokenizer
from scipy.stats import kendalltau, spearmanr, ttest_rel, wilcoxon
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================
# CONFIG (model config must match ResonantBERTv3_KAGGLE_Training.ipynb)
# ============================================================
REPO_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT_DIR     = os.environ.get("CKPT_DIR", os.path.join(REPO_DIR, "model", "checkpoints_v3"))
CNN_CSV      = os.path.join(REPO_DIR, "cnn_mass_dataset", "cnn_labels_with_content.csv")
BASELINE_DIR = os.path.join(REPO_DIR, "eval_results", "cnn")
OUTPUT_DIR   = os.path.join(REPO_DIR, "eval_results", "cnn_v3")

SEED           = 42
TOKENIZER_NAME = "answerdotai/ModernBERT-base"
MAX_SEQ_LEN    = 512
HIDDEN_DIM     = 768
FACTOR_DIM     = 128
NUM_BINS       = 6
LORA_R, LORA_ALPHA, LORA_DROPOUT = 64, 128, 0.05
SCORE_BATCH    = 32
N_BOOT         = 2000

# Order of the model's value heads (training order), and order of gold columns in the CSV
NEWS_VALUES = ["impact", "conflict", "novelty", "proximity",
               "prominence", "timeliness", "human_interest"]
TOKEN_VALUES = {"conflict", "prominence", "human_interest", "novelty"}

DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP_ON  = DEVICE.type == "cuda"
torch.manual_seed(SEED); np.random.seed(SEED)

# ============================================================
# MODEL CLASSES (identical to training)
# ============================================================
class AttentionPooling(nn.Module):
    def __init__(self, d=768):
        super().__init__(); self.attn = nn.Linear(d, 1)
    def forward(self, reps, mask):
        w = self.attn(reps).squeeze(-1)
        w = w.masked_fill(mask == 0, torch.finfo(w.dtype).min)
        return torch.bmm(torch.softmax(w, -1).unsqueeze(1), reps).squeeze(1)

class SegmentAwareEncoder(nn.Module):
    def __init__(self, name=TOKENIZER_NAME, d=HIDDEN_DIM):
        super().__init__()
        self.backbone = AutoModelForMaskedLM.from_pretrained(name)
        self.pool_h = AttentionPooling(d); self.pool_b = AttentionPooling(d)
        self.pool_j = AttentionPooling(d)
    def forward(self, input_ids, attention_mask, sep_positions):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask,
                            output_hidden_states=True)
        t = out.hidden_states[-1]
        S = t.shape[1]
        pos = torch.arange(S, device=t.device).unsqueeze(0)
        sep = sep_positions.unsqueeze(1)
        hm = (pos >= 1) & (pos < sep) & (attention_mask == 1)
        bm = (pos > sep) & (attention_mask == 1)
        hm = torch.where(hm.any(1, keepdim=True), hm, attention_mask.bool())
        bm = torch.where(bm.any(1, keepdim=True), bm, attention_mask.bool())
        return {'e_h': self.pool_h(t, hm.long()), 'e_b': self.pool_b(t, bm.long()),
                'e_j': self.pool_j(t, attention_mask), 'tokens': t,
                'attention_mask': attention_mask}

class OrdinalValueHead(nn.Module):
    def __init__(self, in_dim, hidden=128, n_bins=NUM_BINS):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(0.1))
        self.coral = nn.Linear(hidden, n_bins - 1)
        self.bias = nn.Parameter(torch.zeros(n_bins - 1))
    def forward(self, x):
        z = self.coral(self.shared(x)) + self.bias
        return {'score': torch.sigmoid(z).sum(-1), 'cum_logits': z}

class TokenValueHead(nn.Module):
    def __init__(self, d=HIDDEN_DIM, fd=FACTOR_DIM):
        super().__init__()
        self.q = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.attn = nn.MultiheadAttention(d, 8, batch_first=True)
        self.proj = nn.Sequential(nn.Linear(d, fd), nn.GELU(), nn.LayerNorm(fd))
        self.ordinal = OrdinalValueHead(fd)
    def forward(self, tokens, mask):
        out, _ = self.attn(self.q.expand(tokens.size(0), -1, -1), tokens, tokens,
                           key_padding_mask=(mask == 0))
        vec = self.proj(out.squeeze(1))
        return {'vec': vec, **self.ordinal(vec)}

class SentenceValueHead(nn.Module):
    def __init__(self, d=HIDDEN_DIM, fd=FACTOR_DIM):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d * 2, 256), nn.GELU(), nn.Dropout(0.1),
                                 nn.Linear(256, fd), nn.LayerNorm(fd))
        self.ordinal = OrdinalValueHead(fd)
    def forward(self, e_h, e_b):
        vec = self.net(torch.cat([e_h, e_b], -1))
        return {'vec': vec, **self.ordinal(vec)}

class FactorInteraction(nn.Module):
    def __init__(self, d=FACTOR_DIM, heads=4, p=0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(d, heads, d * 2, p, batch_first=True, activation='gelu')
        self.enc = nn.TransformerEncoder(layer, 1)
    def forward(self, F_): return self.enc(F_)

class GatedScoring(nn.Module):
    def __init__(self, n_values=len(NEWS_VALUES), d=FACTOR_DIM):
        super().__init__()
        self.value_weights = nn.Parameter(torch.ones(n_values) / n_values)
        self.gate = nn.Sequential(nn.Linear(d, n_values), nn.Sigmoid())
        self.embed_proj = nn.Sequential(nn.Linear(d, d), nn.LayerNorm(d))
        self.score_head = nn.Linear(d, 1)
    def forward(self, F_prime, list_ctx=None):
        w = torch.softmax(self.value_weights, 0)
        if list_ctx is not None:
            w = w.unsqueeze(0) * self.gate(list_ctx)
            w = w / w.sum(-1, keepdim=True)
            agg = torch.einsum('lvd,lv->ld', F_prime, w)
        else:
            agg = (F_prime * w.view(1, -1, 1)).sum(1)
        z = F.normalize(self.embed_proj(agg), p=2, dim=-1)
        return {'viral_embedding': z, 'concept_score': self.score_head(z).squeeze(-1),
                'factor_weights': w}

class ListwideHead(nn.Module):
    def __init__(self, d=FACTOR_DIM):
        super().__init__()
        self.cls = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, 4, d * 2, 0.1, batch_first=True, activation='gelu')
        self.tf = nn.TransformerEncoder(layer, 1)
        self.ordinal = OrdinalValueHead(d, 128)
    def forward(self, items):
        x = torch.cat([items, self.cls.expand(items.size(0), -1, -1)], 1)
        return self.ordinal(self.tf(x)[:, -1])['score']

class ResonantBERTv3(nn.Module):
    def __init__(self, backbone=TOKENIZER_NAME, d=HIDDEN_DIM, fd=FACTOR_DIM):
        super().__init__()
        self.encoder = SegmentAwareEncoder(backbone, d)
        self.value_heads = nn.ModuleDict({
            v: (TokenValueHead(d, fd) if v in TOKEN_VALUES else SentenceValueHead(d, fd))
            for v in NEWS_VALUES})
        self.interaction = FactorInteraction(fd)
        self.scoring = GatedScoring(len(NEWS_VALUES), fd)
        self.listwide = ListwideHead(fd)
        self.risk_head = nn.Sequential(nn.Linear(d, 128), nn.GELU(), nn.Dropout(0.1),
                                       nn.Linear(128, 1), nn.Sigmoid())
    def forward(self, input_ids, attention_mask, sep_positions):
        enc = self.encoder(input_ids, attention_mask, sep_positions)
        vecs, scores = [], []
        for v in NEWS_VALUES:
            h = self.value_heads[v]
            r = h(enc['tokens'], enc['attention_mask']) if isinstance(h, TokenValueHead) \
                else h(enc['e_h'], enc['e_b'])
            vecs.append(r['vec']); scores.append(r['score'])
        F_prime = self.interaction(torch.stack(vecs, 1))
        s = self.scoring(F_prime)
        return {'viral_embedding': s['viral_embedding'], 'concept_score': s['concept_score'],
                'factor_weights': s['factor_weights'], 'value_scores': torch.stack(scores, 1)}

class RankingTowerV2(nn.Module):
    def __init__(self, in_dim=FACTOR_DIM, p_drop=0.1):
        super().__init__()
        self.tower = nn.Sequential(nn.Linear(in_dim, 64), nn.GELU(), nn.Dropout(p_drop), nn.Linear(64, 1))
        self.temperature = nn.Parameter(torch.tensor(1.0))
    def forward(self, z):
        return self.tower(z).squeeze(-1)   # raw logit; sigmoid(raw/T) is monotone, so ranking is identical

def build_model():
    import peft.import_utils
    try:
        import peft.tuners.lora.torchao as _lora_torchao
        peft.import_utils.is_torchao_available = lambda: False
        _lora_torchao.is_torchao_available = lambda: False
    except Exception: pass
    from peft import LoraConfig, get_peft_model
    model = ResonantBERTv3()
    preferred = ['Wqkv', 'Wo', 'Wi', 'query', 'key', 'value', 'dense',
                 'q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
    present = {n.split('.')[-1] for n, m in model.encoder.backbone.named_modules()
               if isinstance(m, nn.Linear) and 'decoder' not in n}
    targets = [p for p in preferred if p in present] or sorted(present)
    model.encoder.backbone = get_peft_model(model.encoder.backbone, LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT, bias='none',
        target_modules=targets))
    return model.to(DEVICE).eval()

def load_checkpoint(model, path):
    state = torch.load(path, map_location='cpu', weights_only=False)['model_state']
    missing, unexpected = model.load_state_dict(state, strict=False)
    bad_missing = [k for k in missing if 'lora_' in k or not k.startswith('encoder.backbone')]
    assert not unexpected, f"{os.path.basename(path)}: unexpected keys {unexpected[:5]}"
    assert not bad_missing, f"{os.path.basename(path)}: trained weights missing {bad_missing[:5]}"
    n = len(state); del state; gc.collect()
    if DEVICE.type == "cuda": torch.cuda.empty_cache()
    return n

# ============================================================
# METRICS (identical to the CNN baseline notebook)
# ============================================================
def dcg_at_k(gains, k):
    gains = gains[:k]
    return float((gains / np.log2(np.arange(2, gains.size + 2))).sum())

def ndcg_at_k(pred, gold, k):
    gains = gold - gold.min()
    idcg = dcg_at_k(np.sort(gains)[::-1], k)
    return dcg_at_k(gains[np.argsort(-pred, kind="stable")], k) / idcg if idcg > 0 else 0.0

def kendall_tau_metric(pred, gold):
    tau, _ = kendalltau(pred, gold)
    return float(tau) if not np.isnan(tau) else 0.0

def pairwise_accuracy(pred, gold):
    correct, total = 0.0, 0
    for i in range(len(pred)):
        for j in range(i + 1, len(pred)):
            if gold[i] == gold[j]: continue
            total += 1
            d = (pred[i] - pred[j]) * (gold[i] - gold[j])
            correct += 1.0 if d > 0 else (0.5 if pred[i] == pred[j] else 0.0)
    return correct / max(total, 1)

def bootstrap_ci(values, n_boot=N_BOOT, seed=SEED):
    values = np.asarray(values)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(n_boot, len(values)))].mean(1)
    return [float(x) for x in np.percentile(means, [2.5, 97.5])]

def evaluate_scores(scores, test_df, test_groups):
    gold = test_df["gold"].to_numpy(dtype=float)
    pg = []
    for rows in test_groups:
        p, g = scores[rows], gold[rows]
        pg.append({"ndcg5": ndcg_at_k(p, g, 5), "ndcg10": ndcg_at_k(p, g, 10),
                   "tau": kendall_tau_metric(p, g), "pa": pairwise_accuracy(p, g)})
    col = lambda k: np.array([m[k] for m in pg])
    return {"ndcg@5": float(col("ndcg5").mean()), "ndcg@10": float(col("ndcg10").mean()),
            "ndcg@10_ci": bootstrap_ci(col("ndcg10")),
            "kendall_tau": float(col("tau").mean()), "pairwise_acc": float(col("pa").mean()),
            "spearman_global": float(np.nan_to_num(spearmanr(scores, gold).correlation)),
            "spearman_by_news_value": {v: float(np.nan_to_num(spearmanr(scores, test_df[v]).correlation))
                                       for v in NEWS_VALUES},
            "per_group_ndcg10": col("ndcg10").tolist()}

# ============================================================
# MAIN
# ============================================================
def step_of(path):
    m = re.search(r"step(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else -1

@torch.no_grad()
def score_checkpoint(model, tokenizer, test_df, rankers):
    sep_token = tokenizer.sep_token or "[SEP]"
    sep_id = tokenizer.sep_token_id
    heads, bodies = test_df["headline"].tolist(), test_df["body"].tolist()
    out = {"concept": [], "value_sum": [], "values": []}
    for name in rankers: out[name] = []
    for i in range(0, len(heads), SCORE_BATCH):
        texts = [f"{h} {sep_token} {b}" for h, b in zip(heads[i:i+SCORE_BATCH], bodies[i:i+SCORE_BATCH])]
        enc = tokenizer(texts, padding=True, truncation=True, max_length=MAX_SEQ_LEN, return_tensors="pt")
        seps = []
        for ids in enc["input_ids"]:
            locs = (ids == sep_id).nonzero(as_tuple=True)[0]
            seps.append(locs[0].item() if len(locs) >= 1 else 1)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=AMP_ON):
            o = model(enc["input_ids"].to(DEVICE), enc["attention_mask"].to(DEVICE),
                      torch.tensor(seps, dtype=torch.long, device=DEVICE))
        z = o["viral_embedding"].float()
        vs = o["value_scores"].float()
        out["concept"].append(o["concept_score"].float().cpu().numpy())
        out["value_sum"].append(vs.sum(1).cpu().numpy())
        out["values"].append(vs.cpu().numpy())
        for name, tower in rankers.items():
            out[name].append(tower(z).float().cpu().numpy())
        print(f"    scored {min(i+SCORE_BATCH, len(heads))}/{len(heads)}", end="\r")
    print()
    return {k: np.concatenate(v) for k, v in out.items()}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="checkpoint file names to evaluate (default: all)")
    args = ap.parse_args()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---- test split + baselines (identical setup to the baseline notebook) ----
    with open(os.path.join(BASELINE_DIR, "cnn_split.json"), encoding="utf-8") as f:
        split = json.load(f)
    base = pd.read_csv(os.path.join(BASELINE_DIR, "test_scores_cnn_baselines.csv"), encoding="utf-8")
    assert base["id"].tolist() == split["test_ids"], "baseline scores and split out of sync"
    BASELINES = [c for c in base.columns if c not in ["id", "section", "gold"] + NEWS_VALUES]
    cnn = pd.read_csv(CNN_CSV, encoding="utf-8").set_index("id")
    test_df = base.copy()
    test_df["headline"] = cnn.loc[test_df["id"], "title"].astype(str).values
    test_df["body"]     = cnn.loc[test_df["id"], "content"].astype(str).values
    row = {i: r for r, i in enumerate(test_df["id"])}
    test_groups = [np.array([row[i] for i in g]) for g in split["test_groups"]]
    print(f"CNN test: {len(test_df)} articles, {len(test_groups)} groups, target={split['target']}")

    results = {m: evaluate_scores(test_df[m].to_numpy(dtype=float), test_df, test_groups) for m in BASELINES}
    best_base = max(BASELINES, key=lambda m: results[m]["ndcg@10"])
    print(f"Best baseline: {best_base} (NDCG@10={results[best_base]['ndcg@10']:.4f}, "
          f"rho={results[best_base]['spearman_global']:.4f})")

    # ---- checkpoints + rankers ----
    ckpts = sorted(glob.glob(os.path.join(CKPT_DIR, "BERTv3-*.pt")), key=lambda p: (step_of(p), "final" in p))
    if args.only: ckpts = [p for p in ckpts if os.path.basename(p) in args.only]
    assert ckpts, f"No BERTv3-*.pt in {CKPT_DIR}"
    final_name = next((os.path.basename(p) for p in ckpts if "final" in p), os.path.basename(ckpts[-1]))
    ranker_for = {}   # encoder checkpoint name -> {ranker label: tower}
    for rp in sorted(glob.glob(os.path.join(CKPT_DIR, "RANKERv2-*.pt"))):
        rc = torch.load(rp, map_location="cpu", weights_only=False)
        # Rankers from retrain_ranker.py record their encoder; the training notebook's ranker
        # was trained on the final encoder.
        base_ckpt = rc.get("base_checkpoint", final_name)
        tower = RankingTowerV2(); tower.load_state_dict(rc["model_state"], strict=True)
        ranker_for.setdefault(base_ckpt, {})[f"ranker:{os.path.basename(rp)[:-3]}"] = tower.to(DEVICE).eval()
        print(f"Ranker {os.path.basename(rp)} -> encoder {base_ckpt}")

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    model = build_model()
    gold_vals = test_df[NEWS_VALUES].to_numpy(dtype=float)
    rows, value_fit, score_cols = [], {}, {}

    for path in ckpts:
        name = os.path.basename(path)
        n = load_checkpoint(model, path)
        print(f"\n{name}: {n} trained tensors loaded")
        rankers = ranker_for.get(name, {})
        out = score_checkpoint(model, tokenizer, test_df, rankers)

        pv = out["values"]
        value_fit[name] = {v: {"spearman": float(np.nan_to_num(spearmanr(pv[:, j], gold_vals[:, j]).correlation)),
                               "mae": float(np.abs(pv[:, j] - gold_vals[:, j]).mean())}
                           for j, v in enumerate(NEWS_VALUES)}
        for j, v in enumerate(NEWS_VALUES):
            score_cols[f"{name}|pred_{v}"] = pv[:, j]

        for head in ["concept", "value_sum", *rankers]:
            label = f"{name} [{head}]"
            r = evaluate_scores(out[head], test_df, test_groups)
            results[label] = r
            score_cols[label] = out[head]
            rows.append({"checkpoint": name, "step": step_of(path), "head": head, **{k: r[k] for k in
                         ["ndcg@5", "ndcg@10", "kendall_tau", "pairwise_acc", "spearman_global"]}})
            print(f"  [{head:<28}] NDCG@10={r['ndcg@10']:.4f}  tau={r['kendall_tau']:+.4f}  "
                  f"PA={r['pairwise_acc']:.4f}  rho={r['spearman_global']:+.4f}")

    # ---- paired significance vs. best baseline ----
    b = np.array(results[best_base]["per_group_ndcg10"])
    for row_ in rows:
        a = np.array(results[f"{row_['checkpoint']} [{row_['head']}]"]["per_group_ndcg10"])
        diff = a - b
        _, p_t = ttest_rel(a, b)
        try:    _, p_w = wilcoxon(a, b)
        except ValueError: p_w = float("nan")
        row_.update({"delta_ndcg10_vs_best": float(diff.mean()), "delta_ci95": bootstrap_ci(diff),
                     "p_ttest": float(p_t), "p_wilcoxon": float(p_w)})

    summary = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print("\n" + "=" * 120)
    print(f"v3 CHECKPOINTS ON CNN TEST (best baseline = {best_base}: NDCG@10 {results[best_base]['ndcg@10']:.4f}, "
          f"tau {results[best_base]['kendall_tau']:+.4f}, rho {results[best_base]['spearman_global']:+.4f})")
    print("=" * 120)
    print(summary.drop(columns=["delta_ci95"]).sort_values("ndcg@10", ascending=False)
          .to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nPER-NEWS-VALUE FIT (predicted value head vs. CNN label): Spearman / MAE")
    fit = pd.DataFrame({ck: {v: f"{d['spearman']:+.3f} / {d['mae']:.2f}" for v, d in vf.items()}
                        for ck, vf in value_fit.items()}).T
    print(fit.to_string())

    # ---- save ----
    summary.to_csv(os.path.join(OUTPUT_DIR, "v3_checkpoint_summary.csv"), index=False)
    with open(os.path.join(OUTPUT_DIR, "results_cnn_v3.json"), "w", encoding="utf-8") as f:
        json.dump({"best_baseline": best_base,
                   "baselines": {m: {k: v for k, v in results[m].items() if k != "per_group_ndcg10"}
                                 for m in BASELINES},
                   "checkpoints": {k: {kk: vv for kk, vv in v.items() if kk != "per_group_ndcg10"}
                                   for k, v in results.items() if k not in BASELINES},
                   "value_fit": value_fit, "significance_vs_best": rows}, f, indent=2)
    pd.concat([test_df[["id", "section", "gold"] + NEWS_VALUES + BASELINES],
               pd.DataFrame(score_cols)], axis=1).to_csv(
        os.path.join(OUTPUT_DIR, "test_scores_cnn_v3.csv"), index=False)

    # ---- checkpoint-selection figure: metric vs. step per head, baselines as reference lines ----
    C_OURS, C_ALT, C_BASE, C_INK2, C_GRID = "#2a78d6", "#eb6834", "#a3a29c", "#52514e", "#e8e7e3"
    head_color = {"concept": C_OURS, "value_sum": C_ALT}
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.3))
    for ax, (key, title) in zip(axes, [("ndcg@10", "NDCG@10"), ("kendall_tau", "Kendall τ"),
                                       ("spearman_global", "Spearman ρ (global)")]):
        steps = summary[summary["step"] >= 0]
        for head, grp in steps.groupby("head"):
            grp = grp.sort_values(["step", "checkpoint"])
            ax.plot(grp["step"], grp[key], "o-", lw=2, ms=6,
                    color=head_color.get(head, "#1baf7a"), label=head)
        for m in BASELINES:
            if m == best_base or m == "Random":
                ax.axhline(results[m][key], color=C_BASE if m == "Random" else C_INK2,
                           ls="--" if m == best_base else ":", lw=1.2)
                ax.text(ax.get_xlim()[0], results[m][key], f" {m}", va="bottom", fontsize=8, color=C_INK2)
        ax.set_title(title, fontweight="bold"); ax.set_xlabel("training step")
        ax.grid(color=C_GRID); ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("ResonantBERT v3 checkpoints on CNN test (zero-shot) vs. baselines", fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "v3_checkpoint_selection.png"), dpi=200, bbox_inches="tight")
    print(f"\nSaved: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Controlled ablation study for ResonantBERT v3 (train on BBC Track A, test zero-shot on CNN).

Every variant uses the same backbone (ModernBERT-base + LoRA r=64/alpha=128), data,
token budget, optimiser and schedule; only the component under test changes.

Variants
  full            v3: token/sentence value heads + factor interaction + gated concept head
                  + list-wide head; loss = CORAL + ListNet + ApproxNDCG + list-wide
  no_rank         full architecture, pointwise CORAL loss only (no ranking losses)
  no_interaction  factor-interaction transformer replaced by identity
  no_token        all 7 value heads sentence-level (no token-attention heads)
  plain_coral     plain ModernBERT: pooled embedding -> 7 CORAL heads (no v3 architecture)
  plain_mse       plain ModernBERT: pooled embedding -> 1 regression on the summed gold

Splits
  random    BBC (all years) -> CNN test split of the baseline notebook (eval_results/cnn/cnn_split.json)
  temporal  BBC 2017-2022 -> CNN test articles from 2023-2025 (eval_results/cnn/cnn_split_temporal.json)

Differences from the Kaggle v3 run, on purpose:
  * training lists are random articles within a section (the Kaggle run chunked
    gold-sorted articles, so each list held near-ties and the ranking losses had little signal)
  * checkpoint selection uses article-level Spearman on a held-out BBC validation set,
    never the CNN test set
  * a fixed, reduced token budget (--train-groups) so all variants x seeds fit on one GPU

Usage
  python ablation_study.py --variant full --seed 42 --split random
Outputs
  runs/ablations/<split>/<variant>_seed<seed>/{best.pt, train_log.json}
  eval_results/ablations/<split>/<variant>_seed<seed>/{metrics.json, test_scores.csv}
"""

import argparse, json, math, os, random, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from transformers import AutoTokenizer

from evaluate_v3_cnn import (  # model blocks + metrics shared with the v3 CNN evaluation
    SegmentAwareEncoder, OrdinalValueHead, TokenValueHead, SentenceValueHead,
    FactorInteraction, GatedScoring, ListwideHead, NEWS_VALUES, TOKEN_VALUES,
    TOKENIZER_NAME, HIDDEN_DIM, FACTOR_DIM, NUM_BINS, LORA_R, LORA_ALPHA, LORA_DROPOUT,
    CNN_CSV, BASELINE_DIR, evaluate_scores)

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BBC_CSV  = os.path.join(REPO_DIR, "bbc_articles", "track_a_all.csv")
VARIANTS = ["full", "no_rank", "no_interaction", "no_token", "plain_coral", "plain_mse"]

# Training recipe (LRs, warmup, schedule, loss weights from ResonantBERTv3_KAGGLE_Training.ipynb)
LR_BACKBONE, LR_HEADS, WEIGHT_DECAY, MAX_GRAD_NORM = 1e-5, 2e-4, 0.01, 1.0
WARMUP_RATIO = 0.06
W_POINT, W_LIST, W_NDCG, W_WIDE = 1.0, 1.0, 0.5, 0.25
DATA_SEED = 42           # data split is fixed across model seeds; only init/order vary with --seed

# ============================================================
# MODEL
# ============================================================
class AblationModel(nn.Module):
    def __init__(self, variant):
        super().__init__()
        self.variant = variant
        self.encoder = SegmentAwareEncoder(TOKENIZER_NAME, HIDDEN_DIM)
        self.plain = variant.startswith("plain")
        if variant == "plain_coral":
            self.heads = nn.ModuleList([OrdinalValueHead(HIDDEN_DIM) for _ in NEWS_VALUES])
        elif variant == "plain_mse":
            self.reg = nn.Sequential(nn.Dropout(0.1), nn.Linear(HIDDEN_DIM, 1))
        else:
            token = TOKEN_VALUES if variant != "no_token" else set()
            self.value_heads = nn.ModuleDict({
                v: (TokenValueHead(HIDDEN_DIM, FACTOR_DIM) if v in token else SentenceValueHead(HIDDEN_DIM, FACTOR_DIM))
                for v in NEWS_VALUES})
            self.interaction = nn.Identity() if variant == "no_interaction" else FactorInteraction(FACTOR_DIM)
            self.scoring = GatedScoring(len(NEWS_VALUES), FACTOR_DIM)
            self.listwide = ListwideHead(FACTOR_DIM)

    def forward(self, input_ids, attention_mask, sep_positions, group_size=None):
        enc = self.encoder(input_ids, attention_mask, sep_positions)
        if self.variant == "plain_mse":
            return {"scalar": self.reg(enc["e_j"]).squeeze(-1)}
        if self.variant == "plain_coral":
            outs = [h(enc["e_j"]) for h in self.heads]
            return {"value_scores": torch.stack([o["score"] for o in outs], 1),
                    "value_logits": torch.stack([o["cum_logits"] for o in outs], 1)}
        vecs, scores, logits = [], [], []
        for v in NEWS_VALUES:
            h = self.value_heads[v]
            r = h(enc["tokens"], enc["attention_mask"]) if isinstance(h, TokenValueHead) else h(enc["e_h"], enc["e_b"])
            vecs.append(r["vec"]); scores.append(r["score"]); logits.append(r["cum_logits"])
        F_prime = self.interaction(torch.stack(vecs, 1))
        s = self.scoring(F_prime)
        out = {"concept_score": s["concept_score"], "value_scores": torch.stack(scores, 1),
               "value_logits": torch.stack(logits, 1)}
        if group_size:   # lists are contiguous, equal-size blocks of the batch
            items = F_prime.mean(1).view(-1, group_size, F_prime.size(-1))
            out["listwide_score"] = self.listwide(items)
        return out

def build(variant, device):
    import peft.import_utils
    try:
        import peft.tuners.lora.torchao as _t
        peft.import_utils.is_torchao_available = lambda: False
        _t.is_torchao_available = lambda: False
    except Exception: pass
    from peft import LoraConfig, get_peft_model
    model = AblationModel(variant)
    bb = model.encoder.backbone
    bb.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    preferred = ['Wqkv', 'Wo', 'Wi', 'query', 'key', 'value', 'dense']
    present = {n.split('.')[-1] for n, m in bb.named_modules() if isinstance(m, nn.Linear) and 'decoder' not in n}
    targets = [p for p in preferred if p in present] or sorted(present)
    model.encoder.backbone = get_peft_model(bb, LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT, bias='none', target_modules=targets))
    for n, p in model.named_parameters():
        if not n.startswith("encoder.backbone"): p.requires_grad = True
    return model.to(device)

def trainable_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            if "lora_" in k or not k.startswith("encoder.backbone")}

# ============================================================
# LOSSES
# ============================================================
def coral_loss(cum_logits, targets):
    t = targets.clamp(0, NUM_BINS - 1).long()
    bins = torch.arange(1, NUM_BINS, device=t.device)
    return F.binary_cross_entropy_with_logits(cum_logits.float(), (t.unsqueeze(1) >= bins).float())

def listnet_top1(scores, targets):
    return -(torch.softmax(targets.float(), -1) * torch.log_softmax(scores.float(), -1)).sum(-1).mean()

def approx_ndcg(scores, targets, alpha=10.0):
    s, y = scores.float(), targets.float()                       # (G, L)
    rank = 1.0 + torch.sigmoid(alpha * (s.unsqueeze(1) - s.unsqueeze(2))).sum(1)
    gains = 2.0 ** (y - y.min(-1, keepdim=True).values) - 1.0   # shift so the floor is 0 (as the metric)
    dcg = (gains / torch.log2(rank + 1.0)).sum(-1)
    ideal = torch.log2(torch.arange(2, y.size(1) + 2, device=y.device).float())
    idcg = (gains.sort(-1, descending=True).values / ideal).sum(-1)
    return -(dcg / idcg.clamp_min(1e-8)).mean()

def compute_loss(variant, out, batch, group_size):
    if variant == "plain_mse":
        return F.mse_loss(out["scalar"].float(), batch["gold"] / 35.0)
    vt = batch["values"]
    loss = W_POINT * sum(coral_loss(out["value_logits"][:, i], vt[:, i]) for i in range(len(NEWS_VALUES))) / len(NEWS_VALUES)
    if variant in ("plain_coral", "no_rank"):
        return loss
    s = out["concept_score"].view(-1, group_size)
    y = batch["gold"].view(-1, group_size)
    loss = loss + W_LIST * listnet_top1(s, y) + W_NDCG * approx_ndcg(s, y)
    loss = loss + W_WIDE * F.mse_loss(out["listwide_score"].float(), y.max(-1).values / 35.0 * (NUM_BINS - 1))
    return loss

# ============================================================
# DATA
# ============================================================
def load_bbc(split, max_val=4000):
    df = pd.read_csv(BBC_CSV, usecols=["id", "title", "content", "date", "section"] + NEWS_VALUES,
                     encoding="utf-8", encoding_errors="replace")
    df = df.dropna(subset=["title", "content"])
    df = df[df["content"].str.strip().str.len() > 20].reset_index(drop=True)
    df["section"] = df["section"].fillna("unknown").astype(str)
    df["year"] = pd.to_numeric(df["date"].astype(str).str[:4], errors="coerce")
    if split == "temporal":
        df = df[df["year"] <= 2022].reset_index(drop=True)
    df["gold"] = df[NEWS_VALUES].sum(axis=1).astype(float)
    perm = np.random.RandomState(DATA_SEED).permutation(len(df))
    n_val = min(max_val, int(0.1 * len(df)))
    return df.iloc[perm[n_val:]].reset_index(drop=True), df.iloc[perm[:n_val]].reset_index(drop=True)

def make_groups(train, n_groups, group_size, seed):
    """Random lists within a section (not gold-sorted chunks), equal size, n_groups total."""
    rng = np.random.RandomState(seed)
    groups = []
    for _, sdf in train.groupby("section", sort=True):
        idx = sdf.index.to_numpy()[rng.permutation(len(sdf))]
        groups += [idx[i:i + group_size] for i in range(0, len(idx) - group_size + 1, group_size)]
    groups = [groups[i] for i in rng.permutation(len(groups))]
    return groups[:n_groups]

def cnn_test(split):
    path = os.path.join(BASELINE_DIR, "cnn_split.json" if split == "random" else "cnn_split_temporal.json")
    if split == "temporal" and not os.path.exists(path):
        make_temporal_split(path)
    with open(path, encoding="utf-8") as f:
        sp = json.load(f)
    cnn = pd.read_csv(CNN_CSV, encoding="utf-8").set_index("id")
    t = cnn.loc[sp["test_ids"]].reset_index()
    t["headline"], t["body"] = t["title"].astype(str), t["content"].astype(str)
    t["gold"] = t[NEWS_VALUES].sum(axis=1).astype(float)
    row = {i: r for r, i in enumerate(t["id"])}
    return t, [np.array([row[i] for i in g]) for g in sp["test_groups"]]

def make_temporal_split(path, group_size=10):
    """CNN test-split articles published 2023-2025, regrouped by section (same rules as the baseline notebook)."""
    with open(os.path.join(BASELINE_DIR, "cnn_split.json"), encoding="utf-8") as f:
        sp = json.load(f)
    cnn = pd.read_csv(CNN_CSV, encoding="utf-8").set_index("id")
    t = cnn.loc[sp["test_ids"]].reset_index()
    t = t[pd.to_numeric(t["date"].astype(str).str[:4], errors="coerce") >= 2023].reset_index(drop=True)
    t["section"] = t["section"].fillna("unknown").astype(str)
    t["gold"] = t[NEWS_VALUES].sum(axis=1)
    rng, groups = np.random.RandomState(DATA_SEED), []
    for _, sdf in t.groupby("section", sort=True):
        rows = sdf.index.to_numpy()[rng.permutation(len(sdf))]
        for i in range(0, len(rows), group_size):
            ch = rows[i:i + group_size]
            if len(ch) >= 2 and t.loc[ch, "gold"].nunique() >= 2:
                groups.append(t.loc[ch, "id"].tolist())
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"target": "sum", "seed": DATA_SEED, "group_size": group_size, "years": "2023-2025",
                   "test_ids": t["id"].tolist(), "test_groups": groups}, f)
    print(f"Wrote temporal CNN split: {len(t)} articles, {len(groups)} groups -> {path}")

def encode(tok, heads, bodies, max_len, device):
    sep = tok.sep_token or "[SEP]"
    enc = tok([f"{h} {sep} {b}" for h, b in zip(heads, bodies)], padding=True, truncation=True,
              max_length=max_len, return_tensors="pt")
    seps = []
    for ids in enc["input_ids"]:
        locs = (ids == tok.sep_token_id).nonzero(as_tuple=True)[0]
        seps.append(locs[0].item() if len(locs) else 1)
    return (enc["input_ids"].to(device), enc["attention_mask"].to(device),
            torch.tensor(seps, dtype=torch.long, device=device))

@torch.no_grad()
def predict(model, tok, frame, max_len, device, bs=32):
    model.eval()
    res = {"value_sum": [], "concept": [], "scalar": [], "values": []}
    for i in range(0, len(frame), bs):
        f = frame.iloc[i:i + bs]
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            o = model(*encode(tok, f["headline"].tolist(), f["body"].tolist(), max_len, device))
        if "value_scores" in o:
            vs = o["value_scores"].float()
            res["value_sum"].append(vs.sum(1).cpu().numpy()); res["values"].append(vs.cpu().numpy())
        if "concept_score" in o: res["concept"].append(o["concept_score"].float().cpu().numpy())
        if "scalar" in o: res["scalar"].append(o["scalar"].float().cpu().numpy())
    model.train()
    return {k: np.concatenate(v) for k, v in res.items() if v}

def primary(pred):
    return pred["scalar"] if "scalar" in pred else pred["value_sum"]

# ============================================================
# MAIN
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=VARIANTS, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split", choices=["random", "temporal"], default="random")
    ap.add_argument("--train-groups", type=int, default=5000, help="lists of --group-size articles (1 epoch)")
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--micro-groups", type=int, default=2, help="lists per forward pass")
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--eval-every", type=int, default=125, help="optimizer steps between BBC-val checks")
    ap.add_argument("--max-val", type=int, default=4000, help="held-out BBC articles for checkpoint selection")
    ap.add_argument("--max-test", type=int, default=None, help="smoke tests only: truncate CNN test groups")
    args = ap.parse_args()

    tag = f"{args.variant}_seed{args.seed}"
    run_dir = os.path.join(REPO_DIR, "runs", "ablations", args.split, tag)
    out_dir = os.path.join(REPO_DIR, "eval_results", "ablations", args.split, tag)
    if os.path.exists(os.path.join(out_dir, "metrics.json")):
        print(f"{tag} ({args.split}) already done, skipping"); return
    os.makedirs(run_dir, exist_ok=True); os.makedirs(out_dir, exist_ok=True)

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train, val = load_bbc(args.split, args.max_val)
    groups = make_groups(train, args.train_groups, args.group_size, args.seed)
    val["headline"], val["body"] = val["title"].astype(str), val["content"].astype(str)
    print(f"[{args.split}/{tag}] BBC train pool={len(train)} -> {len(groups)} lists x {args.group_size} | val={len(val)}")

    tok = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    model = build(args.variant, device)
    lora = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("encoder.backbone")]
    heads = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("encoder.backbone")]
    opt = torch.optim.AdamW([{"params": lora, "lr": LR_BACKBONE}, {"params": heads, "lr": LR_HEADS}],
                            weight_decay=WEIGHT_DECAY)
    n_micro = len(groups) // args.micro_groups
    total = n_micro // args.accum
    warm = int(WARMUP_RATIO * total)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: s / max(warm, 1) if s < warm else
        max(0.1, 0.5 * (1 + math.cos(math.pi * (s - warm) / max(total - warm, 1)))))
    print(f"optimizer steps={total} (micro={args.micro_groups} lists, accum={args.accum}) warmup={warm}")

    log, best_rho, best_state, step, t0 = [], -np.inf, None, 0, time.time()
    gs = args.group_size if args.variant in ("full", "no_interaction", "no_token") else None

    def validate():
        rho = float(spearmanr(primary(predict(model, tok, val, args.max_len, device)), val["gold"]).correlation)
        return rho

    model.train()
    opt.zero_grad(set_to_none=True)
    for mi in range(n_micro):
        idx = np.concatenate(groups[mi * args.micro_groups:(mi + 1) * args.micro_groups])
        b = train.loc[idx]
        batch = {"gold": torch.tensor(b["gold"].values, dtype=torch.float, device=device),
                 "values": torch.tensor(b[NEWS_VALUES].values, dtype=torch.float, device=device)}
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out = model(*encode(tok, b["title"].astype(str).tolist(), b["content"].astype(str).tolist(),
                                args.max_len, device), group_size=gs)
            loss = compute_loss(args.variant, out, batch, args.group_size) / args.accum
        loss.backward()
        if (mi + 1) % args.accum == 0:
            torch.nn.utils.clip_grad_norm_(lora + heads, MAX_GRAD_NORM)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1
            log.append({"step": step, "loss": float(loss.item() * args.accum)})
            if step % 25 == 0:
                rate = (mi + 1) * args.micro_groups * args.group_size / (time.time() - t0)
                eta = (n_micro - mi - 1) * args.micro_groups * args.group_size / rate / 60
                print(f"  step {step:>5}/{total} loss={np.mean([l['loss'] for l in log[-25:]]):.4f} "
                      f"{rate:.1f} art/s, ETA {eta:.0f} min", flush=True)
            if step % args.eval_every == 0 or step == total:
                rho = validate()
                log[-1]["val_spearman"] = rho
                if rho > best_rho:
                    best_rho, best_state = rho, trainable_state(model)
                print(f"  >>> step {step} BBC-val Spearman={rho:.4f} (best {best_rho:.4f})", flush=True)

    torch.save({"model_state": best_state, "variant": args.variant, "seed": args.seed, "split": args.split,
                "best_val_spearman": best_rho, "args": vars(args)}, os.path.join(run_dir, "best.pt"))
    with open(os.path.join(run_dir, "train_log.json"), "w") as f:
        json.dump({"log": log, "best_val_spearman": best_rho, "minutes": (time.time() - t0) / 60}, f)

    # ---- CNN zero-shot test with the BBC-val-selected weights ----
    model.load_state_dict(best_state, strict=False)
    test, test_groups = cnn_test(args.split)
    if args.max_test:
        test_groups = test_groups[:args.max_test]
    pred = predict(model, tok, test, args.max_len, device)
    metrics = {"variant": args.variant, "seed": args.seed, "split": args.split,
               "best_bbc_val_spearman": best_rho, "train_minutes": (time.time() - t0) / 60,
               "n_test": len(test), "n_groups": len(test_groups), "heads": {}}
    scores = test[["id", "section", "gold"] + NEWS_VALUES].copy()
    for head in ["value_sum", "scalar", "concept"]:
        if head not in pred: continue
        r = evaluate_scores(pred[head], test, test_groups)
        metrics["heads"][head] = r
        scores[head] = pred[head]
        print(f"  CNN [{head:<9}] NDCG@10={r['ndcg@10']:.4f} tau={r['kendall_tau']:+.4f} "
              f"PA={r['pairwise_acc']:.4f} rho={r['spearman_global']:+.4f}")
    if "values" in pred:
        metrics["value_fit"] = {v: float(spearmanr(pred["values"][:, j], test[v]).correlation)
                                for j, v in enumerate(NEWS_VALUES)}
        for j, v in enumerate(NEWS_VALUES): scores[f"pred_{v}"] = pred["values"][:, j]
    metrics["primary_head"] = "scalar" if "scalar" in pred else "value_sum"
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    scores.to_csv(os.path.join(out_dir, "test_scores.csv"), index=False)
    print(f"Done {tag} ({args.split}) in {(time.time() - t0) / 60:.1f} min -> {out_dir}")

if __name__ == "__main__":
    main()

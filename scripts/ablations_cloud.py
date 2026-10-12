#!/usr/bin/env python3
"""
ResonantBERT v3 — ABLATION STUDY, single self-contained cloud script
(train on BBC Track A, select on held-out BBC, test zero-shot on CNN)

Runs 16 trainings back to back on the GPU with the most free VRAM:

  split     variant          what it tests                                       seeds
  random    full             ResonantBERT v3 (all components)                     42, 43
  random    plain_coral      plain ModernBERT -> 7 CORAL news-value heads         42, 43
  random    plain_mse        plain ModernBERT -> 1 regression on summed gold      42, 43
  random    no_rank          v3 without ranking losses (CORAL only)               42, 43
  random    no_interaction   v3 without the factor-interaction transformer        42, 43
  random    no_token         v3 with all 7 heads sentence-level                   42, 43
  temporal  full             BBC 2017-2022 -> CNN 2023-2025                       42, 43
  temporal  plain_coral      BBC 2017-2022 -> CNN 2023-2025                       42, 43

Same recipe as the v3 cloud training script (ModernBERT-base, LoRA r=64/alpha=128 on all
linear layers, LR 1e-5 backbone / 2e-4 heads, 6% warmup + cosine to 10%, bf16, loss weights
1.0/1.0/0.5/0.25), with three deliberate fixes:
  * ranking lists are random articles within a section (the v3 run chunked gold-sorted
    articles, so every list held near-ties and the ranking losses had almost no signal)
  * checkpoint selection = article-level Spearman on 4,000 held-out BBC articles (never CNN)
  * the same fixed budget for every variant (TRAIN_GROUPS lists x 8 articles, 1 epoch)

The CNN test splits are rebuilt exactly as in the local baseline notebook and checked
against fingerprints of the local split files, so results are directly comparable.

Resumable: a run whose metrics.json exists is skipped. Just re-run the script.

Copy back: ./working/ablation_results.zip -> unzip in the repo root locally
(it contains eval_results/ablations/<split>/<variant>_seed<seed>/{metrics.json,
test_scores.csv, train_log.json}), then run `python ablation_report.py`.

Usage:
  python ablations_cloud.py
"""

# ============================================================
# 0. TORCH COMPAT PATCH + IMPORTS
# ============================================================
import torch, types
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

import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "scipy", "scikit-learn", "peft", "accelerate"])

import os, json, math, random, gc, time, hashlib, zipfile, warnings
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import kendalltau, spearmanr
from transformers import AutoModelForMaskedLM, AutoTokenizer
warnings.filterwarnings("ignore", message=".*ConstantInputWarning.*")

# ============================================================
# 1. CONFIG — edit here
# ============================================================
SEEDS          = [42, 43]
RUNS           = [("random", v) for v in ["full", "plain_coral", "plain_mse", "no_rank", "no_interaction", "no_token"]] \
               + [("temporal", v) for v in ["full", "plain_coral"]]
TRAIN_GROUPS   = 5000       # lists of 8 BBC articles per run (5000 -> 40k articles, 1 epoch)
GROUP_SIZE     = 8
LISTS_PER_STEP = 8          # lists per optimizer step (micro-batch x accumulation); fixed for comparability
EVAL_EVERY     = 125        # optimizer steps between BBC-val checks (plus one at the end)
MAX_VAL        = 4000       # held-out BBC articles for checkpoint selection
SAVE_CKPTS     = True       # save best LoRA+heads per run (~100 MB each); results zip never includes them
ALLOW_SPLIT_MISMATCH = False
MICRO_OVERRIDE = None      # lists per forward pass (1, 2, 4 or 8); None = auto from free VRAM

REPO_URL   = "https://github.com/riju-talk/ResonantBERT-ResonantRANK.git"
WORK_DIR   = os.path.abspath("./working")
REPO_DIR   = os.path.abspath("./tmp/ResonantBERT-ResonantRANK")
RES_DIR    = os.path.join(WORK_DIR, "eval_results", "ablations")   # copied back
CKPT_DIR   = os.path.join(WORK_DIR, "ablation_checkpoints")
os.makedirs(RES_DIR, exist_ok=True); os.makedirs(CKPT_DIR, exist_ok=True)
os.environ.setdefault("HF_HOME", os.path.abspath("./tmp/hf"))

TOKENIZER_NAME = "answerdotai/ModernBERT-base"
MAX_SEQ_LEN, HIDDEN_DIM, FACTOR_DIM, NUM_BINS = 512, 768, 128, 6
LORA_R, LORA_ALPHA, LORA_DROPOUT = 64, 128, 0.05
LR_BACKBONE, LR_HEADS, WEIGHT_DECAY, MAX_GRAD_NORM, WARMUP_RATIO = 1e-5, 2e-4, 0.01, 1.0, 0.06
W_POINT, W_LIST, W_NDCG, W_WIDE = 1.0, 1.0, 0.5, 0.25
DATA_SEED = 42              # data splits fixed across model seeds

NEWS_VALUES  = ["impact", "conflict", "novelty", "proximity", "prominence", "timeliness", "human_interest"]
TOKEN_VALUES = {"conflict", "prominence", "human_interest", "novelty"}

# Fingerprints of the local split files (eval_results/cnn/cnn_split*.json)
SPLIT_FINGERPRINT = {"random": "9b13422954017137", "temporal": "2ddd8fe9f1ed2334"}

# ============================================================
# 2. GPU — largest free VRAM; micro-batch sized to memory
# ============================================================
def pick_largest_free_gpu():
    best_idx, best_free = 0, -1.0
    for i in range(torch.cuda.device_count()):
        try: free, total = torch.cuda.mem_get_info(i)
        except Exception: free = total = torch.cuda.get_device_properties(i).total_memory
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)} | free {free/2**30:.1f} / {total/2**30:.1f} GB")
        if free > best_free: best_idx, best_free = i, free
    return best_idx, best_free

assert torch.cuda.is_available(), "No GPU found"
GPU_IDX, GPU_FREE = pick_largest_free_gpu()
torch.cuda.set_device(GPU_IDX)
DEVICE = torch.device(f"cuda:{GPU_IDX}")
if not torch.cuda.is_bf16_supported():
    print("WARNING: this GPU has no bf16 support (e.g. T4/P100); use A100/L4/A10/H100 for matching numerics")
free_gb = GPU_FREE / 2**30
MICRO = MICRO_OVERRIDE or (8 if free_gb >= 40 else 4 if free_gb >= 20 else 2 if free_gb >= 10 else 1)
assert LISTS_PER_STEP % MICRO == 0, "MICRO must divide LISTS_PER_STEP"
ACCUM = LISTS_PER_STEP // MICRO
print(f"--> GPU {GPU_IDX}: {torch.cuda.get_device_name(GPU_IDX)} ({free_gb:.1f} GB free) | "
      f"micro={MICRO} lists x accum={ACCUM} = {LISTS_PER_STEP} lists/step")

# ============================================================
# 3. REPO + DATA (only the LFS file we need)
# ============================================================
env = {**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"}
if not os.path.exists(REPO_DIR):
    subprocess.run(["git", "clone", "--quiet", REPO_URL, REPO_DIR], check=True, env=env)
else:
    subprocess.run(["git", "-C", REPO_DIR, "pull", "--quiet"], check=True, env=env)
subprocess.run(["git", "-C", REPO_DIR, "lfs", "install", "--local"], check=True)
subprocess.run(["git", "-C", REPO_DIR, "lfs", "pull", "--include", "bbc_articles/track_a_all.csv,cnn_mass_dataset/cnn_labels_with_content.csv"], check=True)

BBC_CSV = os.path.join(REPO_DIR, "bbc_articles", "track_a_all.csv")
CNN_CSV = os.path.join(REPO_DIR, "cnn_mass_dataset", "cnn_labels_with_content.csv")
for p in (BBC_CSV, CNN_CSV):
    assert os.path.exists(p) and os.path.getsize(p) > 1e6, f"{p} missing or still an LFS pointer"

bbc_all = pd.read_csv(BBC_CSV, usecols=["id", "title", "content", "date", "section"] + NEWS_VALUES,
                      encoding="utf-8", encoding_errors="replace")
bbc_all = bbc_all.dropna(subset=["title", "content"])
bbc_all = bbc_all[bbc_all["content"].str.strip().str.len() > 20].reset_index(drop=True)
bbc_all["section"] = bbc_all["section"].fillna("unknown").astype(str)
bbc_all["year"] = pd.to_numeric(bbc_all["date"].astype(str).str[:4], errors="coerce")
bbc_all["gold"] = bbc_all[NEWS_VALUES].sum(axis=1).astype(float)
print(f"BBC Track A: {len(bbc_all)} articles")

def bbc_pool(split):
    """Same BBC train/val split as the local ablation_study.py."""
    df = bbc_all if split == "random" else bbc_all[bbc_all["year"] <= 2022].reset_index(drop=True)
    perm = np.random.RandomState(DATA_SEED).permutation(len(df))
    n_val = min(MAX_VAL, int(0.1 * len(df)))
    return df.iloc[perm[n_val:]].reset_index(drop=True), df.iloc[perm[:n_val]].reset_index(drop=True)

def make_lists(train, seed):
    """Random lists within a section (not gold-sorted chunks)."""
    rng = np.random.RandomState(seed)
    lists = []
    for _, sdf in train.groupby("section", sort=True):
        idx = sdf.index.to_numpy()[rng.permutation(len(sdf))]
        lists += [idx[i:i + GROUP_SIZE] for i in range(0, len(idx) - GROUP_SIZE + 1, GROUP_SIZE)]
    lists = [lists[i] for i in rng.permutation(len(lists))]
    return lists[:TRAIN_GROUPS]

# ---- CNN test splits, rebuilt exactly as the local baseline notebook ----
def cnn_splits():
    cnn = pd.read_csv(CNN_CSV, encoding="utf-8")
    cnn = cnn.dropna(subset=["title", "content"])
    cnn = cnn[cnn["content"].str.strip() != ""].reset_index(drop=True)
    cnn["section"] = cnn["section"].fillna("unknown").astype(str)
    cnn["gold"] = cnn[NEWS_VALUES].sum(axis=1).astype(float)
    idx = np.random.RandomState(42).permutation(len(cnn))
    test = cnn.iloc[np.sort(idx[:int(round(0.10 * len(cnn)))])].reset_index(drop=True)

    def groups_of(t, seed=42, size=10):
        rng, out = np.random.RandomState(seed), []
        for _, sdf in t.groupby("section", sort=True):
            rows = sdf.index.to_numpy()[rng.permutation(len(sdf))]
            for i in range(0, len(rows), size):
                ch = rows[i:i + size]
                if len(ch) >= 2 and t.loc[ch, "gold"].nunique() >= 2:
                    out.append(t.loc[ch, "id"].tolist())
        return out

    temporal = test[pd.to_numeric(test["date"].astype(str).str[:4], errors="coerce") >= 2023].reset_index(drop=True)
    splits = {}
    for name, t in [("random", test), ("temporal", temporal)]:
        g = groups_of(t)
        fp = hashlib.sha256(json.dumps([t["id"].tolist(), g]).encode()).hexdigest()[:16]
        ok = fp == SPLIT_FINGERPRINT[name]
        print(f"CNN {name} test: {len(t)} articles, {len(g)} groups, fingerprint {fp} "
              f"{'OK (matches local)' if ok else '!! DOES NOT MATCH LOCAL SPLIT'}")
        if not ok and not ALLOW_SPLIT_MISMATCH:
            raise SystemExit("CNN split differs from the local one (different cnn_labels_with_content.csv?). "
                             "Fix the data or set ALLOW_SPLIT_MISMATCH = True.")
        t = t.copy(); t["headline"], t["body"] = t["title"].astype(str), t["content"].astype(str)
        row = {i: r for r, i in enumerate(t["id"])}
        splits[name] = (t, [np.array([row[i] for i in grp]) for grp in g])
    return splits

CNN_TEST = cnn_splits()
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

# ============================================================
# 4. MODEL (blocks identical to the v3 training script)
# ============================================================
class AttentionPooling(nn.Module):
    def __init__(self, d=768):
        super().__init__(); self.attn = nn.Linear(d, 1)
    def forward(self, reps, mask):
        w = self.attn(reps).squeeze(-1)
        w = w.masked_fill(mask == 0, torch.finfo(w.dtype).min)
        return torch.bmm(torch.softmax(w, -1).unsqueeze(1), reps).squeeze(1)

class SegmentAwareEncoder(nn.Module):
    def __init__(self, name=TOKENIZER_NAME, d=768):
        super().__init__()
        self.backbone = AutoModelForMaskedLM.from_pretrained(name)
        self.pool_h, self.pool_b, self.pool_j = AttentionPooling(d), AttentionPooling(d), AttentionPooling(d)
    def forward(self, input_ids, attention_mask, sep_positions):
        t = self.backbone(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True).hidden_states[-1]
        pos = torch.arange(t.shape[1], device=t.device).unsqueeze(0)
        sep = sep_positions.unsqueeze(1)
        hm = (pos >= 1) & (pos < sep) & (attention_mask == 1)
        bm = (pos > sep) & (attention_mask == 1)
        hm = torch.where(hm.any(1, keepdim=True), hm, attention_mask.bool())
        bm = torch.where(bm.any(1, keepdim=True), bm, attention_mask.bool())
        return {'e_h': self.pool_h(t, hm.long()), 'e_b': self.pool_b(t, bm.long()),
                'e_j': self.pool_j(t, attention_mask), 'tokens': t, 'attention_mask': attention_mask}

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
    def __init__(self, d=768, fd=128):
        super().__init__()
        self.q = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.attn = nn.MultiheadAttention(d, 8, batch_first=True)
        self.proj = nn.Sequential(nn.Linear(d, fd), nn.GELU(), nn.LayerNorm(fd))
        self.ordinal = OrdinalValueHead(fd)
    def forward(self, tokens, mask):
        out, _ = self.attn(self.q.expand(tokens.size(0), -1, -1), tokens, tokens, key_padding_mask=(mask == 0))
        vec = self.proj(out.squeeze(1))
        return {'vec': vec, **self.ordinal(vec)}

class SentenceValueHead(nn.Module):
    def __init__(self, d=768, fd=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d * 2, 256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, fd), nn.LayerNorm(fd))
        self.ordinal = OrdinalValueHead(fd)
    def forward(self, e_h, e_b):
        vec = self.net(torch.cat([e_h, e_b], -1))
        return {'vec': vec, **self.ordinal(vec)}

class FactorInteraction(nn.Module):
    def __init__(self, d=128, heads=4, p=0.1):
        super().__init__()
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, heads, d * 2, p, batch_first=True, activation='gelu'), 1)
    def forward(self, F_): return self.enc(F_)

class GatedScoring(nn.Module):
    def __init__(self, n_values=7, d=128):
        super().__init__()
        self.value_weights = nn.Parameter(torch.ones(n_values) / n_values)
        self.gate = nn.Sequential(nn.Linear(d, n_values), nn.Sigmoid())
        self.embed_proj = nn.Sequential(nn.Linear(d, d), nn.LayerNorm(d))
        self.score_head = nn.Linear(d, 1)
    def forward(self, F_prime):
        w = torch.softmax(self.value_weights, 0)
        z = F.normalize(self.embed_proj((F_prime * w.view(1, -1, 1)).sum(1)), p=2, dim=-1)
        return {'concept_score': self.score_head(z).squeeze(-1)}

class ListwideHead(nn.Module):
    def __init__(self, d=128):
        super().__init__()
        self.cls = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.tf = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 4, d * 2, 0.1, batch_first=True, activation='gelu'), 1)
        self.ordinal = OrdinalValueHead(d, 128)
    def forward(self, items):
        x = torch.cat([items, self.cls.expand(items.size(0), -1, -1)], 1)
        return self.ordinal(self.tf(x)[:, -1])['score']

class AblationModel(nn.Module):
    def __init__(self, variant):
        super().__init__()
        self.variant = variant
        self.encoder = SegmentAwareEncoder(TOKENIZER_NAME, HIDDEN_DIM)
        if variant == "plain_coral":
            self.heads = nn.ModuleList([OrdinalValueHead(HIDDEN_DIM) for _ in NEWS_VALUES])
        elif variant == "plain_mse":
            self.reg = nn.Sequential(nn.Dropout(0.1), nn.Linear(HIDDEN_DIM, 1))
        else:
            token = TOKEN_VALUES if variant != "no_token" else set()
            self.value_heads = nn.ModuleDict({v: TokenValueHead(HIDDEN_DIM, FACTOR_DIM) if v in token
                                              else SentenceValueHead(HIDDEN_DIM, FACTOR_DIM) for v in NEWS_VALUES})
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
        out = {"concept_score": self.scoring(F_prime)["concept_score"],
               "value_scores": torch.stack(scores, 1), "value_logits": torch.stack(logits, 1)}
        if group_size:   # lists are contiguous equal-size blocks of the batch
            out["listwide_score"] = self.listwide(F_prime.mean(1).view(-1, group_size, F_prime.size(-1)))
        return out

def build_model(variant):
    try:
        import peft.import_utils, peft.tuners.lora.torchao as _t
        peft.import_utils.is_torchao_available = lambda: False
        _t.is_torchao_available = lambda: False
    except Exception: pass
    from peft import LoraConfig, get_peft_model
    model = AblationModel(variant)
    bb = model.encoder.backbone
    bb.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    preferred = ['Wqkv', 'Wo', 'Wi', 'query', 'key', 'value', 'dense', 'q_proj', 'k_proj', 'v_proj',
                 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
    present = {n.split('.')[-1] for n, m in bb.named_modules() if isinstance(m, nn.Linear) and 'decoder' not in n}
    targets = [p for p in preferred if p in present] or sorted(present)
    model.encoder.backbone = get_peft_model(bb, LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
                                                           bias='none', target_modules=targets))
    for n, p in model.named_parameters():
        if not n.startswith("encoder.backbone"): p.requires_grad = True
    return model.to(DEVICE)

def trainable_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            if "lora_" in k or not k.startswith("encoder.backbone")}

# ============================================================
# 5. LOSSES (v3 losses, vectorised over equal-size lists)
# ============================================================
def coral_loss(cum_logits, targets):
    t = targets.clamp(0, NUM_BINS - 1).long()
    bins = torch.arange(1, NUM_BINS, device=t.device)
    return F.binary_cross_entropy_with_logits(cum_logits.float(), (t.unsqueeze(1) >= bins).float())

def listnet_top1(scores, targets):
    return -(torch.softmax(targets.float(), -1) * torch.log_softmax(scores.float(), -1)).sum(-1).mean()

def approx_ndcg(scores, targets, alpha=10.0):
    s, y = scores.float(), targets.float()
    rank = 1.0 + torch.sigmoid(alpha * (s.unsqueeze(1) - s.unsqueeze(2))).sum(1)
    gains = 2.0 ** (y - y.min(-1, keepdim=True).values) - 1.0
    dcg = (gains / torch.log2(rank + 1.0)).sum(-1)
    idcg = (gains.sort(-1, descending=True).values / torch.log2(torch.arange(2, y.size(1) + 2, device=y.device).float())).sum(-1)
    return -(dcg / idcg.clamp_min(1e-8)).mean()

def compute_loss(variant, out, gold, values):
    if variant == "plain_mse":
        return F.mse_loss(out["scalar"].float(), gold / 35.0)
    loss = W_POINT * sum(coral_loss(out["value_logits"][:, i], values[:, i]) for i in range(len(NEWS_VALUES))) / len(NEWS_VALUES)
    if variant in ("plain_coral", "no_rank"):
        return loss
    s, y = out["concept_score"].view(-1, GROUP_SIZE), gold.view(-1, GROUP_SIZE)
    loss = loss + W_LIST * listnet_top1(s, y) + W_NDCG * approx_ndcg(s, y)
    return loss + W_WIDE * F.mse_loss(out["listwide_score"].float(), y.max(-1).values / 35.0 * (NUM_BINS - 1))

# ============================================================
# 6. ENCODE / PREDICT / METRICS (metrics identical to the local evaluation)
# ============================================================
def encode(heads, bodies):
    sep = tokenizer.sep_token or "[SEP]"
    enc = tokenizer([f"{h} {sep} {b}" for h, b in zip(heads, bodies)], padding=True, truncation=True,
                    max_length=MAX_SEQ_LEN, return_tensors="pt")
    seps = []
    for ids in enc["input_ids"]:
        locs = (ids == tokenizer.sep_token_id).nonzero(as_tuple=True)[0]
        seps.append(locs[0].item() if len(locs) else 1)
    return enc["input_ids"].to(DEVICE), enc["attention_mask"].to(DEVICE), torch.tensor(seps, dtype=torch.long, device=DEVICE)

@torch.no_grad()
def predict(model, frame, bs=64):
    model.eval()
    res = {"value_sum": [], "concept": [], "scalar": [], "values": []}
    for i in range(0, len(frame), bs):
        f = frame.iloc[i:i + bs]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o = model(*encode(f["headline"].tolist(), f["body"].tolist()))
        if "value_scores" in o:
            vs = o["value_scores"].float()
            res["value_sum"].append(vs.sum(1).cpu().numpy()); res["values"].append(vs.cpu().numpy())
        if "concept_score" in o: res["concept"].append(o["concept_score"].float().cpu().numpy())
        if "scalar" in o: res["scalar"].append(o["scalar"].float().cpu().numpy())
    model.train()
    return {k: np.concatenate(v) for k, v in res.items() if v}

primary = lambda pred: pred["scalar"] if "scalar" in pred else pred["value_sum"]

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

def evaluate_scores(scores, test, groups):
    gold = test["gold"].to_numpy(dtype=float)
    pg = [{"ndcg5": ndcg_at_k(scores[r], gold[r], 5), "ndcg10": ndcg_at_k(scores[r], gold[r], 10),
           "tau": kendall_tau_metric(scores[r], gold[r]), "pa": pairwise_accuracy(scores[r], gold[r])} for r in groups]
    col = lambda k: np.array([m[k] for m in pg])
    return {"ndcg@5": float(col("ndcg5").mean()), "ndcg@10": float(col("ndcg10").mean()),
            "kendall_tau": float(col("tau").mean()), "pairwise_acc": float(col("pa").mean()),
            "spearman_global": float(np.nan_to_num(spearmanr(scores, gold).correlation)),
            "spearman_by_news_value": {v: float(np.nan_to_num(spearmanr(scores, test[v]).correlation)) for v in NEWS_VALUES},
            "per_group_ndcg10": col("ndcg10").tolist()}

# ============================================================
# 7. ONE RUN
# ============================================================
def run(split, variant, seed):
    tag = f"{variant}_seed{seed}"
    out_dir = os.path.join(RES_DIR, split, tag)
    if os.path.exists(os.path.join(out_dir, "metrics.json")):
        print(f"[skip] {split}/{tag} already done"); return
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n{'=' * 70}\n{split}/{tag}\n{'=' * 70}")

    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    train, val = bbc_pool(split)
    lists = make_lists(train, seed)
    val = val.assign(headline=val["title"].astype(str), body=val["content"].astype(str))
    print(f"BBC train pool={len(train)} -> {len(lists)} lists x {GROUP_SIZE} | val={len(val)}")

    model = build_model(variant)
    lora  = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("encoder.backbone")]
    heads = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("encoder.backbone")]
    opt = torch.optim.AdamW([{"params": lora, "lr": LR_BACKBONE}, {"params": heads, "lr": LR_HEADS}], weight_decay=WEIGHT_DECAY)
    n_micro = len(lists) // MICRO
    total = n_micro // ACCUM
    warm = int(WARMUP_RATIO * total)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: s / max(warm, 1) if s < warm else
                                              max(0.1, 0.5 * (1 + math.cos(math.pi * (s - warm) / max(total - warm, 1)))))
    gs = GROUP_SIZE if variant in ("full", "no_interaction", "no_token") else None
    print(f"optimizer steps={total}, warmup={warm}")

    log, best_rho, best_state, step, t0 = [], -np.inf, None, 0, time.time()
    model.train(); opt.zero_grad(set_to_none=True)
    for mi in range(n_micro):
        b = train.loc[np.concatenate(lists[mi * MICRO:(mi + 1) * MICRO])]
        gold = torch.tensor(b["gold"].values, dtype=torch.float, device=DEVICE)
        values = torch.tensor(b[NEWS_VALUES].values, dtype=torch.float, device=DEVICE)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(*encode(b["title"].astype(str).tolist(), b["content"].astype(str).tolist()), group_size=gs)
            loss = compute_loss(variant, out, gold, values) / ACCUM
        loss.backward()
        if (mi + 1) % ACCUM: continue
        torch.nn.utils.clip_grad_norm_(lora + heads, MAX_GRAD_NORM)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1
        log.append({"step": step, "loss": float(loss.item() * ACCUM)})
        if step % 25 == 0:
            rate = (mi + 1) * MICRO * GROUP_SIZE / (time.time() - t0)
            print(f"  step {step:>4}/{total} loss={np.mean([l['loss'] for l in log[-25:]]):.4f} "
                  f"{rate:.0f} art/s ETA {(n_micro - mi - 1) * MICRO * GROUP_SIZE / rate / 60:.0f} min "
                  f"mem={torch.cuda.max_memory_allocated(DEVICE) / 2**30:.1f}GB", flush=True)
        if step % EVAL_EVERY == 0 or step == total:
            rho = float(spearmanr(primary(predict(model, val)), val["gold"]).correlation)
            log[-1]["val_spearman"] = rho
            if rho > best_rho: best_rho, best_state = rho, trainable_state(model)
            print(f"  >>> step {step} BBC-val Spearman={rho:.4f} (best {best_rho:.4f})", flush=True)
    train_min = (time.time() - t0) / 60

    if SAVE_CKPTS:
        torch.save({"model_state": best_state, "variant": variant, "seed": seed, "split": split,
                    "best_val_spearman": best_rho}, os.path.join(CKPT_DIR, f"{split}_{tag}.pt"))
    with open(os.path.join(out_dir, "train_log.json"), "w") as f:
        json.dump({"log": log, "best_val_spearman": best_rho, "train_minutes": train_min,
                   "config": {"train_groups": TRAIN_GROUPS, "group_size": GROUP_SIZE, "micro": MICRO, "accum": ACCUM,
                              "lists_per_step": LISTS_PER_STEP, "gpu": torch.cuda.get_device_name(DEVICE)}}, f)

    # ---- CNN zero-shot test with BBC-val-selected weights ----
    model.load_state_dict(best_state, strict=False)
    test, groups = CNN_TEST[split]
    pred = predict(model, test)
    metrics = {"variant": variant, "seed": seed, "split": split, "best_bbc_val_spearman": best_rho,
               "train_minutes": train_min, "n_test": len(test), "n_groups": len(groups), "heads": {},
               "primary_head": "scalar" if "scalar" in pred else "value_sum"}
    scores = test[["id", "section", "gold"] + NEWS_VALUES].copy()
    for head in ["value_sum", "scalar", "concept"]:
        if head not in pred: continue
        r = evaluate_scores(pred[head], test, groups)
        metrics["heads"][head] = r; scores[head] = pred[head]
        print(f"  CNN [{head:<9}] NDCG@10={r['ndcg@10']:.4f} tau={r['kendall_tau']:+.4f} "
              f"PA={r['pairwise_acc']:.4f} rho={r['spearman_global']:+.4f}")
    if "values" in pred:
        metrics["value_fit"] = {v: float(spearmanr(pred["values"][:, j], test[v]).correlation) for j, v in enumerate(NEWS_VALUES)}
        for j, v in enumerate(NEWS_VALUES): scores[f"pred_{v}"] = pred["values"][:, j]
    scores.to_csv(os.path.join(out_dir, "test_scores.csv"), index=False)
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:   # written last = run complete
        json.dump(metrics, f, indent=2)
    print(f"Done {split}/{tag}: train {train_min:.1f} min")

    del model, opt, best_state; gc.collect(); torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(DEVICE)

# ============================================================
# 8. QUEUE + SUMMARY + ZIP
# ============================================================
JOBS = [(s, v, seed) for seed in SEEDS for (s, v) in RUNS if s == "random"] + \
       [(s, v, seed) for seed in SEEDS for (s, v) in RUNS if s == "temporal"]
print(f"\n{len(JOBS)} runs queued")
t_all = time.time()
for i, (split, variant, seed) in enumerate(JOBS, 1):
    print(f"\n### run {i}/{len(JOBS)} — elapsed {(time.time() - t_all) / 3600:.1f} h")
    try:
        run(split, variant, seed)
    except torch.cuda.OutOfMemoryError:
        print(f"!! OOM in {split}/{variant}_seed{seed} (MICRO={MICRO}): set MICRO_OVERRIDE = {max(1, MICRO // 2)} and re-run")
        gc.collect(); torch.cuda.empty_cache()

rows = []
for split, variant, seed in JOBS:
    p = os.path.join(RES_DIR, split, f"{variant}_seed{seed}", "metrics.json")
    if os.path.exists(p):
        m = json.load(open(p)); r = m["heads"][m["primary_head"]]
        rows.append({"split": split, "variant": variant, "seed": seed, "bbc_val_rho": m["best_bbc_val_spearman"],
                     **{k: r[k] for k in ["ndcg@10", "kendall_tau", "pairwise_acc", "spearman_global"]},
                     "train_min": m["train_minutes"]})
summary = pd.DataFrame(rows)
summary.to_csv(os.path.join(RES_DIR, "cloud_summary.csv"), index=False)
pd.set_option("display.width", 200)
print("\n" + summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
if len(summary):
    print("\nMean over seeds:")
    print(summary.groupby(["split", "variant"])[["ndcg@10", "kendall_tau", "spearman_global"]].agg(["mean", "std"])
          .to_string(float_format=lambda x: f"{x:.4f}"))

zip_path = os.path.join(WORK_DIR, "ablation_results.zip")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    for root, _, files in os.walk(RES_DIR):
        for fn in files:
            full = os.path.join(root, fn)
            z.write(full, os.path.relpath(full, WORK_DIR))   # -> eval_results/ablations/...
print(f"\nALL DONE: {len(summary)}/{len(JOBS)} runs in {(time.time() - t_all) / 3600:.1f} h")
print(f"Copy back: {zip_path}  (unzip in the repo root, then: python ablation_report.py)")
print(f"Checkpoints (optional): {CKPT_DIR}")

#!/usr/bin/env python3
"""
Retrain RankingTower v2 on the lowest-loss ResonantBERT v3 checkpoint.

Evaluates every BERT checkpoint in checkpoints_v3/ on the validation set
(combined v3 loss + ranking metrics), selects the lowest-loss checkpoint,
caches frozen embeddings from it, and retrains the RankingTower v2 from
scratch. Saves the new ranker + comparison report.

Usage:
  python retrain_ranker_lowest_loss.py
  python retrain_ranker_lowest_loss.py --metric val_loss   # default
  python retrain_ranker_lowest_loss.py --metric spearman   # alternative
"""

import argparse, os, re, gc, glob, json, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForMaskedLM, AutoTokenizer
from scipy.stats import kendalltau, spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================
# CONFIG (must match train_resonantbert_v3.py)
# ============================================================
SEED           = 42
REPO_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # run from this repo; no re-clone
CKPT_DIR       = os.environ.get("CKPT_DIR", os.path.join(REPO_DIR, "model", "checkpoints_v3"))

TOKENIZER_NAME = "answerdotai/ModernBERT-base"
MAX_SEQ_LEN    = 512
HIDDEN_DIM     = 768
FACTOR_DIM     = 128
NUM_BINS       = 6
LORA_R         = 64
LORA_ALPHA     = 128    # from ResonantBERTv3_KAGGLE_Training.ipynb
LORA_DROPOUT   = 0.05

LR_RANKER      = 1e-3
RANKER_STEPS   = 1000
GROUPS_PER_BATCH = 4
MAX_GROUP_SIZE   = 8
VAL_FRAC         = 0.10
MAX_GRAD_NORM    = 1.0
USE_BF16         = True

NEWS_VALUES  = ["impact", "conflict", "novelty", "proximity",
                "prominence", "timeliness", "human_interest"]
TOKEN_VALUES = {"conflict", "prominence", "human_interest", "novelty"}
SENT_VALUES  = {"impact", "proximity", "timeliness"}

W_POINT, W_LIST, W_NDCG, W_WIDE = 1.0, 1.0, 0.5, 0.25

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_amp_on = USE_BF16 and DEVICE.type == 'cuda'
_amp_dtype = torch.bfloat16 if USE_BF16 else torch.float16
print(f"Device: {DEVICE}")

# ============================================================
# DATA (same pipeline as training script)
# ============================================================
CSV_PATH = os.path.join(REPO_DIR, "bbc_articles", "track_a_all.csv")
assert os.path.exists(CSV_PATH), f"{CSV_PATH} not found"
assert os.path.getsize(CSV_PATH) > 1e6, f"{CSV_PATH} is an LFS pointer; run `git lfs pull`"

df = pd.read_csv(CSV_PATH, encoding="utf-8", encoding_errors="replace")
col_map = {nv: nv for nv in NEWS_VALUES if nv in df.columns}
assert len(col_map) == len(NEWS_VALUES), f"Missing columns: {len(col_map)}/{len(NEWS_VALUES)}"

text_cols = {}
for c in ['title', 'headline', 'heading']:
    if c in df.columns: text_cols['headline'] = c; break
for c in ['content', 'body', 'text', 'article']:
    if c in df.columns: text_cols['body'] = c; break
section_col = next((c for c in ['section','category','topic','section_name']
                    if c in df.columns), None)

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

def compute_gold_score(row):
    return sum(row[col_map[nv]] for nv in NEWS_VALUES)

df['gold_score'] = df.apply(compute_gold_score, axis=1)
df[text_cols['headline']] = df[text_cols['headline']].fillna('').astype(str)
df[text_cols['body']]     = df[text_cols['body']].fillna('').astype(str)
df = df[df[text_cols['body']].str.strip().str.len() > 20].reset_index(drop=True)

groups = []
if section_col and df[section_col].nunique() > 5:
    for sec, sec_df in df.groupby(section_col):
        sec_df = sec_df.sort_values('gold_score', ascending=False).reset_index(drop=True)
        for start in range(0, len(sec_df), MAX_GROUP_SIZE):
            chunk = sec_df.iloc[start:start+MAX_GROUP_SIZE]
            if len(chunk) < 2: continue
            groups.append([{
                'headline': r[text_cols['headline']], 'body': r[text_cols['body']],
                'rank': i, 'gold_score': r['gold_score'],
                'value_targets': [float(r[col_map[nv]]) for nv in NEWS_VALUES],
            } for i, (_, r) in enumerate(chunk.iterrows())])
else:
    idx = list(range(len(df)))
    random.Random(SEED).shuffle(idx)
    for start in range(0, len(idx), MAX_GROUP_SIZE):
        ci = idx[start:start+MAX_GROUP_SIZE]
        if len(ci) < 2: continue
        chunk = df.iloc[ci].sort_values('gold_score', ascending=False).reset_index(drop=True)
        groups.append([{
            'headline': r[text_cols['headline']], 'body': r[text_cols['body']],
            'rank': i, 'gold_score': r['gold_score'],
            'value_targets': [float(r[col_map[nv]]) for nv in NEWS_VALUES],
        } for i, (_, r) in enumerate(chunk.iterrows())])

idx = list(range(len(groups)))
random.Random(SEED).shuffle(idx)
n_val = max(1, int(round(VAL_FRAC * len(groups))))
val_groups  = [groups[i] for i in sorted(idx[:n_val])]
train_groups = [groups[i] for i in sorted(idx[n_val:])]
print(f"Train: {len(train_groups)} groups | Val: {len(val_groups)} groups")

class NewsValueDataset(Dataset):
    def __init__(self, groups): self.groups = groups
    def __len__(self): return len(self.groups)
    def __getitem__(self, idx):
        g = self.groups[idx]
        return {
            'headlines':     [a['headline'] for a in g],
            'bodies':        [a['body'] for a in g],
            'ranks':         [a['rank'] for a in g],
            'gold_scores':   [a['gold_score'] for a in g],
            'value_targets': [a['value_targets'] for a in g],
        }

def collate_v3(batch):
    sep_token = tokenizer.sep_token or "[SEP]"
    texts, ranks, gids, gold_scores, value_targets = [], [], [], [], []
    for gi, item in enumerate(batch):
        for h, b, r, gs, vt in zip(item['headlines'], item['bodies'], item['ranks'],
                                   item['gold_scores'], item['value_targets']):
            texts.append(f"{h} {sep_token} {b}")
            ranks.append(r); gids.append(gi); gold_scores.append(gs)
            value_targets.append(vt)
    encoded = tokenizer(texts, padding=True, truncation=True,
                        max_length=MAX_SEQ_LEN, return_tensors="pt")
    sep_id = tokenizer.sep_token_id
    sep_positions = []
    for i in range(encoded['input_ids'].shape[0]):
        locs = (encoded['input_ids'][i] == sep_id).nonzero(as_tuple=True)[0]
        sep_positions.append(locs[0].item() if len(locs) >= 1 else 1)
    return {
        'input_ids': encoded['input_ids'], 'attention_mask': encoded['attention_mask'],
        'sep_positions': torch.tensor(sep_positions, dtype=torch.long),
        'ranks': torch.tensor(ranks, dtype=torch.long),
        'group_ids': torch.tensor(gids, dtype=torch.long),
        'gold_score': torch.tensor(gold_scores, dtype=torch.float),
        'value_targets': torch.tensor(value_targets, dtype=torch.float),
    }

train_loader = DataLoader(NewsValueDataset(train_groups), batch_size=GROUPS_PER_BATCH,
                          shuffle=True, collate_fn=collate_v3)
val_loader   = DataLoader(NewsValueDataset(val_groups),   batch_size=GROUPS_PER_BATCH,
                          shuffle=False, collate_fn=collate_v3)

# ============================================================
# MODEL CLASSES (identical to training script)
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
        layer = nn.TransformerEncoderLayer(d, heads, d * 2, p,
                                           batch_first=True, activation='gelu')
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
        return {'viral_embedding': z,
                'concept_score': self.score_head(z).squeeze(-1),
                'factor_weights': w}

class ListwideHead(nn.Module):
    def __init__(self, d=FACTOR_DIM):
        super().__init__()
        self.cls = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, 4, d * 2, 0.1,
                                           batch_first=True, activation='gelu')
        self.tf = nn.TransformerEncoder(layer, 1)
        self.ordinal = OrdinalValueHead(d, 128)
    def forward(self, items):
        x = torch.cat([items, self.cls.expand(items.size(0), -1, -1)], 1)
        return self.ordinal(self.tf(x)[:, -1])['score']

def pad_groups(embeds, group_ids):
    uniq, inv = torch.unique(group_ids, return_inverse=True)
    counts = torch.bincount(inv)
    Lmax = int(counts.max())
    out = embeds.new_zeros(len(uniq), Lmax, embeds.size(1))
    slot = torch.zeros(len(uniq), dtype=torch.long, device=embeds.device)
    for i in range(embeds.size(0)):
        g = inv[i]; out[g, slot[g]] = embeds[i]; slot[g] += 1
    return out

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
    def forward(self, input_ids, attention_mask, sep_positions, group_ids=None):
        enc = self.encoder(input_ids, attention_mask, sep_positions)
        vecs, scores, logits = [], [], []
        for v in NEWS_VALUES:
            h = self.value_heads[v]
            r = h(enc['tokens'], enc['attention_mask']) if isinstance(h, TokenValueHead) \
                else h(enc['e_h'], enc['e_b'])
            vecs.append(r['vec']); scores.append(r['score']); logits.append(r['cum_logits'])
        F_prime = self.interaction(torch.stack(vecs, 1))
        s = self.scoring(F_prime)
        out = {'viral_embedding': s['viral_embedding'],
               'concept_score': s['concept_score'],
               'factor_weights': s['factor_weights'],
               'value_scores': torch.stack(scores, 1),
               'value_logits': torch.stack(logits, 1),
               'risk_score': self.risk_head(enc['e_j']).squeeze(-1)}
        if group_ids is not None:
            out['listwide_score'] = self.listwide(pad_groups(F_prime.mean(1), group_ids))
        return out

class RankingTowerV2(nn.Module):
    def __init__(self, in_dim=FACTOR_DIM, p_drop=0.1):
        super().__init__()
        self.tower = nn.Sequential(nn.Linear(in_dim, 64), nn.GELU(),
                                   nn.Dropout(p_drop), nn.Linear(64, 1))
        self.temperature = nn.Parameter(torch.tensor(1.0))
    def forward(self, z):
        raw = self.tower(z).squeeze(-1)
        cal = torch.sigmoid(raw / self.temperature)
        return {'raw_score': raw, 'calibrated_score': cal, 'final_score': cal}

# ============================================================
# LOSSES + METRICS (identical to training script)
# ============================================================
def coral_loss(cum_logits, targets):
    t = targets.clamp(0, NUM_BINS - 1).long()
    bins = torch.arange(1, NUM_BINS, device=t.device)
    y = (t.unsqueeze(1) >= bins).float()
    return F.binary_cross_entropy_with_logits(cum_logits, y)

def listnet_top1_loss(scores, targets):
    p = torch.log_softmax(scores.float(), -1)
    q = torch.softmax(targets.float(), -1)
    return -(q * p).sum()

def approx_ndcg_loss(scores, targets, alpha=10.0):
    scores, targets = scores.float(), targets.float()
    s_diff = scores.unsqueeze(0) - scores.unsqueeze(1)
    approx_rank = 1.0 + torch.sigmoid(alpha * s_diff).sum(0)
    gains = 2.0 ** targets - 1.0
    dcg = (gains / torch.log2(approx_rank + 1.0)).sum()
    ideal = torch.arange(2, len(targets) + 2, device=targets.device).float()
    idcg = (gains.sort(descending=True).values / torch.log2(ideal)).sum()
    return -dcg / idcg.clamp_min(1e-8)

def v3_combined_loss(out, batch):
    loss = torch.tensor(0.0, device=batch['input_ids'].device)
    for i in range(len(NEWS_VALUES)):
        loss = loss + coral_loss(out['value_logits'][:, i], batch['value_targets'][:, i])
    loss = W_POINT * loss / len(NEWS_VALUES)
    gids = batch['group_ids']
    uniq = torch.unique(gids)
    for g in uniq:
        m = gids == g
        if m.sum() < 2: continue
        s = out['concept_score'][m]
        y = batch['gold_score'][m].float()
        loss = loss + (W_LIST * listnet_top1_loss(s, y)
                       + W_NDCG * approx_ndcg_loss(s, y)) / len(uniq)
    if 'listwide_score' in out:
        gold_max = torch.stack([batch['gold_score'][gids == g].max().float()
                                for g in uniq])
        loss = loss + W_WIDE * F.mse_loss(out['listwide_score'], gold_max)
    return loss

def dcg_at_k(gains, k):
    gains = gains[:k]
    return float((gains / np.log2(np.arange(2, gains.size + 2))).sum())

def ndcg_at_k(pred_scores, gold_ranks, k=10):
    n = len(gold_ranks)
    rel = (n - 1) - gold_ranks
    dcg  = dcg_at_k(rel[np.argsort(-pred_scores)], k)
    idcg = dcg_at_k(rel[np.argsort(-rel)], k)
    return dcg / idcg if idcg > 0 else 0.0

def kendall_tau_metric(pred_scores, gold_ranks):
    if len(pred_scores) < 2: return 0.0
    tau, _ = kendalltau(pred_scores, -gold_ranks)
    return float(tau) if not np.isnan(tau) else 0.0

def spearman_metric(pred_scores, gold_ranks):
    if len(pred_scores) < 2: return 0.0
    rho, _ = spearmanr(pred_scores, -gold_ranks)
    return float(rho) if not np.isnan(rho) else 0.0

# ============================================================
# CHECKPOINT SCANNING
# ============================================================
def build_model():
    """ResonantBERT v3 with the backbone wrapped in the same LoRA config as training,
    so the checkpoint's lora_A/lora_B tensors have somewhere to load into."""
    import peft.import_utils
    try:
        import peft.tuners.lora.torchao as _lora_torchao
        peft.import_utils.is_torchao_available = lambda: False
        _lora_torchao.is_torchao_available = lambda: False
    except Exception: pass
    from peft import LoraConfig, get_peft_model

    model = ResonantBERTv3(backbone=TOKENIZER_NAME, d=HIDDEN_DIM, fd=FACTOR_DIM)
    preferred = ['Wqkv', 'Wo', 'Wi', 'query', 'key', 'value', 'dense',
                 'q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
    present = {n.split('.')[-1] for n, m in model.encoder.backbone.named_modules()
               if isinstance(m, nn.Linear) and 'decoder' not in n}
    targets = [p for p in preferred if p in present] or sorted(present)
    model.encoder.backbone = get_peft_model(model.encoder.backbone, LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT, bias='none',
        target_modules=targets))
    return model.to(DEVICE)

def load_ckpt_direct(model, path):
    """Loads LoRA adapters + all heads; frozen ModernBERT weights stay from the hub.
    Fails loudly if any checkpoint tensor has no home or any trained tensor is missing."""
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    state = ckpt['model_state']
    missing, unexpected = model.load_state_dict(state, strict=False)
    bad_missing = [k for k in missing if 'lora_' in k or not k.startswith('encoder.backbone')]
    assert not unexpected, f"{os.path.basename(path)}: unexpected keys {unexpected[:5]}"
    assert not bad_missing, f"{os.path.basename(path)}: trained weights missing {bad_missing[:5]}"
    loaded, skipped = len(state), len(unexpected)
    del ckpt, state
    gc.collect(); torch.cuda.empty_cache()
    return loaded, skipped

@torch.no_grad()
def eval_checkpoint_loss_and_metrics(model, loader):
    """Returns (mean_val_loss, metrics dict) over the full val loader."""
    model.eval()
    losses, ndcgs, taus, rhos = [], [], [], []
    for batch in loader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        with torch.amp.autocast('cuda', enabled=_amp_on, dtype=_amp_dtype):
            out = model(batch['input_ids'], batch['attention_mask'],
                        batch['sep_positions'], group_ids=batch['group_ids'])
            loss = v3_combined_loss(out, batch)
        losses.append(loss.item())
        s = out['concept_score'].float().cpu().numpy()
        r = batch['ranks'].cpu().numpy()
        gid = batch['group_ids'].cpu().numpy()
        for g in np.unique(gid):
            m = gid == g
            if m.sum() < 2: continue
            ndcgs.append(ndcg_at_k(s[m], r[m], k=5))
            taus.append(kendall_tau_metric(s[m], r[m]))
            rhos.append(spearman_metric(s[m], r[m]))
    return float(np.mean(losses)), {
        'ndcg@5': float(np.mean(ndcgs)), 'kendall_tau': float(np.mean(taus)),
        'spearman': float(np.mean(rhos)), 'n_groups': len(ndcgs)}

def _step_num(p):
    m = re.search(r'step(\d+)', os.path.basename(p))
    return int(m.group(1)) if m else 0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--metric', default='val_loss',
                    choices=['val_loss', 'spearman', 'ndcg@5', 'kendall_tau'],
                    help='Selection metric (lower-is-better for val_loss, higher otherwise)')
    ap.add_argument('--ranker-steps', type=int, default=RANKER_STEPS)
    args = ap.parse_args()

    ckpt_files = sorted(glob.glob(os.path.join(CKPT_DIR, 'BERTv3-step*.pt')),
                        key=_step_num)
    assert ckpt_files, f"No BERT checkpoints found in {CKPT_DIR}"
    print(f"Found {len(ckpt_files)} checkpoints to evaluate:\n  "
          + "\n  ".join(os.path.basename(p) for p in ckpt_files))

    # ── Pass 1: evaluate every checkpoint on full val set ──
    results = {}
    model = build_model()
    for path in ckpt_files:
        name = os.path.basename(path)
        loaded, skipped = load_ckpt_direct(model, path)
        vloss, m = eval_checkpoint_loss_and_metrics(model, val_loader)
        results[name] = {'val_loss': vloss, **m}
        print(f"  {name:32s} loss={vloss:9.4f}  ndcg@5={m['ndcg@5']:.4f}  "
              f"tau={m['kendall_tau']:.4f}  rho={m['spearman']:.4f}  "
              f"(loaded {loaded}, skipped {skipped})")
        del results[name]['n_groups']

    # ── Select best ──
    reverse = (args.metric != 'val_loss')
    best_name = max(results, key=lambda k: results[k][args.metric]) if reverse \
           else min(results, key=lambda k: results[k][args.metric])
    best_path = os.path.join(CKPT_DIR, best_name)
    print(f"\n>>> Selected [{args.metric}] best checkpoint: {best_name}")
    print(f"    val_loss={results[best_name]['val_loss']:.4f}  "
          f"spearman={results[best_name]['spearman']:.4f}")

    # ── Plot comparison ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    names = sorted(results, key=_step_num)
    x = [_step_num(n) for n in names]
    axes[0].plot(x, [results[n]['val_loss'] for n in names], 'o-', color='#e74c3c')
    axes[0].set_xlabel('Step'); axes[0].set_ylabel('Combined val loss')
    axes[0].set_title('Checkpoint val loss', fontweight='bold'); axes[0].grid(alpha=0.3)
    axes[1].plot(x, [results[n]['spearman'] for n in names], 's-', label='Spearman ρ', color='#3498db')
    axes[1].plot(x, [results[n]['kendall_tau'] for n in names], '^-', label='Kendall τ', color='#e67e22')
    axes[1].plot(x, [results[n]['ndcg@5'] for n in names], 'o-', label='NDCG@5', color='#2ecc71')
    axes[1].axvline(_step_num(best_name), ls='--', color='gray', alpha=0.6,
                    label=f'selected (step {_step_num(best_name)})')
    axes[1].set_xlabel('Step'); axes[1].set_ylabel('Score')
    axes[1].set_title('Checkpoint ranking metrics', fontweight='bold')
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    cmp_path = os.path.join(CKPT_DIR, 'checkpoint_comparison.png')
    plt.savefig(cmp_path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"Saved comparison plot: {cmp_path}")

    # ── Pass 2: load best checkpoint, cache embeddings, retrain ranker ──
    load_ckpt_direct(model, best_path)
    model.eval()
    for p in model.parameters(): p.requires_grad = False

    @torch.no_grad()
    def cache_embeddings(encoder, loader):
        emb, ranks, gids, gold_scores = [], [], [], []
        off = 0
        for bi, batch in enumerate(loader):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            with torch.amp.autocast('cuda', enabled=_amp_on, dtype=_amp_dtype):
                out = encoder(batch['input_ids'], batch['attention_mask'],
                              batch['sep_positions'])
            emb.append(out['viral_embedding'].float().cpu())
            ranks.append(batch['ranks'].cpu())
            gids.append(batch['group_ids'].cpu() + off)
            gold_scores.append(batch['gold_score'].cpu())
            off += batch['group_ids'].max().item() + 1
        print(f"  cached {sum(e.shape[0] for e in emb)} embeddings")
        return {'embeddings': torch.cat(emb), 'ranks': torch.cat(ranks),
                'group_ids': torch.cat(gids), 'gold_scores': torch.cat(gold_scores)}

    print(f"\nCaching embeddings from {best_name} ...")
    cache = cache_embeddings(model, train_loader)
    emb  = cache['embeddings'].to(DEVICE)
    rks  = cache['ranks'].to(DEVICE)
    gids_r = cache['group_ids'].to(DEVICE)
    gold = cache['gold_scores'].to(DEVICE)

    ranker = RankingTowerV2(in_dim=FACTOR_DIM).to(DEVICE)
    ropt = torch.optim.AdamW(ranker.parameters(), lr=LR_RANKER, weight_decay=0.01)

    def group_batches(gids, per_batch):
        u = gids.unique()[torch.randperm(gids.unique().numel(), device=gids.device)]
        for i in range(0, u.numel(), per_batch):
            yield torch.isin(gids, u[i:i+per_batch])

    ranker.train()
    r_losses = []
    _rit = iter(group_batches(gids_r, GROUPS_PER_BATCH))
    step = 0
    while step < args.ranker_steps:
        try: m = next(_rit)
        except StopIteration:
            _rit = iter(group_batches(gids_r, GROUPS_PER_BATCH)); m = next(_rit)
        if m.sum() < 2: continue
        ropt.zero_grad(set_to_none=True)
        raw = ranker.tower(emb[m]).squeeze(-1)
        g_ids_m, g_scores = gids_r[m], gold[m]
        loss = torch.tensor(0.0, device=DEVICE)
        for g in g_ids_m.unique():
            gm = g_ids_m == g
            if gm.sum() < 2: continue
            loss = loss + listnet_top1_loss(raw[gm], g_scores[gm])
        loss = loss / max(g_ids_m.unique().numel(), 1)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ranker.parameters(), MAX_GRAD_NORM)
        ropt.step()
        step += 1
        r_losses.append(loss.item())
        if step % 100 == 0:
            print(f"  [ranker step {step:>5}] loss={np.mean(r_losses[-100:]):.4f}")

    tag = f"RANKERv2-{best_name.replace('BERTv3-','').replace('.pt','')}"
    ranker_path = os.path.join(CKPT_DIR, f'{tag}.pt')
    torch.save({'model_state': ranker.state_dict(), 'ranker_step': step,
                'base_checkpoint': best_name,
                'base_val_loss': results[best_name]['val_loss'],
                'base_spearman': results[best_name]['spearman']},
               ranker_path)
    print(f"\nRanker saved: {ranker_path}")
    print(f"Temperature T = {ranker.temperature.item():.4f}")

    # ── Final eval: best encoder + new ranker on val ──
    ranker.eval()
    @torch.no_grad()
    def full_eval(encoder, tower, loader):
        ndcg5s, ndcg10s, taus, rhos = [], [], [], []
        for batch in loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            with torch.amp.autocast('cuda', enabled=_amp_on, dtype=_amp_dtype):
                enc_out = encoder(batch['input_ids'], batch['attention_mask'],
                                  batch['sep_positions'])
            s = tower(enc_out['viral_embedding'])['final_score'].float().cpu().numpy()
            r = batch['ranks'].cpu().numpy()
            gid = batch['group_ids'].cpu().numpy()
            for g in np.unique(gid):
                m = gid == g
                if m.sum() < 2: continue
                ndcg5s.append(ndcg_at_k(s[m], r[m], k=5))
                ndcg10s.append(ndcg_at_k(s[m], r[m], k=10))
                taus.append(kendall_tau_metric(s[m], r[m]))
                rhos.append(spearman_metric(s[m], r[m]))
        return {'ndcg@5': float(np.mean(ndcg5s)), 'ndcg@10': float(np.mean(ndcg10s)),
                'kendall_tau': float(np.mean(taus)), 'spearman': float(np.mean(rhos)),
                'n_groups': len(ndcg5s)}

    print("\nRunning final evaluation on validation set...")
    val_res = full_eval(model, ranker, val_loader)
    print(f"\n{'='*60}")
    print(f"FINAL: {best_name} + {tag}")
    print(f"{'='*60}")
    for k, v in val_res.items(): print(f"  {k:12s}: {v:.4f}" if isinstance(v, float) else f"  {k:12s}: {v}")

    # Ranker loss curve
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(r_losses, alpha=0.3, color='purple', linewidth=0.5)
    w = min(50, len(r_losses) // 5)
    if w > 1:
        sm = np.convolve(r_losses, np.ones(w)/w, mode='valid')
        ax.plot(range(w-1, len(r_losses)), sm, color='purple', linewidth=2)
    ax.set_xlabel('Step'); ax.set_ylabel('Loss')
    ax.set_title(f'RankingTower v2 Loss (base: {best_name})', fontweight='bold')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    loss_path = os.path.join(CKPT_DIR, f'ranker_loss_{tag}.png')
    plt.savefig(loss_path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"Saved: {loss_path}")

    # Save scan report
    report = {'selection_metric': args.metric, 'selected': best_name,
              'checkpoint_scan': results,
              'final_val_metrics': val_res,
              'ranker': {'steps': step, 'path': ranker_path,
                         'temperature': float(ranker.temperature.item())}}
    with open(os.path.join(CKPT_DIR, 'retrain_ranker_report.json'), 'w') as f:
        json.dump(report, f, indent=2)
    print("Saved: retrain_ranker_report.json")

if __name__ == '__main__':
    main()
#!/usr/bin/env python3
"""
Fast RankingTower v2 retraining on a fixed ResonantBERT v3 encoder (~10 min on a laptop GPU).

Differences from retrain_ranker.py, and why:
  * no checkpoint scan: its BBC validation lists were gold-sorted chunks (near-ties),
    so every checkpoint scored tau ~= 0.01 and the scan could not discriminate.
    The encoder is the end-of-training checkpoint (BERTv3-step1881.pt).
  * ranker lists are random articles within a section (not gold-sorted chunks),
    so ListNet sees real score differences.
  * embeddings for a ~20k-article BBC sample (2,500 train + 500 val lists), not all 144k:
    the tower has 8.4k parameters.

Writes model/checkpoints_v3/RANKERv2-fast.pt, appends its CNN scores to
eval_results/cnn_v3/test_scores_cnn_v3.csv and prints the comparison.
"""

import json, os, time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from transformers import AutoTokenizer

from evaluate_v3_cnn import (build_model, load_checkpoint, RankingTowerV2, evaluate_scores, NEWS_VALUES,
                             TOKENIZER_NAME, MAX_SEQ_LEN, CKPT_DIR, BASELINE_DIR, OUTPUT_DIR, CNN_CSV, DEVICE, SEED)
from ablation_study import load_bbc

ENCODER     = "BERTv3-step1881.pt"
TRAIN_LISTS = 2500
VAL_LISTS   = 500
LIST_SIZE   = 8
STEPS       = 1000
LISTS_PER_STEP = 4
LR          = 1e-3
TAG         = "RANKERv2-fast"

def make_lists(df, n, seed):
    rng, lists = np.random.RandomState(seed), []
    for _, sdf in df.groupby("section", sort=True):
        idx = sdf.index.to_numpy()[rng.permutation(len(sdf))]
        lists += [idx[i:i + LIST_SIZE] for i in range(0, len(idx) - LIST_SIZE + 1, LIST_SIZE)]
    return [lists[i] for i in rng.permutation(len(lists))[:n]]

@torch.no_grad()
def embed(model, tok, heads, bodies, bs=32):
    sep, out = tok.sep_token or "[SEP]", []
    for i in range(0, len(heads), bs):
        e = tok([f"{h} {sep} {b}" for h, b in zip(heads[i:i + bs], bodies[i:i + bs])], padding=True,
                truncation=True, max_length=MAX_SEQ_LEN, return_tensors="pt")
        seps = [((ids == tok.sep_token_id).nonzero(as_tuple=True)[0][:1].tolist() or [1])[0] for ids in e["input_ids"]]
        with torch.autocast(DEVICE.type, dtype=torch.bfloat16, enabled=DEVICE.type == "cuda"):
            o = model(e["input_ids"].to(DEVICE), e["attention_mask"].to(DEVICE), torch.tensor(seps, device=DEVICE))
        out.append(o["viral_embedding"].float().cpu())
        print(f"    embedded {min(i + bs, len(heads))}/{len(heads)}", end="\r")
    print()
    return torch.cat(out)

def listnet(scores, gold):   # (L, n)
    return -(torch.softmax(gold, -1) * torch.log_softmax(scores, -1)).sum(-1).mean()

def main():
    t0 = time.time()
    torch.manual_seed(SEED); np.random.seed(SEED)
    tok = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    model = build_model()
    load_checkpoint(model, os.path.join(CKPT_DIR, ENCODER))
    model.eval()

    train, _ = load_bbc("random")
    lists = make_lists(train, TRAIN_LISTS + VAL_LISTS, SEED)
    idx = np.concatenate(lists)
    print(f"Encoder {ENCODER} | embedding {len(idx)} BBC articles ({len(lists)} lists of {LIST_SIZE})")
    sub = train.loc[idx]
    Z = embed(model, tok, sub["title"].astype(str).tolist(), sub["content"].astype(str).tolist())
    Z = Z.view(len(lists), LIST_SIZE, -1).to(DEVICE)
    Y = torch.tensor(sub["gold"].values, dtype=torch.float, device=DEVICE).view(len(lists), LIST_SIZE)
    Ztr, Ytr, Zva, Yva = Z[:TRAIN_LISTS], Y[:TRAIN_LISTS], Z[TRAIN_LISTS:], Y[TRAIN_LISTS:]

    def val_metrics(tower):
        with torch.no_grad():
            s = tower(Zva.reshape(-1, Zva.size(-1))).view_as(Yva)
        tau = np.mean([spearmanr(a, b).correlation for a, b in zip(s.cpu().numpy(), Yva.cpu().numpy())
                       if np.ptp(b) > 0])
        return float(tau), float(spearmanr(s.flatten().cpu(), Yva.flatten().cpu()).correlation)

    tower = RankingTowerV2().to(DEVICE)
    opt = torch.optim.AdamW(tower.parameters(), lr=LR, weight_decay=0.01)
    best, best_state = -1.0, None
    for step in range(1, STEPS + 1):
        tower.train()
        b = torch.randint(0, TRAIN_LISTS, (LISTS_PER_STEP,), device=DEVICE)
        loss = listnet(tower(Ztr[b].reshape(-1, Ztr.size(-1))).view(LISTS_PER_STEP, LIST_SIZE), Ytr[b])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(tower.parameters(), 1.0); opt.step()
        if step % 100 == 0:
            tower.eval()
            within, glob = val_metrics(tower)
            if within > best: best, best_state = within, {k: v.detach().clone() for k, v in tower.state_dict().items()}
            print(f"  step {step:>4} loss={loss.item():.4f} | BBC-val within-list rho={within:.4f} global rho={glob:.4f}")
    tower.load_state_dict(best_state); tower.eval()
    path = os.path.join(CKPT_DIR, f"{TAG}.pt")
    torch.save({"model_state": tower.state_dict(), "ranker_step": STEPS, "base_checkpoint": ENCODER,
                "bbc_val_within_list_spearman": best}, path)
    print(f"Saved {path} (BBC-val within-list rho={best:.4f})")

    # ---- CNN test, same split/metrics as everything else ----
    with open(os.path.join(BASELINE_DIR, "cnn_split.json"), encoding="utf-8") as f:
        split = json.load(f)
    csv = os.path.join(OUTPUT_DIR, "test_scores_cnn_v3.csv")
    df = pd.read_csv(csv, encoding="utf-8")
    assert df["id"].tolist() == split["test_ids"]
    cnn = pd.read_csv(CNN_CSV, encoding="utf-8").set_index("id")
    zt = embed(model, tok, cnn.loc[df["id"], "title"].astype(str).tolist(),
               cnn.loc[df["id"], "content"].astype(str).tolist()).to(DEVICE)
    with torch.no_grad():
        df[f"{ENCODER} [ranker:{TAG}]"] = tower(zt).float().cpu().numpy()
    df.to_csv(csv, index=False)

    row = {i: r for r, i in enumerate(df["id"])}
    groups = [np.array([row[i] for i in g]) for g in split["test_groups"]]
    print("\nCNN test (201 groups):")
    for name, col in [("v3 summed news values", f"{ENCODER} [value_sum]"),
                      ("v3 + NEW fast ranker", f"{ENCODER} [ranker:{TAG}]"),
                      ("v3 + old ranker", "BERTv3-final-step1881.pt [ranker:RANKERv2-1000]"),
                      ("v3 concept head", f"{ENCODER} [concept]"),
                      ("TF-IDF (CNN-trained)", "TF-IDF + Ridge")]:
        if col not in df: continue
        r = evaluate_scores(df[col].to_numpy(dtype=float), df, groups)
        print(f"  {name:<24} NDCG@10={r['ndcg@10']:.4f} tau={r['kendall_tau']:+.4f} "
              f"PA={r['pairwise_acc']:.4f} rho={r['spearman_global']:+.4f}")
    print(f"\nTotal {(time.time() - t0) / 60:.1f} min")

if __name__ == "__main__":
    main()

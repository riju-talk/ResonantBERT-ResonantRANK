#!/usr/bin/env python3
"""
Evaluation-only baselines for the ablation study (no fine-tuning), on both splits:

  random    BBC (all years) -> CNN test split          (eval_results/cnn/cnn_split.json)
  temporal  BBC 2017-2022   -> CNN test, 2023-2025 only (eval_results/cnn/cnn_split_temporal.json)

Methods
  Random, Rule-Based                      no training
  TF-IDF + Ridge (BBC-trained)            BBC train pool of the split
  TF-IDF + Ridge (CNN-trained)            CNN train split (temporal: CNN train articles <= 2022)
  Frozen ModernBERT + Ridge (BBC-trained) mean-pooled ModernBERT-base embeddings, no fine-tuning
  ResonantBERT v3 Kaggle step1881         reference only on the temporal split (it saw 2023+ BBC)

Writes eval_results/ablations/<split>/baselines/{metrics.json, test_scores.csv}
"""

import json, os
import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from scipy.stats import spearmanr
from transformers import AutoModel, AutoTokenizer

from evaluate_v3_cnn import NEWS_VALUES, TOKENIZER_NAME, CNN_CSV, BASELINE_DIR, OUTPUT_DIR as V3_DIR, evaluate_scores
from ablation_study import load_bbc, cnn_test, DATA_SEED

REPO_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_ROOT    = os.path.join(REPO_DIR, "eval_results", "ablations")
CACHE_DIR   = os.path.join(OUT_ROOT, "cache")
FROZEN_N    = 30000          # BBC articles embedded for the frozen-encoder probe
MAX_LEN     = 512
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

URGENCY_WORDS = ['breaking', 'urgent', 'immediately', 'now', 'just in', 'emergency', 'crisis', 'disaster',
                 'alert', 'exclusive', 'developing', 'live', 'update']
EMOTION_WORDS = ['shocking', 'stunning', 'incredible', 'unbelievable', 'outrageous', 'furious', 'devastated',
                 'thrilled', 'amazed', 'horrifying', 'heartbreaking', 'terrifying', 'disgusting', 'inspiring']
NOVELTY_WORDS = ['first', 'unprecedented', 'never before', 'new', 'revealed', 'exclusive', 'breakthrough',
                 'discovery', 'groundbreaking', 'revolutionary', 'surprising', 'unexpected']
SOCIAL_WORDS  = ['death', 'killed', 'dies', 'war', 'attack', 'election', 'president', 'government', 'scandal',
                 'protest', 'arrested', 'charged', 'million', 'billion']

def rule_based_score(heads, bodies):   # identical to the CNN baseline notebook
    out = []
    for h, b in zip(heads, bodies):
        t = (h + " " + b).lower()
        s = (sum(w in t for w in URGENCY_WORDS) * 0.25 + sum(w in t for w in EMOTION_WORDS) * 0.25 +
             sum(w in t for w in NOVELTY_WORDS) * 0.20 + sum(w in t for w in SOCIAL_WORDS) * 0.15 +
             (1.0 if 8 <= len(h.split()) <= 16 else 0.5) * 0.10 + min(len(b.split()) / 500.0, 1.0) * 0.05)
        out.append(s)
    return np.array(out)

def tfidf_ridge(train_texts, train_y, val_texts, val_y, test_texts):
    vec = TfidfVectorizer(max_features=20000, stop_words="english", ngram_range=(1, 2), sublinear_tf=True, min_df=2)
    Xtr, Xva = vec.fit_transform(train_texts), vec.transform(val_texts)
    alpha = max([0.3, 1.0, 3.0, 10.0], key=lambda a: spearmanr(Ridge(alpha=a).fit(Xtr, train_y).predict(Xva), val_y).correlation)
    return Ridge(alpha=alpha).fit(Xtr, train_y).predict(vec.transform(test_texts)), alpha

text = lambda h, b: h + " " + b[:2000]

@torch.no_grad()
def embed(frame, tok, enc, bs=64):
    sep = tok.sep_token or "[SEP]"
    out = []
    for i in range(0, len(frame), bs):
        f = frame.iloc[i:i + bs]
        e = tok([f"{h} {sep} {b}" for h, b in zip(f["title"].astype(str), f["content"].astype(str))],
                padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt").to(DEVICE)
        with torch.autocast(DEVICE.type, dtype=torch.bfloat16, enabled=DEVICE.type == "cuda"):
            h = enc(**e).last_hidden_state.float()
        m = e["attention_mask"].unsqueeze(-1).float()
        out.append(((h * m).sum(1) / m.sum(1)).cpu().numpy())
        print(f"    embedded {min(i + bs, len(frame))}/{len(frame)}", end="\r")
    print()
    return np.concatenate(out)

def cached_embed(name, frame, tok, enc):
    path = os.path.join(CACHE_DIR, f"frozen_{name}.npz")
    if os.path.exists(path):
        z = np.load(path, allow_pickle=True)
        if z["ids"].tolist() == frame["id"].tolist():
            return z["X"]
    X = embed(frame, tok, enc)
    np.savez(path, X=X, ids=np.array(frame["id"].tolist(), dtype=object))
    return X

def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    enc = AutoModel.from_pretrained(TOKENIZER_NAME).to(DEVICE).eval()
    cnn = pd.read_csv(CNN_CSV, encoding="utf-8")
    cnn["gold"] = cnn[NEWS_VALUES].sum(axis=1).astype(float)
    cnn["year"] = pd.to_numeric(cnn["date"].astype(str).str[:4], errors="coerce")
    with open(os.path.join(BASELINE_DIR, "cnn_split.json"), encoding="utf-8") as f:
        cnn_train_ids = set(json.load(f)["train_ids"])
    v3 = pd.read_csv(os.path.join(V3_DIR, "test_scores_cnn_v3.csv"), encoding="utf-8").set_index("id")

    for split in ["random", "temporal"]:
        print(f"\n===== {split} =====")
        out_dir = os.path.join(OUT_ROOT, split, "baselines")
        os.makedirs(out_dir, exist_ok=True)
        train, val = load_bbc(split)                     # same BBC pool/val as the ablation runs
        test, groups = cnn_test(split)                   # builds the temporal split on first use
        test_texts = [text(h, b) for h, b in zip(test["headline"], test["body"])]
        cnn_tr = cnn[cnn["id"].isin(cnn_train_ids)]
        if split == "temporal":
            cnn_tr = cnn_tr[cnn_tr["year"] <= 2022]
        print(f"BBC train={len(train)} val={len(val)} | CNN train={len(cnn_tr)} | CNN test={len(test)} ({len(groups)} groups)")

        scores = test[["id", "section", "gold"] + NEWS_VALUES].copy()
        scores["Random"] = np.random.default_rng(DATA_SEED).random(len(test))
        scores["Rule-Based"] = rule_based_score(test["headline"].tolist(), test["body"].tolist())

        s, a = tfidf_ridge([text(h, b) for h, b in zip(train["title"].astype(str), train["content"].astype(str))],
                           train["gold"], [text(h, b) for h, b in zip(val["title"].astype(str), val["content"].astype(str))],
                           val["gold"], test_texts)
        scores["TF-IDF + Ridge (BBC-trained)"] = s; print(f"  TF-IDF BBC alpha={a}")
        cv = cnn_tr.sample(frac=0.1, random_state=DATA_SEED)
        ct = cnn_tr.drop(cv.index)
        s, a = tfidf_ridge([text(h, b) for h, b in zip(ct["title"].astype(str), ct["content"].astype(str))], ct["gold"],
                           [text(h, b) for h, b in zip(cv["title"].astype(str), cv["content"].astype(str))], cv["gold"],
                           test_texts)
        scores["TF-IDF + Ridge (CNN-trained)"] = s; print(f"  TF-IDF CNN alpha={a}")

        fr = train.sample(n=min(FROZEN_N, len(train)), random_state=DATA_SEED).reset_index(drop=True)
        Xtr = cached_embed(f"{split}_train", fr, tok, enc)
        Xva = cached_embed(f"{split}_val", val, tok, enc)
        tframe = test[["id", "title", "content"]]
        Xte = cached_embed(f"{split}_test", tframe, tok, enc)
        alpha = max([1.0, 10.0, 100.0, 1000.0],
                    key=lambda a: spearmanr(Ridge(alpha=a).fit(Xtr, fr["gold"]).predict(Xva), val["gold"]).correlation)
        scores["Frozen ModernBERT + Ridge (BBC-trained)"] = Ridge(alpha=alpha).fit(Xtr, fr["gold"]).predict(Xte)
        print(f"  Frozen ModernBERT alpha={alpha}")

        col = "BERTv3-step1881.pt [value_sum]"
        if col in v3.columns:
            name = "ResonantBERT v3 Kaggle (full BBC)" + (" [not time-clean]" if split == "temporal" else "")
            scores[name] = v3.loc[scores["id"], col].to_numpy()

        metrics = {}
        for m in [c for c in scores.columns if c not in ["id", "section", "gold"] + NEWS_VALUES]:
            r = evaluate_scores(scores[m].to_numpy(dtype=float), test, groups)
            metrics[m] = r
            print(f"  {m:<46} NDCG@10={r['ndcg@10']:.4f} tau={r['kendall_tau']:+.4f} "
                  f"PA={r['pairwise_acc']:.4f} rho={r['spearman_global']:+.4f}")
        with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        scores.to_csv(os.path.join(out_dir, "test_scores.csv"), index=False)
        print(f"  saved {out_dir}")

if __name__ == "__main__":
    main()

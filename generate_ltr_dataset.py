#!/usr/bin/env python3
"""
Generate a Learning-to-Rank dataset from BBC News YAML articles.

For each article (query), finds the top-k most similar articles using
TF-IDF cosine similarity and outputs a single JSON array.
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── Configuration ────────────────────────────────────────────────────────────
ARTICLES_DIR = Path("/Users/anuragguchhait/development/ResonantBERT-ResonantRANK/bbc_articles/articles")
OUTPUT_PATH = Path("/Users/anuragguchhait/development/ResonantBERT-ResonantRANK/bbc_articles/ranking.json")
TOP_K = 10
MAX_FEATURES = 50_000  # TF-IDF vocabulary cap
BATCH_SIZE = 1_000     # queries processed per batch for cosine sim


def load_articles(articles_dir: Path) -> list[dict]:
    """Load all YAML files and extract id, title, section, content."""
    articles = []
    yaml_files = sorted(f for f in os.listdir(articles_dir) if f.endswith(".yaml"))
    total = len(yaml_files)
    print(f"Loading {total} YAML files...")

    for i, fname in enumerate(yaml_files):
        if (i + 1) % 10_000 == 0 or i == 0:
            print(f"  [{i+1}/{total}] Loading {fname}")
        filepath = articles_dir / fname
        try:
            with open(filepath, "r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh)
            articles.append({
                "id": data.get("id", fname.replace(".yaml", "")),
                "title": data.get("title", ""),
                "section": data.get("section", ""),
                "content": data.get("content", ""),
            })
        except Exception as e:
            print(f"  WARNING: Skipping {fname}: {e}", file=sys.stderr)

    print(f"Loaded {len(articles)} articles successfully.")
    return articles


def build_tfidf(articles: list[dict]) -> csr_matrix:
    """Build TF-IDF matrix from article content (title + content)."""
    print(f"Building TF-IDF matrix (max_features={MAX_FEATURES})...")
    t0 = time.time()

    # Combine title and content for richer representation
    corpus = [f"{a['title']} {a['content']}" for a in articles]

    vectorizer = TfidfVectorizer(
        max_features=MAX_FEATURES,
        stop_words="english",
        sublinear_tf=True,      # log(1 + tf) — standard IR practice
        min_df=2,               # ignore terms appearing in only 1 doc
        max_df=0.95,            # ignore terms appearing in >95% of docs
        dtype=np.float32,       # save memory
    )
    tfidf_matrix = vectorizer.fit_transform(corpus)

    elapsed = time.time() - t0
    print(f"TF-IDF matrix shape: {tfidf_matrix.shape} (built in {elapsed:.1f}s)")
    return tfidf_matrix


def compute_topk_batch(
    tfidf_matrix: csr_matrix,
    articles: list[dict],
    top_k: int,
    batch_size: int,
) -> list[dict]:
    """Compute top-k similar docs for every article, in batches."""
    n = len(articles)
    results = [None] * n
    num_batches = (n + batch_size - 1) // batch_size

    print(f"Computing top-{top_k} for {n} queries in {num_batches} batches...")
    t0 = time.time()

    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, n)
        batch_queries = tfidf_matrix[start:end]

        # Compute cosine similarity: (batch_size × n) dense matrix
        sim_matrix = cosine_similarity(batch_queries, tfidf_matrix)

        for local_i in range(end - start):
            global_i = start + local_i
            scores = sim_matrix[local_i]

            # Zero out self-similarity so the article doesn't retrieve itself
            scores[global_i] = -1.0

            # Get top-k indices (partial sort for efficiency)
            if top_k < len(scores):
                top_indices = np.argpartition(scores, -top_k)[-top_k:]
            else:
                top_indices = np.arange(len(scores))

            # Sort the top-k by score descending
            top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

            top_k_list = []
            for rank, idx in enumerate(top_indices[:top_k], start=1):
                top_k_list.append({
                    "doc_id": articles[idx]["id"],
                    "title": articles[idx]["title"],
                    "rank": rank,
                    "score": round(float(scores[idx]), 6),
                })

            results[global_i] = {
                "query_id": articles[global_i]["id"],
                "query_title": articles[global_i]["title"],
                "top_k": top_k_list,
            }

        elapsed = time.time() - t0
        pct = (end / n) * 100
        eta = (elapsed / end) * (n - end) if end > 0 else 0
        print(f"  Batch {batch_idx+1}/{num_batches} done — "
              f"{end}/{n} queries ({pct:.1f}%) — "
              f"elapsed {elapsed:.0f}s — ETA {eta:.0f}s")

    return results


def validate(results: list[dict], top_k: int):
    """Quick validation checks on the output."""
    print("\nValidating output...")
    assert len(results) > 0, "Empty results!"

    errors = 0
    for i, r in enumerate(results):
        qid = r["query_id"]
        tk = r["top_k"]

        # Check no self-retrieval
        doc_ids = [d["doc_id"] for d in tk]
        if qid in doc_ids:
            print(f"  ERROR: {qid} retrieved itself!")
            errors += 1

        # Check ranks are 1..k
        ranks = [d["rank"] for d in tk]
        expected = list(range(1, len(tk) + 1))
        if ranks != expected:
            print(f"  ERROR: {qid} has non-consecutive ranks: {ranks}")
            errors += 1

        # Check scores are non-increasing
        scores = [d["score"] for d in tk]
        for j in range(1, len(scores)):
            if scores[j] > scores[j-1] + 1e-6:
                print(f"  ERROR: {qid} has non-monotonic scores at rank {j+1}")
                errors += 1
                break

    if errors == 0:
        print(f"  ✓ All {len(results)} query objects passed validation.")
    else:
        print(f"  ✗ {errors} validation errors found.")


def main():
    overall_t0 = time.time()

    # Step 1: Load articles
    articles = load_articles(ARTICLES_DIR)
    if not articles:
        print("No articles found. Exiting.", file=sys.stderr)
        sys.exit(1)

    # Step 2: Build TF-IDF
    tfidf_matrix = build_tfidf(articles)

    # Step 3: Compute top-k
    results = compute_topk_batch(tfidf_matrix, articles, TOP_K, BATCH_SIZE)

    # Step 4: Validate
    validate(results, TOP_K)

    # Step 5: Write output
    print(f"\nWriting {len(results)} query objects to {OUTPUT_PATH}...")
    t0 = time.time()
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Written in {time.time() - t0:.1f}s")

    total_time = time.time() - overall_t0
    file_size_mb = OUTPUT_PATH.stat().st_size / (1024 * 1024)
    print(f"\n{'='*60}")
    print(f"Done! Total time: {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"Output: {OUTPUT_PATH} ({file_size_mb:.1f} MB)")
    print(f"Queries: {len(results)}")
    print(f"Top-k: {TOP_K}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

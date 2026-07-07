"""
BBC News → YAML + JSON Ranking Pipeline
========================================
Source: RealTimeData/bbc_news_alltime (HuggingFace)

Confirmed schema (from HF dataset viewer):
  title           str
  published_date  str  e.g. "2017-01-21"
  authors         str  (often null)
  description     str  (short blurb, often null)
  section         str  (often null for early subsets)
  content         str
  link            str
  top_image       str

Output structure:
  <out_dir>/
    articles/
      bbc_2017_001.yaml
      bbc_2017_002.yaml
      ...
    ranking.json

Usage:
  pip install datasets pandas pyyaml tqdm

  # All subsets (2017-01 → 2025-06, ~102 monthly parquets, ~150k articles)
  python bbc_to_yaml_pipeline.py

  # Single year
  python bbc_to_yaml_pipeline.py --year 2023

  # Single month
  python bbc_to_yaml_pipeline.py --month 2021-05

  # Quick smoke-test (first N rows only)
  python bbc_to_yaml_pipeline.py --limit 200 --out ./test_out
"""

import argparse
import json
import re
import unicodedata
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml
from tqdm import tqdm

# ──────────────────────────────────────────────
# DATASET CONFIG
# ──────────────────────────────────────────────

HF_DATASET = "RealTimeData/bbc_news_alltime"
DEFAULT_OUT = Path("./bbc_articles")

# All known monthly subsets (2017-01 → 2025-06)
ALL_SUBSETS = [
    f"{y}-{m:02d}"
    for y in range(2017, 2026)
    for m in range(1, 13)
    if not (y == 2025 and m > 6)   # dataset ends at 2025-06 as of scrape date
]

# ──────────────────────────────────────────────
# ID COUNTER  (year → sequential int)
# ──────────────────────────────────────────────

_COUNTER: dict[int, int] = defaultdict(int)


def _next_id(year: int) -> str:
    _COUNTER[year] += 1
    return f"bbc_{year}_{_COUNTER[year]:03d}"


# ──────────────────────────────────────────────
# TEXT HELPERS
# ──────────────────────────────────────────────

def _clean(text) -> str:
    if not text or (isinstance(text, float)):   # NaN guard
        return ""
    text = unicodedata.normalize("NFKC", str(text))
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _parse_date(raw) -> str:
    """Return YYYY-MM-DD string or empty string."""
    if not raw or (isinstance(raw, float)):
        return ""
    s = str(raw).strip()
    # already ISO
    if re.match(r"\d{4}-\d{2}-\d{2}", s):
        return s[:10]
    for fmt in ("%d %B %Y", "%B %d, %Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(s, fmt).date().isoformat()
        except ValueError:
            pass
    return s[:10]


def _year(date_str: str) -> int:
    try:
        return int(date_str[:4])
    except (ValueError, TypeError):
        return 0


# ──────────────────────────────────────────────
# YAML LITERAL BLOCK SCALAR  (forces | style)
# ──────────────────────────────────────────────

class _Literal(str):
    pass


def _literal_representer(dumper, data):
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")


yaml.add_representer(_Literal, _literal_representer)


# ──────────────────────────────────────────────
# ROW → YAML DOC
# ──────────────────────────────────────────────

def _row_to_doc(row: pd.Series) -> tuple[str, dict]:
    """
    Map one HF row to (article_id, yaml_doc_dict).
    Uses confirmed column names: title, published_date, authors,
    description, section, content, link, top_image.
    """
    date    = _parse_date(row.get("published_date"))
    year    = _year(date) or 0
    art_id  = _next_id(year)

    title   = _clean(row.get("title"))
    section = _clean(row.get("section")) or "General"
    content = _clean(row.get("content"))
    desc    = _clean(row.get("description"))
    url     = _clean(row.get("link"))

    # Combine description + content for richness, avoiding duplication
    if desc and desc not in content:
        full_content = desc + "\n\n" + content
    else:
        full_content = content

    doc = {
        "id":      art_id,
        "title":   title,
        "date":    date,
        "section": section,
        "url":     url,
        "content": _Literal(full_content) if full_content else "",
    }
    return art_id, doc


# ──────────────────────────────────────────────
# YAML WRITER
# ──────────────────────────────────────────────

def _write_yaml(doc: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.dump(
            doc,
            fh,
            allow_unicode=True,
            default_flow_style=False,
            sort_keys=False,        # preserve schema key order
        )


# ──────────────────────────────────────────────
# PLACEHOLDER SCORER  (swap in ResonantBERT v2.3)
# ──────────────────────────────────────────────

def _score(doc: dict) -> float:
    """
    Stub scorer — replace with your model inference.
    Current heuristic: word count normalised to (0, 1].
    """
    words = len(str(doc.get("content", "")).split())
    return round(min(words / 800, 1.0), 4)


# ──────────────────────────────────────────────
# RANKING JSON BUILDER
# ──────────────────────────────────────────────

def _build_ranking(records: list[tuple[str, dict]]) -> dict:
    """records: [(yaml_filename, doc_dict), ...]"""
    scored = sorted(
        ((fname, _score(doc)) for fname, doc in records),
        key=lambda x: x[1],
        reverse=True,
    )
    return {
        "ranking_date": datetime.now(timezone.utc).date().isoformat(),
        "model":        "placeholder_length_heuristic",  # → resonantbert_v2.3
        "articles": [
            {"yaml_file": fname, "score": score, "rank": i + 1}
            for i, (fname, score) in enumerate(scored)
        ],
    }


# ──────────────────────────────────────────────
# MAIN PIPELINE
# ──────────────────────────────────────────────

def convert(
    out_dir:    Path        = DEFAULT_OUT,
    subsets:    list[str]   = None,
    limit:      int | None  = None,
) -> None:
    articles_dir = out_dir / "articles"
    articles_dir.mkdir(parents=True, exist_ok=True)

    target_subsets = subsets or ALL_SUBSETS
    records: list[tuple[str, dict]] = []
    skipped = 0
    total_rows = 0

    for subset in tqdm(target_subsets, desc="Subsets", unit="month"):
        try:
            # Load one monthly parquet via pandas + HF path
            parquet_url = (
                f"hf://datasets/{HF_DATASET}/{subset}/"
                f"train-00000-of-00001-*.parquet"
            )
            # Use datasets library for reliable wildcard resolution
            from datasets import load_dataset
            ds = load_dataset(
                HF_DATASET,
                name=subset,
                split="train",
                trust_remote_code=True,
            )
            df = ds.to_pandas()
        except Exception as exc:
            tqdm.write(f"  ⚠ Could not load subset {subset}: {exc}")
            continue

        if limit is not None:
            remaining = limit - total_rows
            if remaining <= 0:
                break
            df = df.head(remaining)

        for _, row in df.iterrows():
            if not _clean(row.get("title")):
                skipped += 1
                continue

            art_id, doc = _row_to_doc(row)
            fname = f"{art_id}.yaml"
            _write_yaml(doc, articles_dir / fname)
            records.append((fname, doc))

        total_rows += len(df)
        if limit is not None and total_rows >= limit:
            break

    # ── Ranking JSON ──
    ranking      = _build_ranking(records)
    ranking_path = out_dir / "ranking.json"
    with open(ranking_path, "w", encoding="utf-8") as fh:
        json.dump(ranking, fh, indent=2, ensure_ascii=False)

    print(f"\n✓  {len(records):,} YAML articles  →  {articles_dir}")
    print(f"✓  ranking.json             →  {ranking_path}")
    if skipped:
        print(f"   (skipped {skipped} rows with no title)")


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Convert RealTimeData/bbc_news_alltime → YAML + ranking JSON"
    )
    ap.add_argument(
        "--out", default=str(DEFAULT_OUT),
        help="Output root directory (default: ./bbc_articles)",
    )
    ap.add_argument(
        "--year", type=int, default=None,
        help="Process only a single year, e.g. --year 2023",
    )
    ap.add_argument(
        "--month", default=None,
        help="Process only a single month, e.g. --month 2021-05",
    )
    ap.add_argument(
        "--limit", type=int, default=None,
        help="Stop after N articles total (useful for smoke-testing)",
    )
    args = ap.parse_args()

    # Resolve subset list from filters
    if args.month:
        subsets = [args.month]
    elif args.year:
        subsets = [s for s in ALL_SUBSETS if s.startswith(str(args.year))]
    else:
        subsets = ALL_SUBSETS

    convert(
        out_dir  = Path(args.out),
        subsets  = subsets,
        limit    = args.limit,
    )
    
#!/usr/bin/env python3
"""
backfill_content.py
====================
Adds `content` and `url` columns to an already-labeled CSV (e.g. cnn_labels.csv)
by looking up each row's `id`, finding the matching source YAML file
(<id>.yaml under --dataset-root, searched recursively), and copying its
`content`/`url` fields across. Does NOT re-run any labeling or touch any
score column — pure local join, no API calls.

Output column order matches the BBC schema:
    id,title,date,section,url,content,timeliness,proximity,impact,
    prominence,conflict,novelty,human_interest

USAGE
-----
python backfill_content.py \
    --input cnn_labels.csv \
    --dataset-root /Users/anuragguchhait/development/ResonantBERT-ResonantRANK/cnn_mass_dataset \
    --output cnn_labels_with_content.csv
"""

import argparse
import csv
import logging
import sys
from pathlib import Path

import yaml
from tqdm import tqdm

LOG = logging.getLogger("backfill_content")

RUBRIC_KEYS = [
    "timeliness",
    "proximity",
    "impact",
    "prominence",
    "conflict",
    "novelty",
    "human_interest",
]

OUTPUT_FIELDS = ["id", "title", "date", "section", "url", "content"] + RUBRIC_KEYS


def build_yaml_index(dataset_root: Path) -> dict:
    """Map article id -> Path, by filename (id.yaml), searched recursively.
    Falls back to reading each file's `id:` field only if filename lookup
    misses, to stay robust to any renamed files."""
    LOG.info("Indexing YAML files under %s ...", dataset_root)
    index = {}
    all_files = list(dataset_root.rglob("*.yaml")) + list(dataset_root.rglob("*.yml"))
    for fpath in all_files:
        stem = fpath.stem  # e.g. cnn_2020_00722
        index[stem] = fpath
    LOG.info("Indexed %d YAML files by filename.", len(index))
    return index


def load_yaml_fields(fpath: Path) -> tuple[str, str]:
    """Return (content, url) from a single-article YAML file."""
    try:
        with open(fpath, "r", encoding="utf-8") as f:
            doc = yaml.safe_load(f)
    except Exception as e:
        LOG.warning("Failed to parse %s: %s", fpath, e)
        return "", ""
    if not isinstance(doc, dict):
        return "", ""
    content = doc.get("content") or doc.get("body") or doc.get("text") or ""
    url = doc.get("url") or doc.get("link") or ""
    return str(content), str(url)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Existing labeled CSV (e.g. cnn_labels.csv)")
    parser.add_argument("--dataset-root", required=True, help="Root dir of source YAML files (searched recursively)")
    parser.add_argument("--output", required=True, help="Output CSV with content/url backfilled")
    parser.add_argument("--miss-log", default=None, help="Where to log ids that had no matching YAML (default: <output>.misses.txt)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=[logging.StreamHandler(sys.stdout)])

    input_path = Path(args.input)
    dataset_root = Path(args.dataset_root)
    output_path = Path(args.output)
    miss_log_path = Path(args.miss_log) if args.miss_log else output_path.with_suffix(".misses.txt")

    if not input_path.exists():
        LOG.error("Input CSV not found: %s", input_path)
        sys.exit(1)
    if not dataset_root.exists():
        LOG.error("Dataset root not found: %s", dataset_root)
        sys.exit(1)

    yaml_index = build_yaml_index(dataset_root)

    with open(input_path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    LOG.info("Loaded %d rows from %s", len(rows), input_path)

    missing_ids = []
    out_rows = []
    for row in tqdm(rows, desc="Backfilling"):
        art_id = row.get("id", "")
        fpath = yaml_index.get(art_id)
        if fpath is None:
            missing_ids.append(art_id)
            content, url = "", ""
        else:
            content, url = load_yaml_fields(fpath)

        out_row = {
            "id": art_id,
            "title": row.get("title", ""),
            "date": row.get("date", ""),
            "section": row.get("section", ""),
            "url": url,
            "content": content,
        }
        for k in RUBRIC_KEYS:
            out_row[k] = row.get(k, "")
        out_rows.append(out_row)

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(out_rows)

    if missing_ids:
        with open(miss_log_path, "w", encoding="utf-8") as f:
            f.write("\n".join(missing_ids))
        LOG.warning(
            "%d/%d ids had no matching YAML file — left content/url blank for those. "
            "See %s for the list.",
            len(missing_ids), len(rows), miss_log_path,
        )
    else:
        LOG.info("All %d rows matched a source YAML file successfully.", len(rows))

    LOG.info("Wrote %d rows to %s", len(out_rows), output_path)


if __name__ == "__main__":
    main()

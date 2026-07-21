#!/usr/bin/env python3
"""
label_cnn_track_a.py
=====================
Stratified-samples ~5,000 articles out of a ~19k CNN YAML corpus (2017-2025),
then labels them via the NVIDIA NIM API using the Track A editorial
scorecard rubric (track_a_final.txt), writing incremental, resumable output.

USAGE
-----
export NVIDIA_API_KEY="nvapi-xxxxxxxx"

python label_cnn_track_a.py \
    --input /Users/anuragguchhait/development/ResonantBERT-ResonantRANK/cnn_mass_dataset \
    --output /home/claude/cnn_track_a_labels.csv \
    --sample-size 5000 \
    --model mistralai/mistral-nemo-12b-instruct \
    --concurrency 6 \
    --rpm 40

--input accepts either a single YAML file, or (as in this project's case) a
root directory with one subfolder per year (2017/, 2018/, ..., 2025/), each
containing one or more .yaml/.yml files. The script recursively discovers
every YAML file under the given root regardless of how deep or how the
files are split within each year folder.

Re-running with the same --output resumes automatically: already-labeled
article ids are skipped.

DATASET ASSUMPTIONS
--------------------
- Input is a single YAML file, OR a directory tree (year subfolders or any
  nesting) containing YAML files. Each YAML file may hold a top-level list
  of article dicts, a dict with a list under a common key
  ("articles"/"data"/"items"/"records"), or a single article dict (e.g. if
  the crawler wrote one file per article).
- Field names are auto-detected from a set of common aliases (see
  FIELD_ALIASES below). If your schema uses something exotic, edit
  FIELD_ALIASES rather than the rest of the script.
- An article-level unique id is required for checkpointing/resume. If no
  explicit id/url field exists, a stable hash of (title+date+body[:200]) is
  used instead.

SAMPLING STRATEGY (as specified)
---------------------------------
Stratify by (year, section). Within each stratum, cap the number drawn at
an equal per-stratum quota computed via water-filling, so that:
  - no single (year, section) bucket can dominate the 5k sample,
  - small strata just contribute everything they have,
  - the leftover quota is redistributed across remaining strata,
  - the final sample sums to exactly --sample-size (or less, if the full
    corpus itself has fewer than --sample-size articles).
This maximizes cross-temporal and cross-section diversity in the eval set.
"""

import argparse
import asyncio
import csv
import hashlib
import json
import logging
import math
import os
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

load_dotenv(override=True)

import httpx
import yaml
from dateutil import parser as dateparser
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)
from tqdm import tqdm

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

NIM_ENDPOINT = "https://integrate.api.nvidia.com/v1/chat/completions"

RUBRIC_KEYS = [
    "timeliness",
    "proximity",
    "impact",
    "prominence",
    "conflict",
    "novelty",
    "human_interest",
]

# Common field-name aliases seen across BBC/CNN pipeline YAML exports.
# Edit these lists if your actual schema differs.
FIELD_ALIASES: dict[str, list[str]] = {
    "id": ["id", "article_id", "uuid"],
    "url": ["url", "link", "source_url"],
    "title": ["title", "headline"],
    "date": ["date", "published_date", "publish_date", "publish_time", "timestamp"],
    "section": ["section", "category"],
    "body": ["body", "content", "text", "article_body"],
    "publisher": ["publisher", "source", "outlet"],
}

LOG = logging.getLogger("label_cnn_track_a")


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------

@dataclass
class Article:
    id: str
    title: str
    date_str: str
    year: Optional[int]
    section: str
    body: str
    publisher: str
    raw: dict = field(default_factory=dict)


def _first_present(d: dict, keys: list[str]) -> Optional[Any]:
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return d[k]
    return None


def _stable_id(title: str, date_str: str, body: str) -> str:
    h = hashlib.sha1(f"{title}|{date_str}|{body[:200]}".encode("utf-8", "ignore"))
    return h.hexdigest()[:16]


def _extract_year(date_str: str) -> Optional[int]:
    if not date_str:
        return None
    try:
        return dateparser.parse(str(date_str), fuzzy=True).year
    except Exception:
        return None


def _year_from_path(path: Path) -> Optional[int]:
    """Fallback: infer year from a parent directory named e.g. '2017'."""
    for part in path.parts:
        if len(part) == 4 and part.isdigit() and 2000 <= int(part) <= 2100:
            return int(part)
    return None


def _yaml_docs_to_entries(raw: Any) -> list[dict]:
    """Normalize a loaded YAML object into a flat list of article dicts.
    Handles: top-level list, dict-with-list-under-common-key, or a single
    article represented as one dict (common when there's one YAML file per
    article rather than one file per year)."""
    if isinstance(raw, list):
        return [e for e in raw if isinstance(e, dict)]
    if isinstance(raw, dict):
        for key in ("articles", "data", "items", "records"):
            if key in raw and isinstance(raw[key], list):
                return [e for e in raw[key] if isinstance(e, dict)]
        # Looks like a single article dict (has body/title-ish keys)
        if any(k in raw for k in FIELD_ALIASES["body"] + FIELD_ALIASES["title"]):
            return [raw]
    return []


def _iter_yaml_files(root: Path):
    for ext in ("*.yaml", "*.yml"):
        yield from root.rglob(ext)


def load_articles(path: str) -> list[Article]:
    root = Path(path)
    articles: list[Article] = []
    skipped_no_body = 0
    skipped_bad_file = 0

    if root.is_dir():
        LOG.info("Input is a directory — walking for YAML files under %s", root)
        yaml_files = sorted(_iter_yaml_files(root))
        LOG.info("Found %d YAML files across year subfolders", len(yaml_files))
        if not yaml_files:
            raise ValueError(f"No .yaml/.yml files found anywhere under {root}")
    else:
        yaml_files = [root]

    for fpath in tqdm(yaml_files, desc="Reading YAML files"):
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f)
        except Exception as e:
            LOG.warning("Skipping unreadable file %s: %s", fpath, e)
            skipped_bad_file += 1
            continue

        entries = _yaml_docs_to_entries(raw)
        path_year_fallback = _year_from_path(fpath)

        for entry in entries:
            title = _first_present(entry, FIELD_ALIASES["title"]) or ""
            date_str = _first_present(entry, FIELD_ALIASES["date"]) or ""
            section = _first_present(entry, FIELD_ALIASES["section"]) or "unknown"
            body = _first_present(entry, FIELD_ALIASES["body"]) or ""
            publisher = _first_present(entry, FIELD_ALIASES["publisher"]) or "CNN"
            art_id = _first_present(entry, FIELD_ALIASES["id"]) or _first_present(
                entry, FIELD_ALIASES["url"]
            )

            if not body.strip():
                skipped_no_body += 1
                continue

            if not art_id:
                art_id = _stable_id(str(title), str(date_str), str(body))
            else:
                art_id = str(art_id)

            year = _extract_year(str(date_str)) or path_year_fallback

            articles.append(
                Article(
                    id=art_id,
                    title=str(title),
                    date_str=str(date_str),
                    year=year,
                    section=str(section).strip().lower() or "unknown",
                    body=str(body),
                    publisher=str(publisher),
                    raw=entry,
                )
            )

    LOG.info(
        "Loaded %d usable articles (skipped %d empty-body, %d unreadable files)",
        len(articles),
        skipped_no_body,
        skipped_bad_file,
    )
    return articles


# --------------------------------------------------------------------------
# Stratified sampling: capped-equal water-filling by (year, section)
# --------------------------------------------------------------------------

def stratified_sample(
    articles: list[Article], target: int, seed: int = 42
) -> list[Article]:
    rng = random.Random(seed)

    strata: dict[tuple[Optional[int], str], list[Article]] = defaultdict(list)
    for a in articles:
        strata[(a.year, a.section)].append(a)

    # Shuffle within each stratum so the "first N" we take are random, not
    # file-order-biased.
    for key in strata:
        rng.shuffle(strata[key])

    sizes = {k: len(v) for k, v in strata.items()}
    remaining_target = min(target, sum(sizes.values()))
    remaining_strata_keys = sorted(sizes.keys(), key=lambda k: sizes[k])
    allocation: dict[tuple[Optional[int], str], int] = {}

    n_left = len(remaining_strata_keys)
    for key in remaining_strata_keys:
        if n_left == 0:
            break
        quota = math.ceil(remaining_target / n_left)
        take = min(sizes[key], quota)
        allocation[key] = take
        remaining_target -= take
        n_left -= 1

    sampled: list[Article] = []
    for key, n in allocation.items():
        sampled.extend(strata[key][:n])

    rng.shuffle(sampled)

    LOG.info(
        "Stratified sample: %d articles across %d (year, section) strata "
        "(target was %d)",
        len(sampled),
        len(allocation),
        target,
    )
    return sampled


def print_sample_breakdown(sampled: list[Article]) -> None:
    by_year: dict[Any, int] = defaultdict(int)
    by_section: dict[Any, int] = defaultdict(int)
    for a in sampled:
        by_year[a.year] += 1
        by_section[a.section] += 1
    LOG.info("-- Sample breakdown by year --")
    for y in sorted(by_year, key=lambda x: (x is None, x)):
        LOG.info("  %s: %d", y, by_year[y])
    LOG.info("-- Sample breakdown by section (top 15) --")
    for s, c in sorted(by_section.items(), key=lambda kv: -kv[1])[:15]:
        LOG.info("  %s: %d", s, c)


# --------------------------------------------------------------------------
# NIM API labeling
# --------------------------------------------------------------------------

class RateLimiter:
    """Simple token-bucket limiter to respect requests-per-minute caps."""

    def __init__(self, rpm: int):
        self.min_interval = 60.0 / max(rpm, 1)
        self._lock = asyncio.Lock()
        self._last = 0.0

    async def wait(self):
        async with self._lock:
            now = time.monotonic()
            delta = now - self._last
            if delta < self.min_interval:
                await asyncio.sleep(self.min_interval - delta)
            self._last = time.monotonic()


class RetryableAPIError(Exception):
    pass


def build_user_message(a: Article) -> str:
    return (
        f"TITLE: {a.title}\n"
        f"PUBLICATION DATE: {a.date_str}\n"
        f"SECTION: {a.section}\n"
        f"ARTICLE BODY:\n{a.body}"
    )


def parse_scores(text: str) -> Optional[dict]:
    text = text.strip()
    # Strip stray code fences if the model adds them despite instructions.
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end == -1:
            return None
        try:
            obj = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None

    if not isinstance(obj, dict):
        return None
    if set(obj.keys()) != set(RUBRIC_KEYS):
        return None
    for k in RUBRIC_KEYS:
        v = obj[k]
        if not isinstance(v, int) or not (0 <= v <= 5):
            return None
    return obj


@retry(
    retry=retry_if_exception_type(RetryableAPIError),
    wait=wait_random_exponential(multiplier=1, max=30),
    stop=stop_after_attempt(5),
    reraise=True,
)
async def call_nim(
    client: httpx.AsyncClient,
    limiter: RateLimiter,
    api_key: str,
    model: str,
    system_prompt: str,
    article: Article,
) -> dict:
    await limiter.wait()
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": build_user_message(article)},
        ],
        "temperature": 0.0,
        "max_tokens": 200,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    try:
        resp = await client.post(
            NIM_ENDPOINT, json=payload, headers=headers, timeout=60.0
        )
    except httpx.RequestError as e:
        raise RetryableAPIError(f"network error: {type(e).__name__} {e}") from e

    if resp.status_code == 429 or resp.status_code >= 500:
        raise RetryableAPIError(f"status {resp.status_code}: {resp.text[:200]}")
    if resp.status_code != 200:
        raise ValueError(f"non-retryable status {resp.status_code}: {resp.text[:300]}")

    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    scores = parse_scores(content)
    if scores is None:
        raise RetryableAPIError(f"unparseable/invalid model output: {content[:200]}")
    return scores


# --------------------------------------------------------------------------
# Checkpointing / incremental CSV writer
# --------------------------------------------------------------------------

CSV_FIELDS = ["id", "title", "year", "section", "publisher", "date"] + RUBRIC_KEYS


def load_done_ids(output_path: str) -> set[str]:
    if not os.path.exists(output_path):
        return set()
    done = set()
    with open(output_path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("id"):
                done.add(row["id"])
    return done


def append_result(output_path: str, article: Article, scores: dict, write_header: bool):
    row = {
        "id": article.id,
        "title": article.title,
        "year": article.year,
        "section": article.section,
        "publisher": article.publisher,
        "date": article.date_str,
        **scores,
    }
    with open(output_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def append_failure(fail_log_path: str, article: Article, error: str):
    with open(fail_log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"id": article.id, "title": article.title, "error": error}) + "\n")


# --------------------------------------------------------------------------
# Main pipeline
# --------------------------------------------------------------------------

async def run_labeling(
    articles: list[Article],
    output_path: str,
    fail_log_path: str,
    api_key: str,
    model: str,
    system_prompt: str,
    concurrency: int,
    rpm: int,
):
    done_ids = load_done_ids(output_path)
    todo = [a for a in articles if a.id not in done_ids]
    LOG.info(
        "%d already labeled (resumed), %d remaining to label",
        len(done_ids),
        len(todo),
    )
    if not todo:
        LOG.info("Nothing left to do.")
        return

    write_header = not os.path.exists(output_path) or os.path.getsize(output_path) == 0
    header_lock = asyncio.Lock()
    file_lock = asyncio.Lock()
    limiter = RateLimiter(rpm)
    sem = asyncio.Semaphore(concurrency)

    ok_count = 0
    fail_count = 0

    async with httpx.AsyncClient() as client:

        async def worker(article: Article, pbar: tqdm):
            nonlocal ok_count, fail_count, write_header
            async with sem:
                try:
                    scores = await call_nim(
                        client, limiter, api_key, model, system_prompt, article
                    )
                    async with file_lock:
                        async with header_lock:
                            hdr = write_header
                            write_header = False
                        append_result(output_path, article, scores, hdr)
                    ok_count += 1
                except Exception as e:  # noqa: BLE001 - log and continue, don't kill the batch
                    LOG.warning("FAILED id=%s: %s", article.id, e)
                    append_failure(fail_log_path, article, str(e))
                    fail_count += 1
            pbar.update(1)

        with tqdm(total=len(todo), desc="Labeling") as pbar:
            await asyncio.gather(*[worker(a, pbar) for a in todo])

    LOG.info("Done. Success: %d, Failed: %d (see %s)", ok_count, fail_count, fail_log_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the CNN dataset — either a single YAML file, or a root "
        "directory containing year subfolders (e.g. cnn_mass_dataset/2017/, "
        "cnn_mass_dataset/2018/, ...), each holding one or more .yaml/.yml files. "
        "The script recursively finds every YAML file under the given path.",
    )
    parser.add_argument(
        "--output",
        default="cnn_track_a_labels.csv",
        help="Output CSV (append/resume-safe)",
    )
    parser.add_argument(
        "--fail-log", default=None, help="JSONL log of failed articles (default: <output>.failures.jsonl)"
    )
    parser.add_argument(
        "--rubric",
        default=str(Path(__file__).parent / "track_a_final.txt"),
        help="Path to the Track A system-prompt rubric file",
    )
    parser.add_argument("--sample-size", type=int, default=5000)
    parser.add_argument(
        "--model",
        default="mistralai/mistral-nemo-12b-instruct",
        help="NVIDIA NIM model string. Fallback: mistralai/mixtral-8x7b-instruct-v0.1",
    )
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument(
        "--rpm", type=int, default=35, help="Requests per minute cap (stay under NIM free-tier limit)"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only run sampling + print breakdown, no API calls",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    fail_log_path = args.fail_log or f"{args.output}.failures.jsonl"

    articles = load_articles(args.input)
    sampled = stratified_sample(articles, args.sample_size, seed=args.seed)
    print_sample_breakdown(sampled)

    # Persist the exact sample so it's auditable / re-runnable without re-sampling.
    sample_manifest = Path(args.output).with_suffix(".sample_ids.txt")
    sample_manifest.write_text("\n".join(a.id for a in sampled))
    LOG.info("Sample id manifest written to %s", sample_manifest)

    if args.dry_run:
        LOG.info("--dry-run set: stopping before API calls.")
        return

    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        LOG.error("NVIDIA_API_KEY environment variable not set. Aborting.")
        sys.exit(1)

    system_prompt = Path(args.rubric).read_text(encoding="utf-8")

    asyncio.run(
        run_labeling(
            sampled,
            args.output,
            fail_log_path,
            api_key,
            args.model,
            system_prompt,
            args.concurrency,
            args.rpm,
        )
    )


if __name__ == "__main__":
    main()

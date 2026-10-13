<div align="center">

# ResonantBERT + ResonantRANK

**Content-only virality prediction and ranking for news articles**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Status: Research](https://img.shields.io/badge/Status-Research-purple.svg)](#)

</div>

---

## Table of Contents
- [Overview](#overview)
- [Repository Layout](#repository-layout)
- [Installation](#installation)
- [Datasets](#datasets)
- [Usage](#usage)
- [Evaluation and Paper](#evaluation-and-paper)
- [Citation](#citation)
- [License](#license)

---

## Overview

**ResonantBERT-ResonantRANK** is a two-stage system that ranks news articles by their intrinsic virality potential using **content features only** (no timestamps, user profiles or engagement history).

1. **ResonantBERT** (encoder) turns a headline and body into a 128-d viral embedding plus auxiliary signals (risk score, factor attributions).
2. **Resonant Ranker** turns that embedding into a calibrated, risk-adjusted virality score.

Model implementations live in `model/`; trained weights are in `model/checkpoints_v3/`. Architecture details are covered in the paper (`paper/`) and `docs/`.

---

## Repository Layout

```
app/                 FastAPI serving code
data/                dataset download helpers
bbc_articles/        BBC training set (LFS)
cnn_mass_dataset/    CNN evaluation set + labeling scripts (LFS)
mind_data/           MIND dataset bundle, Mind_Data.zip (LFS)
scripts/             evaluation, ablation and retraining scripts
notebooks/           evaluation notebooks
eval_results/        metrics, tables and figures
paper/               LaTeX source of the v3 paper
docs/                design notes and change logs
```

Large files (`*.pt`, `*.npz`, datasets, `Mind_Data.zip`) are stored with **Git LFS**. Run `git lfs install` before cloning.

---

## Installation

```bash
git lfs install
git clone https://github.com/riju-talk/ResonantBERT-ResonantRANK.git
cd ResonantBERT-ResonantRANK

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

Requirements: Python 3.8+, PyTorch 2.0+, Hugging Face Transformers 4.30+, scikit-learn.

---

## Datasets

| Dataset | Location | Use |
|---------|----------|-----|
| BBC articles | `bbc_articles/` (`articles/`, `ranking.json`, `track_a_all.csv`) | Training |
| CNN articles | `cnn_mass_dataset/<year>/*.yaml`, `cnn_labels.csv`, `cnn_labels_with_content.csv` | Evaluation |
| MIND | `mind_data/Mind_Data.zip` | Additional data (see below) |

**MIND bundle.** Unzip `mind_data/Mind_Data.zip` to get `Mind_Data/` containing `news.csv`, `behaviors.csv`, `mind_labels.csv`, `entity_embedding.vec` and `relation_embedding.vec`. Ignore the `__MACOSX/` entries.

```bash
unzip mind_data/Mind_Data.zip -x "__MACOSX/*" -d mind_data/
```

**Article format (YAML):**
```yaml
headline: "Article Title"
body: "Full article text..."
publish_date: "2017-01-01"
source: "bbc"
topic: "politics"
virality_label: 0.75  # optional ground-truth score
```

---

## Usage

Run scripts from the repo root:

```bash
python scripts/evaluate_v3_cnn.py       # evaluate v3 on CNN
python scripts/ablation_study.py        # ablations
python scripts/retrain_ranker_fast.py   # retrain the ranker
python scripts/make_v3_figures.py       # regenerate figures
```

Minimal inference:

```python
import torch
from model.ResonantBERT import ResonantBERT
from model.RankingTower import ResonantRanker

encoder = ResonantBERT(backbone_name="answerdotai/ModernBERT-base")
ranker = ResonantRanker(in_dim=128, calibration="platt")

with torch.no_grad():
    enc_out = encoder(headline_ids, headline_mask, body_ids, body_mask)
    rank_out = ranker(enc_out["viral_embedding"], enc_out["risk_score"])
    print(rank_out["final_score"])
```

Serving: see `app/` (FastAPI).

---

## Evaluation and Paper

Metrics, tables and figures are in `eval_results/` (e.g. `eval_results/cnn_v3/`). The paper source is `paper/resonantbert_v3_paper.tex`. Notebooks in `notebooks/` reproduce the evaluation.

---

## Citation

```bibtex
@misc{ResonantBERT2024,
  title={ResonantBERT-ResonantRANK: End-to-End Pure Viral Ranking System},
  author={Your Name},
  year={2024},
  note={Content-only architecture for intrinsic virality prediction}
}
```

---

## License

MIT License.

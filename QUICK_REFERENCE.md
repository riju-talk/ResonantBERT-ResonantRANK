# ResonantBERT-ResonantRANK: Quick Reference Guide

## 🚀 Quick Start

### Basic Usage
```python
from model.ResonantBERT import ResonantBERT
from model.RankingTower import ResonantRanker, ResonantViralPipeline

# Initialize encoder and ranker
encoder = ResonantBERT()
ranker = ResonantRanker(calibration="platt")

# Create end-to-end pipeline
pipeline = ResonantViralPipeline(encoder, ranker)

# Rank articles
result = pipeline.rank_candidates(batch)
print(f"Virality Score: {result['final_score']}")
```

---

## 📐 Dimension Reference

### Input Dimensions
```
Headlines:    (B, seq_len)  → tokenized headlines
Bodies:       (B, seq_len)  → tokenized article bodies
Masks:        (B, seq_len)  → attention masks (0=padding, 1=valid)
```

### ResonantBERT Outputs
```
viral_embedding:      (B, 128)    ✓ L2-normalized
concept_score:        (B,)        ✓ concept resonance
risk_score:           (B,)        ✓ in [0, 1]
factor_attributions:  (B, 9)      ✓ per-factor scores
F_prime:              (B, 9, 128) ✓ factor matrix (for inspection)
```

### ResonantRanker Outputs
```
raw_score:            (B,)        ✓ unbounded
calibrated_score:     (B,)        ✓ in [0, 1]
final_score:          (B,)        ✓ in [0, 1], risk-adjusted
```

---

## 🔧 Architecture Overview

### Part 1: ResonantBERT (Encoder)

```
Headlines + Bodies (seq_len)
    ↓
Dual-Stream ModernBERT (768-dim each)
    ↓
Cross-Document Alignment (64-dim each)
    ↓
Evidence: [eₕ, eᵦ, aₕ, aᵦ] → 1664-dim
    ↓
9 Virality Factors (9 × 128-dim)
    ↓
Factorized Self-Attention (9 × 128-dim)
    ↓
Residual-Query Cross-Attention (r, c: 128-dim)
    ↓
Gated Fusion → 128-dim viral embedding
    ↓
Auxiliary: concept_score, risk_score, factor_attributions
```

### Part 2: ResonantRanker (Scorer)

```
Viral Embedding (128-dim)
    ↓
Ranking Tower (gated MLPs: 128→128→128→64→1)
    ↓
Raw Score
    ↓
Calibration (Platt or Isotonic)
    ↓
Calibrated Score [0, 1]
    ↓
Risk Adjustment × (1 - λ × risk_score)
    ↓
Final Score [0, 1]
```

---

## 9️⃣ Virality Factors

| # | Factor | Meaning |
|----|--------|---------|
| 1 | **Novelty** | New information / unexpectedness |
| 2 | **Arousal** | Emotional activation intensity |
| 3 | **Urgency** | Time-sensitivity / immediacy |
| 4 | **Social Impact** | Relevance to social groups |
| 5 | **Readability** | Accessibility / clarity |
| 6 | **Specificity** | Concreteness / detail level |
| 7 | **Narrative** | Story structure strength |
| 8 | **Proportion** | Content depth vs. headline ratio |
| 9 | **Affect** | Emotional valence (positive/negative) |

Access via:
```python
factor_names = [
    "novelty", "arousal", "urgency", "social_impact", "readability",
    "specificity", "narrative", "proportion", "affect"
]

for name, score in zip(factor_names, output["factor_attributions"][0]):
    print(f"{name}: {score:.4f}")
```

---

## 🎛️ Configuration Options

### ResonantBERT
```python
encoder = ResonantBERT(
    backbone_name="answerdotai/ModernBERT-base",  # HuggingFace model
    hidden_dim=768,                                # BERT hidden size
    align_dim=64,                                  # alignment dimension
    factor_dim=128,                                # factor dimension
    num_factors=9,                                 # number of factors
    p_drop=0.1,                                    # dropout probability
)
```

### ResonantRanker
```python
ranker = ResonantRanker(
    in_dim=128,                    # viral embedding dimension
    p_drop=0.1,                    # dropout probability
    calibration="platt",           # "platt" or "isotonic"
    risk_lambda=0.3,               # risk penalty weight
)
```

---

## 📊 Loss Functions (Training)

### ResonantBERT Training
```python
L_total = α₁×L_ranking + α₂×L_contrastive + α₃×L_risk + α₄×L_consist

Where:
  L_ranking      = ListNet Loss (pairwise ranking)
  L_contrastive  = Triplet Loss (hard negative mining)
  L_risk         = BCE (risk prediction)
  L_consist      = Factor consistency regularization
```

### ResonantRanker Training
```python
L_total = α₁×L_ranking + α₂×L_calibration + α₃×L_risk_adjust

Where:
  L_ranking         = Ranking loss on final scores
  L_calibration     = Calibration loss (score → probability)
  L_risk_adjust     = Risk adjustment auxiliary loss
```

---

## 🔄 Data Flow Example

### Inference
```python
# Input batch
batch = {
    "headline_input_ids": (4, 128),      # B=4, seq_len=128
    "headline_attention_mask": (4, 128),
    "body_input_ids": (4, 128),
    "body_attention_mask": (4, 128),
}

# Encoder
enc_out = encoder(
    batch["headline_input_ids"],
    batch["headline_attention_mask"],
    batch["body_input_ids"],
    batch["body_attention_mask"],
)
# enc_out["viral_embedding"]: (4, 128)
# enc_out["risk_score"]: (4,) in [0, 1]

# Ranker
rank_out = ranker(
    enc_out["viral_embedding"],  # (4, 128)
    enc_out["risk_score"],       # (4,)
)
# rank_out["final_score"]: (4,) in [0, 1]

# Results
for i, score in enumerate(rank_out["final_score"]):
    print(f"Article {i}: {score:.4f}")
```

---

## 🧮 Key Formulas

### Gated Residual Layer
```
h = Dropout(GeLU(LayerNorm(Linear(x))))
g = sigmoid(Linear(x))
output = g ⊙ h + (1 - g) ⊙ residual_proj(x)
```

### Gated Fusion
```
flat = concat[F'(flat), c, r]  # (B, 1408)
gate_values = sigmoid(Linear(flat))
gated = flat ⊙ gate_values
output = Linear → LayerNorm → GeLU → (B, 128)
```

### Platt Scaling
```
calibrated_score = sigmoid(a × raw_score + b)
```

### Risk Adjustment
```
final_score = calibrated_score × (1 - λ × risk_score)
λ ∈ [0, 1]
```

---

## 🎯 Common Workflows

### 1. Rank Multiple Articles
```python
# Tokenize articles
headlines = ["Title 1", "Title 2", "Title 3"]
bodies = ["Body 1", "Body 2", "Body 3"]

# Tokenize (using your tokenizer)
headline_ids = tokenizer(headlines, return_tensors="pt")["input_ids"]
body_ids = tokenizer(bodies, return_tensors="pt")["input_ids"]

# Create batch
batch = {
    "headline_input_ids": headline_ids,
    "headline_attention_mask": torch.ones_like(headline_ids),
    "body_input_ids": body_ids,
    "body_attention_mask": torch.ones_like(body_ids),
}

# Rank
results = pipeline.rank_candidates(batch)

# Sort by virality
sorted_indices = torch.argsort(results["final_score"], descending=True)
for idx in sorted_indices:
    print(f"Rank: {results['final_score'][idx]:.4f}")
```

### 2. Analyze Factor Breakdown
```python
enc_out = encoder(...)
factors = enc_out["factor_attributions"][0]  # (9,)

factor_names = [...]  # 9 factor names
for name, value in zip(factor_names, factors):
    print(f"{name:15s}: {value:.4f}")
```

### 3. Use Isotonic Calibration
```python
# Fit calibrator on validation set
from model.RankingTower import IsotonicCalibrator

calibrator = IsotonicCalibrator()
calibrator.fit(raw_scores_val, labels_val)

# Use in inference
ranker = ResonantRanker(calibration="isotonic")
result = ranker(viral_embedding, risk_score, external_calibrator=calibrator)
```

### 4. Extract Embeddings for Downstream
```python
enc_out = encoder(...)
viral_embedding = enc_out["viral_embedding"]  # (B, 128), L2-normalized

# Use for clustering, retrieval, etc.
similarity = viral_embedding @ viral_embedding.T  # cosine similarity
```

---

## 🐛 Debugging Tips

### Check Dimension Mismatch
```python
try:
    result = encoder(h_ids, h_mask, b_ids, b_mask)
except RuntimeError as e:
    print(f"Dimension error: {e}")
    print(f"h_ids shape: {h_ids.shape}")
    print(f"b_ids shape: {b_ids.shape}")
```

### Verify Value Ranges
```python
enc_out = encoder(...)
print(f"viral_embedding norm: {torch.norm(enc_out['viral_embedding'], dim=-1)}")
print(f"risk_score range: [{enc_out['risk_score'].min()}, {enc_out['risk_score'].max()}]")

rank_out = ranker(enc_out["viral_embedding"], enc_out["risk_score"])
print(f"final_score range: [{rank_out['final_score'].min()}, {rank_out['final_score'].max()}]")
```

### Test Calibration
```python
ranker = ResonantRanker(calibration="platt")
enc_out = encoder(...)
rank_out = ranker(enc_out["viral_embedding"], enc_out["risk_score"])

# Check calibrated scores are in [0, 1]
assert rank_out["calibrated_score"].min() >= 0
assert rank_out["calibrated_score"].max() <= 1
```

---

## 📚 Files Reference

| File | Purpose | Key Classes |
|------|---------|------------|
| `ResonantBERT.py` | Viral embedding encoder | ResonantBERT, CrossDocumentAlignment, NineFactorHeads, ... |
| `RankingTower.py` | Virality scoring ranker | ResonantRanker, RankingTower, PlattScaling, RiskAdjustment, ... |
| `test_architecture.py` | Validation tests | test_resonantbert_architecture(), test_ranking_tower_architecture(), ... |

---

## ✅ Validation Checklist

Before deployment:
- [ ] Test inference on sample batch
- [ ] Verify all output dimensions
- [ ] Check value ranges
- [ ] Validate L2-normalization of embeddings
- [ ] Test both calibration modes
- [ ] Run full test suite: `python test_architecture.py`

---

## 📞 Quick Troubleshooting

| Problem | Solution |
|---------|----------|
| `RuntimeError: Expected X dimension, got Y` | Check batch shapes and tokenization |
| `NaN in outputs` | Check for gradient explosion; reduce learning rate |
| `Scores not in [0, 1]` | Verify calibration is applied; check risk_score range |
| `Embedding not L2-normalized` | Check F.normalize is applied in fusion_head |
| `Memory error` | Reduce batch size; use gradient checkpointing |

---

## 🚀 Next Steps

1. **Prepare data** - Tokenize articles with appropriate tokenizer
2. **Initialize models** - Load ResonantBERT and ResonantRanker
3. **Train encoder** - Use ranking + contrastive losses
4. **Calibrate ranker** - Fit Platt or Isotonic on validation set
5. **Evaluate** - Measure ranking metrics (NDCG, MRR, etc.)
6. **Deploy** - Use ResonantViralPipeline for inference

---

**For full documentation, see README.md and ARCHITECTURE_VERIFICATION.md**

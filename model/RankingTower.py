import torch
import torch.nn as nn
from typing import Optional, Dict


# ============================================================
# PART 2: Resonant Ranker V1 (Ranking Tower)
# ============================================================

class GatedResidualLayer(nn.Module):
    """Linear -> LN -> GeLU -> Dropout, fused with the input via a learned gate."""
    def __init__(self, dim_in: int, dim_out: int, p_drop: float = 0.1):
        super().__init__()
        self.linear = nn.Linear(dim_in, dim_out)
        self.ln = nn.LayerNorm(dim_out)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(p_drop)
        self.gate = nn.Linear(dim_in, dim_out)
        self.residual_proj = nn.Identity() if dim_in == dim_out else nn.Linear(dim_in, dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.dropout(self.act(self.ln(self.linear(x))))
        g = torch.sigmoid(self.gate(x))
        return g * h + (1 - g) * self.residual_proj(x)


class RankingTower(nn.Module):
    """Deep MLP with gated residuals: 128 -> 128 -> 128 -> 64 -> 1 (raw score)."""
    def __init__(self, in_dim: int = 128, p_drop: float = 0.1):
        super().__init__()
        self.layer1 = GatedResidualLayer(in_dim, 128, p_drop)
        self.layer2 = GatedResidualLayer(128, 128, p_drop)
        self.layer3 = GatedResidualLayer(128, 64, p_drop)
        self.output_layer = nn.Linear(64, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.layer1(z)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.output_layer(x).squeeze(-1)  # raw score s_i, (B,)


class PlattScaling(nn.Module):
    """Parametric calibration: s̃ = sigmoid(a*s + b)."""
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(1.0))
        self.b = nn.Parameter(torch.tensor(0.0))

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.a * s + self.b)


class IsotonicCalibrator:
    """Non-parametric calibration. Fit post-hoc on held-out (raw_score, label) pairs
    with sklearn; not part of the autograd graph, so it's a plain Python object
    rather than an nn.Module.
    
    Requires: scikit-learn
    """
    def __init__(self):
        from sklearn.isotonic import IsotonicRegression
        self.model = IsotonicRegression(out_of_bounds="clip")
        self._fitted = False

    def fit(self, raw_scores: torch.Tensor, labels: torch.Tensor):
        self.model.fit(raw_scores.detach().cpu().numpy(), labels.detach().cpu().numpy())
        self._fitted = True

    def predict(self, raw_scores: torch.Tensor) -> torch.Tensor:
        if not self._fitted:
            raise RuntimeError("IsotonicCalibrator must be fit before predict().")
        out = self.model.predict(raw_scores.detach().cpu().numpy())
        return torch.tensor(out, dtype=raw_scores.dtype, device=raw_scores.device)


class RiskAdjustment(nn.Module):
    """Final score = calibrated_score * (1 - lambda * risk_score)."""
    def __init__(self, lam: float = 0.3, learnable: bool = False):
        super().__init__()
        if learnable:
            self.lam = nn.Parameter(torch.tensor(lam))
        else:
            self.register_buffer("lam", torch.tensor(lam))

    def forward(self, calibrated_score: torch.Tensor, risk_score: torch.Tensor) -> torch.Tensor:
        lam = torch.clamp(self.lam, 0.0, 1.0)
        return calibrated_score * (1.0 - lam * risk_score)


class ResonantRanker(nn.Module):
    """
    Takes ResonantBERT's 128-d viral embedding and produces a calibrated,
    risk-adjusted virality score.
    """
    def __init__(self, in_dim: int = 128, p_drop: float = 0.1,
                 calibration: str = "platt", risk_lambda: float = 0.3):
        super().__init__()
        self.tower = RankingTower(in_dim, p_drop)
        self.calibration_mode = calibration
        if calibration == "platt":
            self.calibrator = PlattScaling()
        else:
            self.calibrator = None  # use IsotonicCalibrator externally, post-hoc
        self.risk_adjustment = RiskAdjustment(lam=risk_lambda)

    def forward(self, viral_embedding: torch.Tensor,
                risk_score: torch.Tensor,
                external_calibrator: Optional[IsotonicCalibrator] = None) -> Dict[str, torch.Tensor]:
        raw_score = self.tower(viral_embedding)  # s_i, (B,)

        if self.calibration_mode == "platt":
            calibrated = self.calibrator(raw_score)
        else:
            if external_calibrator is None:
                raise ValueError("Isotonic calibration requires a fitted IsotonicCalibrator instance.")
            calibrated = external_calibrator.predict(raw_score)

        final_score = self.risk_adjustment(calibrated, risk_score)  # ŝ_i ∈ [0,1]

        return {
            "raw_score": raw_score,
            "calibrated_score": calibrated,
            "final_score": final_score,
        }


class HardNegativeMiner:
    """Training-time utility (not part of the forward graph): in-batch negatives,
    teacher/cross-encoder hard negatives, and triplet formation."""

    @staticmethod
    def in_batch_negatives(embeddings: torch.Tensor, exclude_diag: bool = True) -> torch.Tensor:
        # embeddings: (M, D), L2-normalized. Returns pairwise similarity matrix.
        sim = embeddings @ embeddings.T
        if exclude_diag:
            sim.fill_diagonal_(-float("inf"))
        return sim  # (M, M)

    @staticmethod
    def teacher_hard_negatives(cross_encoder_scores: torch.Tensor, top_k: int = 5) -> torch.Tensor:
        # cross_encoder_scores: (M,) scores for candidate pairs against an anchor
        _, idx = torch.topk(cross_encoder_scores, k=min(top_k, cross_encoder_scores.numel()))
        return idx

    @staticmethod
    def form_triplets(anchor: torch.Tensor, positive: torch.Tensor,
                       hard_negative: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {"anchor": anchor, "positive": positive, "hard_negative": hard_negative}


# ============================================================
# End-to-end wrapper (encoder -> ranker)
# ============================================================

class ResonantViralPipeline(nn.Module):
    """Convenience wrapper chaining ResonantBERT -> ResonantRanker for inference."""
    def __init__(self, encoder, ranker):
        super().__init__()
        self.encoder = encoder
        self.ranker = ranker

    @torch.no_grad()
    def rank_candidates(self, batch) -> Dict[str, torch.Tensor]:
        enc_out = self.encoder(
            batch["headline_input_ids"], batch["headline_attention_mask"],
            batch["body_input_ids"], batch["body_attention_mask"],
        )
        rank_out = self.ranker(enc_out["viral_embedding"], enc_out["risk_score"])
        return {**enc_out, **rank_out}
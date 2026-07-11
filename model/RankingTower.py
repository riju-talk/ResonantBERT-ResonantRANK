import torch
import torch.nn as nn
from typing import Optional, Dict


# ============================================================
# PART 2: Resonant Ranker V1 (Ranking Tower)
# ============================================================

class GatedResidualLayer(nn.Module):
    """Gated residual connection: Linear -> LN -> GeLU -> Dropout, fused with input via gate.
    
    Formula:
        h = Dropout(GeLU(LayerNorm(Linear(x))))  # main path
        g = sigmoid(Linear(x))                    # learned gate
        output = g ⊙ h + (1 - g) ⊙ residual_proj(x)
    
    This architecture:
    - Preserves gradients through depth (residual connection)
    - Learns adaptive routing (gating mechanism)
    - Handles dimension changes smoothly
    
    Args:
        dim_in: Input dimension
        dim_out: Output dimension
        p_drop: Dropout probability
    """
    def __init__(self, dim_in: int, dim_out: int, p_drop: float = 0.1):
        super().__init__()
        self.linear = nn.Linear(dim_in, dim_out)
        self.ln = nn.LayerNorm(dim_out)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(p_drop)
        self.gate = nn.Linear(dim_in, dim_out)
        self.residual_proj = nn.Identity() if dim_in == dim_out else nn.Linear(dim_in, dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, dim_in) or (batch, *, dim_in)
        Returns:
            output: (B, dim_out) or (batch, *, dim_out)
        """
        h = self.dropout(self.act(self.ln(self.linear(x))))  # main path
        g = torch.sigmoid(self.gate(x))                       # gate values
        return g * h + (1 - g) * self.residual_proj(x)        # gated residual


class RankingTower(nn.Module):
    """Deep MLP with gated residuals: 128 -> 128 -> 128 -> 64 -> 1 (raw score).
    
    Architecture:
    - Input: (B, 128) viral embedding
    - Layer 1-3: Gated residual layers with residual connections
    - Output layer: Linear(64) -> 1 raw score
    - Final output shape: (B,) - batch of scores
    """
    def __init__(self, in_dim: int = 128, p_drop: float = 0.1):
        super().__init__()
        self.layer1 = GatedResidualLayer(in_dim, 128, p_drop)
        self.layer2 = GatedResidualLayer(128, 128, p_drop)
        self.layer3 = GatedResidualLayer(128, 64, p_drop)
        self.output_layer = nn.Linear(64, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, 128) - viral embedding
        Returns:
            raw_score: (B,) - unsqueezed scores
        """
        x = self.layer1(z)  # (B, 128)
        x = self.layer2(x)  # (B, 128)
        x = self.layer3(x)  # (B, 64)
        raw_score = self.output_layer(x)  # (B, 1)
        return raw_score.squeeze(-1)  # (B,)


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
    """Ranking tower that transforms viral embeddings into calibrated virality scores.
    
    Architecture:
    - RankingTower: Deep MLP with gated residuals (128→128→128→64→1)
    - Calibration: Platt scaling (parametric) or Isotonic regression (non-parametric)
    - Risk adjustment: Final score modulated by controversy level
    
    Outputs:
    - raw_score: (B,) unbounded scores from tower
    - calibrated_score: (B,) scores in [0, 1]
    - final_score: (B,) risk-adjusted scores
    
    Input: Viral embedding (B, 128) + risk score (B,)
    Output: Calibrated virality scores with risk weighting
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
        """Transform viral embedding into calibrated, risk-adjusted virality scores.
        
        Args:
            viral_embedding: (B, 128) - L2-normalized viral embedding from ResonantBERT
            risk_score: (B,) - risk score in [0, 1] from ResonantBERT
            external_calibrator: IsotonicCalibrator instance for isotonic mode
        
        Returns:
            Dictionary containing:
                - raw_score: (B,) - unbounded raw scores from ranking tower
                - calibrated_score: (B,) - scores in [0, 1] after calibration
                - final_score: (B,) - risk-adjusted scores in [0, 1]
        
        Processing:
            1. raw_score = RankingTower(viral_embedding)
            2. calibrated_score = Calibrator(raw_score)  [Platt or Isotonic]
            3. final_score = calibrated_score × (1 - λ × risk_score)
        """
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
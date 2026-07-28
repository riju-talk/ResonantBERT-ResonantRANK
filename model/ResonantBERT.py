import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM
from typing import Optional, Tuple, Dict


# ============================================================
# PART 1: ResonantBERT V1 (Viral Embedding Encoder)
# ============================================================

class CrossDocumentAlignment(nn.Module):
    """Learned query attends from one stream over the other stream's tokens.
    Produces a compact 'aligned evidence' vector (dim 64) per stream."""
    def __init__(self, hidden_dim: int = 768, out_dim: int = 64, num_heads: int = 8):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True
        )
        self.proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, other_stream_tokens: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # other_stream_tokens: (B, m, 768)
        B = other_stream_tokens.size(0)
        q = self.query.expand(B, -1, -1)  # (B, 1, 768)
        attn_out, _ = self.attn(
            q, other_stream_tokens, other_stream_tokens,
            key_padding_mask=key_padding_mask
        )  # (B, 1, 768)
        return self.proj(attn_out.squeeze(1))  # (B, 64)


class FactorHead(nn.Module):
    """One of the 9 virality-factor projections.
    Input: concat[e_h, e_b, a_h, a_b] -> R^1664 -> R^128"""
    def __init__(self, in_dim: int = 1664, hidden_dim: int = 256, out_dim: int = 128, p_drop: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class NineFactorHeads(nn.Module):
    """Novelty, Arousal, Urgency, Social Impact, Readability,
    Specificity, Narrative, Proportion, Affect."""
    FACTOR_NAMES = [
        "novelty", "arousal", "urgency", "social_impact", "readability",
        "specificity", "narrative", "proportion", "affect",
    ]

    def __init__(self, in_dim: int = 1664, out_dim: int = 128):
        super().__init__()
        self.heads = nn.ModuleList([FactorHead(in_dim, out_dim=out_dim) for _ in self.FACTOR_NAMES])
        self.out_dim = out_dim

    def forward(self, evidence: torch.Tensor) -> torch.Tensor:
        # evidence: (B, 1664) -> F: (B, 9, 128)
        outs = [head(evidence) for head in self.heads]
        return torch.stack(outs, dim=1)


class FactorizedSelfAttention(nn.Module):
    """Each of the 9 factor vectors attends to all others in their shared 128-d subspace."""
    def __init__(self, dim: int = 128, num_heads: int = 4, num_layers: int = 1, p_drop: float = 0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=num_heads, dim_feedforward=dim * 4,
            dropout=p_drop, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, F_in: torch.Tensor) -> torch.Tensor:
        # F_in: (B, 9, 128) -> F': (B, 9, 128)
        return self.encoder(F_in)


class ResidualQueryCrossAttention(nn.Module):
    """Residual stream (from e_h, e_b) queries the factorized evidence matrix F'."""
    def __init__(self, hidden_dim: int = 768, factor_dim: int = 128, num_heads: int = 4, p_drop: float = 0.1):
        super().__init__()
        self.residual_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, factor_dim * 2),
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(factor_dim * 2, factor_dim),
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=factor_dim, num_heads=num_heads, batch_first=True
        )

    def forward(self, e_h: torch.Tensor, e_b: torch.Tensor,
                F_prime: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # e_h, e_b: (B, 768); F_prime: (B, 9, 128)
        r = self.residual_mlp(torch.cat([e_h, e_b], dim=-1))  # (B, 128)
        q = r.unsqueeze(1)  # (B, 1, 128)
        c, _ = self.cross_attn(q, F_prime, F_prime)  # (B, 1, 128)
        return r, c.squeeze(1)  # r: (B,128), c: (B,128)


class GatedFusionHead(nn.Module):
    """Fuses F' (flattened), c, and r via a learned gate, then projects to the
    final 128-d viral embedding.
    
    Formula: gated = flat ⊙ sigmoid(Linear(flat))
             output = LayerNorm(GeLU(Linear(gated)))
    
    Dimensions:
    - F'(flat): (B, 9*128) = (B, 1152)
    - c: (B, 128)
    - r: (B, 128)
    - concat_dim = 1152 + 128 + 128 = (B, 1408)
    """
    def __init__(self, num_factors: int = 9, factor_dim: int = 128, out_dim: int = 128, p_drop: float = 0.1):
        super().__init__()
        concat_dim = num_factors * factor_dim + factor_dim + factor_dim  # F'(flat) + c + r = 1408
        self.gate = nn.Linear(concat_dim, concat_dim)
        self.proj = nn.Sequential(
            nn.Linear(concat_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(p_drop),
        )

    def forward(self, F_prime: torch.Tensor, c: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        B = F_prime.size(0)
        flat = torch.cat([F_prime.reshape(B, -1), c, r], dim=-1)  # (B, 1408)
        gate_values = torch.sigmoid(self.gate(flat))  # (B, 1408)
        gated = flat * gate_values
        return self.proj(gated)  # (B, 128)


class ResonantBERT(nn.Module):
    """Dual-stream late-interaction encoder producing 128-d viral embeddings.
    
    Architecture:
    - Shared-weight dual-stream ModernBERT backbone
    - Cross-document alignment (headline ↔ body interaction)
    - 9 virality factor heads (novelty, arousal, urgency, etc.)
    - Factorized self-attention over factors
    - Residual-query cross-attention for concept fusion
    - Gated fusion head for final viral embedding
    
    Outputs:
    - viral_embedding: (B, 128) L2-normalized
    - concept_score: (B,) core concept resonance
    - risk_score: (B,) controversy level
    - factor_attributions: (B, 9) factor breakdown
    
    Input: News articles (headline + body)
    Output: Viral embedding + auxiliary signals
    """
    def __init__(
        self,
        backbone_name: str = "answerdotai/ModernBERT-base",
        hidden_dim: int = 768,
        align_dim: int = 64,
        factor_dim: int = 128,
        num_factors: int = 9,
        p_drop: float = 0.1,
    ):
        super().__init__()
        # 1. Shared-weight dual-stream backbone
        self.backbone = AutoModelForMaskedLM.from_pretrained(backbone_name)
        self.hidden_dim = hidden_dim

        # 2. Cross-document alignment (separate modules per direction)
        self.align_headline_to_body = CrossDocumentAlignment(hidden_dim, align_dim)
        self.align_body_to_headline = CrossDocumentAlignment(hidden_dim, align_dim)

        # 3. 9 factor heads
        evidence_dim = hidden_dim * 2 + align_dim * 2  # e_h + e_b + a_h + a_b = 1664
        self.factor_heads = NineFactorHeads(in_dim=evidence_dim, out_dim=factor_dim)

        # 4. Factorized self-attention over the 9 factors
        self.factorized_attn = FactorizedSelfAttention(dim=factor_dim)

        # 5. Residual-query cross-attention
        self.residual_query_attn = ResidualQueryCrossAttention(hidden_dim, factor_dim)

        # 6. Fusion & output head
        self.fusion_head = GatedFusionHead(num_factors, factor_dim, out_dim=factor_dim, p_drop=p_drop)

        # Auxiliary heads
        self.concept_head = nn.Linear(factor_dim, 1)                 # from c
        self.risk_head = nn.Sequential(nn.Linear(factor_dim, 1), nn.Sigmoid())  # from fused embedding
        self.factor_attr_head = nn.Linear(factor_dim, 1)             # applied per-factor -> 9-d vector

    def _encode_stream(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        token_reps = out.last_hidden_state          # (B, m, 768)
        pooled = token_reps[:, 0, :]                 # [CLS], (B, 768)
        return token_reps, pooled

    def forward(
        self,
        headline_input_ids, headline_attention_mask,
        body_input_ids, body_attention_mask,
    ) -> Dict[str, torch.Tensor]:
        """Encode article headline + body into viral embedding and auxiliary signals.
        
        Args:
            headline_input_ids: (B, seq_len) - tokenized headline
            headline_attention_mask: (B, seq_len) - padding mask for headline
            body_input_ids: (B, seq_len) - tokenized body
            body_attention_mask: (B, seq_len) - padding mask for body
        
        Returns:
            Dictionary containing:
                - viral_embedding: (B, 128) - L2-normalized viral embedding
                - concept_score: (B,) - core concept resonance
                - risk_score: (B,) - controversy/risk level in [0, 1]
                - factor_attributions: (B, 9) - virality factors breakdown
                - F_prime: (B, 9, 128) - factorized evidence matrix (for inspection)
        
        Processing Pipeline:
            1. Dual-stream encoding with ModernBERT
            2. Cross-document alignment (headline ↔ body)
            3. Evidence construction: concat[eₕ, eᵦ, aₕ, aᵦ] → (B, 1664)
            4. 9 Factor heads: evidence → (B, 9, 128)
            5. Factorized self-attention over factors
            6. Residual-query cross-attention (r ⊥ F')
            7. Gated fusion: F' + c + r → (B, 128) viral embedding
            8. Auxiliary outputs: concept, risk, factor attributions
        """
        # 1. Dual-stream backbone (shared weights)
        H, e_h = self._encode_stream(headline_input_ids, headline_attention_mask)  # H:(B,m,768), e_h:(B,768)
        Bd, e_b = self._encode_stream(body_input_ids, body_attention_mask)          # Bd:(B,m,768), e_b:(B,768)

        # 2. Cross-document alignment
        headline_kpm = (headline_attention_mask == 0)
        body_kpm = (body_attention_mask == 0)
        a_h = self.align_headline_to_body(Bd, key_padding_mask=body_kpm)      # (B, 64)
        a_b = self.align_body_to_headline(H, key_padding_mask=headline_kpm)   # (B, 64)

        # 3. 9 factor heads
        evidence = torch.cat([e_h, e_b, a_h, a_b], dim=-1)  # (B, 1664)
        Fmat = self.factor_heads(evidence)                   # (B, 9, 128)

        # 4. Factorized self-attention
        F_prime = self.factorized_attn(Fmat)                 # (B, 9, 128)

        # 5. Residual-query cross-attention
        r, c = self.residual_query_attn(e_h, e_b, F_prime)   # r:(B,128), c:(B,128)

        # 6. Fusion & output head
        fused = self.fusion_head(F_prime, c, r)               # (B, 128)
        viral_embedding = F.normalize(fused, p=2, dim=-1)     # z̃_i, L2-normalized

        # 7. Auxiliary outputs
        concept_score = self.concept_head(c).squeeze(-1)                  # (B,)
        risk_score = self.risk_head(fused).squeeze(-1)                    # (B,) in [0,1]
        
        # Factor attributions: apply linear head to each of 9 factors, then squeeze
        factor_attr = self.factor_attr_head(F_prime)  # (B, 9, 1)
        factor_attr = factor_attr.squeeze(-1)         # (B, 9)

        return {
            "viral_embedding": viral_embedding,   # z̃_i ∈ R^128
            "concept_score": concept_score,       # c_i
            "risk_score": risk_score,             # r_i ∈ [0,1]
            "factor_attributions": factor_attr,   # φ_i ∈ R^9
            "F_prime": F_prime,                   # exposed for L_cons / inspection
        }

import torch
import torch.nn as nn

class RankingHead(nn.Module):
    """
    Takes the fused contextual + emotional representation and maps it to a single virality score.
    """
    def __init__(self, input_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

    def forward(self, fused_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            fused_features: (batch_size, input_dim)
        Returns:
            score: (batch_size,)
        """
        # (batch_size, 1) -> (batch_size,)
        return self.mlp(fused_features).squeeze(-1)

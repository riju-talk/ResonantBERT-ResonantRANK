import torch
import torch.nn as nn
from transformers import AutoModel

class ArticleEncoder(nn.Module):
    """
    Article encoder utilizing a ColBERTv2-inspired late interaction strategy.
    Instead of pooling into a single [CLS] token, it outputs the sequence of token embeddings
    for finer-grained semantic representation.
    """
    def __init__(self, model_name: str = "bert-base-uncased", projection_dim: int = 128):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        # Optional ColBERT-style dimension reduction for efficiency
        self.linear = nn.Linear(self.encoder.config.hidden_size, projection_dim, bias=False)
        self.projection_dim = projection_dim

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
        Returns:
            token_embeddings: (batch_size, seq_len, projection_dim)
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Get last hidden state
        hidden_states = outputs.last_hidden_state  # (B, L, D)
        
        # Project to lower dimension and L2 normalize (ColBERT style)
        projected = self.linear(hidden_states)
        normalized = torch.nn.functional.normalize(projected, p=2, dim=-1)
        
        # Apply mask so padded tokens become exactly zero vectors
        normalized = normalized * attention_mask.unsqueeze(-1)
        
        return normalized

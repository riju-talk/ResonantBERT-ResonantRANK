import torch
import torch.nn as nn

class ResonanceFusionLayer(nn.Module):
    """
    Fuses the semantic token embeddings from the ArticleEncoder
    with the emotion probabilities from the EmotionEncoder.
    """
    def __init__(self, semantic_dim: int = 128, emotion_dim: int = 7, hidden_dim: int = 256):
        super().__init__()
        # We will use an attention mechanism where the emotion acts as a query
        # over the semantic tokens.
        self.emotion_proj = nn.Linear(emotion_dim, semantic_dim)
        self.attention = nn.MultiheadAttention(embed_dim=semantic_dim, num_heads=4, batch_first=True)
        
        # After attention, we fuse them with a feed-forward layer
        self.fc = nn.Sequential(
            nn.Linear(semantic_dim + emotion_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.output_dim = hidden_dim

    def forward(self, article_embs: torch.Tensor, emotion_embs: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            article_embs: (batch_size, seq_len, semantic_dim)
            emotion_embs: (batch_size, emotion_dim)
            attention_mask: (batch_size, seq_len)
        Returns:
            fused_repr: (batch_size, hidden_dim)
        """
        # Project emotion vector to match semantic dimension
        # emotion_embs shape: (B, emotion_dim) -> (B, 1, semantic_dim)
        emotion_query = self.emotion_proj(emotion_embs).unsqueeze(1)
        
        # We need a boolean key_padding_mask for MultiheadAttention
        # 1 means padded (True), 0 means valid (False)
        key_padding_mask = (attention_mask == 0)
        
        # Cross-attention: emotion queries semantic tokens
        # attn_output shape: (B, 1, semantic_dim)
        attn_output, _ = self.attention(
            query=emotion_query,
            key=article_embs,
            value=article_embs,
            key_padding_mask=key_padding_mask
        )
        attn_output = attn_output.squeeze(1) # (B, semantic_dim)
        
        # Concatenate attention output with the original raw emotion signals
        fused_input = torch.cat([attn_output, emotion_embs], dim=-1) # (B, semantic_dim + emotion_dim)
        
        # Final fully connected layer
        fused_repr = self.fc(fused_input) # (B, hidden_dim)
        
        return fused_repr

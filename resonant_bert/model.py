import torch
import torch.nn as nn
from typing import Dict, Any

from .modules.article_encoder import ArticleEncoder
from .modules.emotion_encoder import EmotionEncoder
from .modules.fusion_layer import ResonanceFusionLayer
from .modules.ranking_head import RankingHead

class ResonantBERT(nn.Module):
    """
    The unified ResonantRANK model combining semantic representations (ColBERT-style)
    with emotional resonance, fused together to predict a virality score.
    """
    def __init__(
        self,
        article_model_name: str = "bert-base-uncased",
        emotion_model_name: str = "j-hartmann/emotion-english-distilroberta-base",
        projection_dim: int = 128,
        fusion_hidden_dim: int = 256
    ):
        super().__init__()
        
        # Instantiate encoders
        self.article_encoder = ArticleEncoder(
            model_name=article_model_name,
            projection_dim=projection_dim
        )
        self.emotion_encoder = EmotionEncoder(
            model_name=emotion_model_name
        )
        
        # Get emotion dim dynamically from the loaded model
        emotion_dim = self.emotion_encoder.emotion_dim
        
        # Instantiate fusion and ranking head
        self.fusion_layer = ResonanceFusionLayer(
            semantic_dim=projection_dim,
            emotion_dim=emotion_dim,
            hidden_dim=fusion_hidden_dim
        )
        self.ranking_head = RankingHead(
            input_dim=fusion_hidden_dim
        )

    def forward(self, 
                article_input_ids: torch.Tensor, 
                article_attention_mask: torch.Tensor,
                emotion_input_ids: torch.Tensor = None,
                emotion_attention_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            article_input_ids: (B, L_art)
            article_attention_mask: (B, L_art)
            emotion_input_ids: (B, L_emo) - If None, uses article_input_ids
            emotion_attention_mask: (B, L_emo) - If None, uses article_attention_mask
            
        Returns:
            scores: (B,)
        """
        # Fallback to shared tokenization if separate emotion inputs aren't provided
        if emotion_input_ids is None:
            emotion_input_ids = article_input_ids
        if emotion_attention_mask is None:
            emotion_attention_mask = article_attention_mask

        # 1. Get semantic token embeddings
        article_embs = self.article_encoder(
            input_ids=article_input_ids,
            attention_mask=article_attention_mask
        )
        
        # 2. Get emotion vector
        emotion_embs = self.emotion_encoder(
            input_ids=emotion_input_ids,
            attention_mask=emotion_attention_mask
        )
        
        # 3. Fuse representations
        fused_repr = self.fusion_layer(
            article_embs=article_embs,
            emotion_embs=emotion_embs,
            attention_mask=article_attention_mask
        )
        
        # 4. Predict score
        scores = self.ranking_head(fused_features=fused_repr)
        
        return scores

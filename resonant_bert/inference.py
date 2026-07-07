import torch
from typing import List, Dict, Any

from .model import ResonantBERT

class InferencePipeline:
    """
    Inference pipeline for ResonantBERT.
    Takes raw texts (or pre-tokenized inputs), scores them, and returns a ranked list.
    """
    def __init__(self, model: ResonantBERT, tokenizer, device: torch.device):
        self.model = model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.device = device

    @torch.no_grad()
    def rank(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Args:
            articles: List of dicts, e.g., [{"id": 1, "text": "Headline. Content...", ...}]
        Returns:
            The same list of dicts, updated with 'score' and 'rank', sorted descending.
        """
        if not articles:
            return []

        texts = [a["text"] for a in articles]
        
        # Tokenize
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)
        
        # Forward pass
        scores = self.model(
            article_input_ids=inputs["input_ids"],
            article_attention_mask=inputs["attention_mask"]
        )
        scores = scores.cpu().tolist()
        
        # Add scores to articles
        for article, score in zip(articles, scores):
            article["score"] = float(score)
            
        # Sort by score descending
        ranked_articles = sorted(articles, key=lambda x: x["score"], reverse=True)
        
        # Add rank
        for i, article in enumerate(ranked_articles, start=1):
            article["rank"] = i
            
        return ranked_articles

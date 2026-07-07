import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification

class EmotionEncoder(nn.Module):
    """
    Extracts emotional and sentiment features from the text.
    Can be initialized from a pretrained emotion classification model
    (e.g., j-hartmann/emotion-english-distilroberta-base) or just a sentiment model.
    """
    def __init__(self, model_name: str = "j-hartmann/emotion-english-distilroberta-base"):
        super().__init__()
        # Load a model that naturally predicts emotional dimensions
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.emotion_dim = self.model.config.num_labels

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
        Returns:
            emotion_probs: (batch_size, emotion_dim) - the extracted emotional signals.
        """
        # Exclude token_type_ids since Roberta/DistilBERT don't use them
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        # Use softmax to get a stable probability distribution over emotions
        probs = torch.nn.functional.softmax(logits, dim=-1)
        return probs

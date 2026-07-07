import torch
from transformers import AutoTokenizer

from resonant_bert.model import ResonantBERT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ModelLoader:
    @staticmethod
    def load():
        """Load the trainable ResonantBERT model and tokenizer"""
        # Load the base tokenizer for the article encoder
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        
        # Initialize ResonantBERT with default configurations
        # In a real scenario, you would load state_dict here from a checkpoint
        model = ResonantBERT(
            article_model_name="bert-base-uncased",
            emotion_model_name="j-hartmann/emotion-english-distilroberta-base",
            projection_dim=128,
            fusion_hidden_dim=256
        ).to(device)
        
        model.eval()
        
        return model, tokenizer

model_loader = ModelLoader()
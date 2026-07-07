from resonant_bert.inference import InferencePipeline
from model.loader import device as device_

class ViralRankNet:
    """
    Wrapper around the ResonantBERT InferencePipeline to maintain backward compatibility
    with the existing app/main.py structure.
    """
    
    def __init__(self, model, tokenizer, device):
        self.pipeline = InferencePipeline(model=model, tokenizer=tokenizer, device=device)
        
    def score(self, headlines, contents):
        """
        Score articles based on ResonantBERT prediction.
        
        Args:
            headlines: List[str] - article headlines
            contents: List[str] - full article bodies
            
        Returns:
            List[float] - virality scores
        """
        # Format for InferencePipeline
        articles_dicts = [
            {"text": f"{h} {c}"} for h, c in zip(headlines, contents)
        ]
        
        # The rank method returns dictionaries with "score" and "rank"
        ranked_results = self.pipeline.rank(articles_dicts)
        
        # Extract the scores in the original order
        # We need to map them back since pipeline.rank sorts them
        scores = [0.0] * len(headlines)
        original_texts = [f"{h} {c}" for h, c in zip(headlines, contents)]
        
        for result in ranked_results:
            # Find the original index
            original_idx = original_texts.index(result["text"])
            scores[original_idx] = result["score"]
            
        return scores


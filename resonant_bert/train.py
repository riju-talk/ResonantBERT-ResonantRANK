import torch
from torch.utils.data import DataLoader
from typing import Callable
from tqdm import tqdm

from .model import ResonantBERT
from .losses.ranking import MarginRankingLoss

class Trainer:
    """
    Training pipeline for ResonantBERT using pairwise MarginRankingLoss.
    """
    def __init__(
        self,
        model: ResonantBERT,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        margin: float = 1.0
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.loss_fn = MarginRankingLoss(margin=margin).to(device)

    def train_epoch(self, dataloader: DataLoader) -> float:
        """
        Dataloader is expected to yield batches of pairs: (positive_batch, negative_batch)
        Each batch is a dict of input_ids, attention_mask.
        """
        self.model.train()
        total_loss = 0.0
        
        for batch in tqdm(dataloader, desc="Training"):
            pos_inputs = {k: v.to(self.device) for k, v in batch['positive'].items()}
            neg_inputs = {k: v.to(self.device) for k, v in batch['negative'].items()}
            
            self.optimizer.zero_grad()
            
            # Forward pass for positive (viral) articles
            score_pos = self.model(
                article_input_ids=pos_inputs['input_ids'],
                article_attention_mask=pos_inputs['attention_mask']
            )
            
            # Forward pass for negative (less viral) articles
            score_neg = self.model(
                article_input_ids=neg_inputs['input_ids'],
                article_attention_mask=neg_inputs['attention_mask']
            )
            
            # Compute margin ranking loss
            loss = self.loss_fn(score_pos, score_neg)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(dataloader)

    def evaluate(self, dataloader: DataLoader) -> float:
        """
        Evaluates the model on a validation set.
        Returns the average loss.
        """
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                pos_inputs = {k: v.to(self.device) for k, v in batch['positive'].items()}
                neg_inputs = {k: v.to(self.device) for k, v in batch['negative'].items()}
                
                score_pos = self.model(
                    article_input_ids=pos_inputs['input_ids'],
                    article_attention_mask=pos_inputs['attention_mask']
                )
                
                score_neg = self.model(
                    article_input_ids=neg_inputs['input_ids'],
                    article_attention_mask=neg_inputs['attention_mask']
                )
                
                loss = self.loss_fn(score_pos, score_neg)
                total_loss += loss.item()
                
        return total_loss / len(dataloader)

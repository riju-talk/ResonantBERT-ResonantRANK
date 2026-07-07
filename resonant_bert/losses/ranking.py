import torch
import torch.nn as nn
import torch.nn.functional as F

class MarginRankingLoss(nn.Module):
    """
    Standard pairwise margin ranking loss.
    Tries to enforce: score_pos > score_neg + margin
    """
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.loss_fn = nn.MarginRankingLoss(margin=margin)

    def forward(self, score_pos: torch.Tensor, score_neg: torch.Tensor) -> torch.Tensor:
        """
        Args:
            score_pos: scores for the highly viral articles (batch_size,)
            score_neg: scores for the less viral (hard negative) articles (batch_size,)
        """
        # The target is 1 meaning score_pos should be ranked higher than score_neg
        target = torch.ones_like(score_pos)
        return self.loss_fn(score_pos, score_neg, target)


class ListNetLoss(nn.Module):
    """
    Listwise ranking loss.
    Computes cross-entropy between the predicted score distribution and the true score distribution.
    Useful if you have continuous virality labels for a list of articles.
    """
    def __init__(self):
        super().__init__()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y_pred: predicted scores (batch_size, list_size)
            y_true: ground truth virality metric (batch_size, list_size)
        """
        # Apply softmax to convert scores to probability distributions
        P_y_pred = F.softmax(y_pred, dim=-1)
        P_y_true = F.softmax(y_true, dim=-1)
        
        # Cross entropy
        loss = -torch.sum(P_y_true * torch.log(P_y_pred + 1e-8), dim=-1)
        return torch.mean(loss)

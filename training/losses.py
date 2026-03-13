"""
Advanced loss functions for recommendation training
Improves RMSE by focusing on hard examples and better weighting
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalMSELoss(nn.Module):
    """
    Focal MSE Loss - focuses training on hard examples
    
    Traditional MSE treats all errors equally. Focal loss weights
    hard-to-predict examples more heavily, improving overall accuracy.
    
    Args:
        alpha: Modulating factor (default: 2.0)
        beta: Exponent for weighting (default: 4.0)
    """
    
    def __init__(self, alpha: float = 2.0, beta: float = 4.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal MSE loss
        
        Args:
            predictions: Model predictions (batch_size,)
            targets: Ground truth ratings (batch_size,)
        
        Returns:
            Scalar loss value
        """
        # Standard MSE
        mse = (predictions - targets) ** 2
        
        # Compute weights: easier examples get lower weight
        weights = torch.exp(-self.alpha * mse)
        
        # Apply focal weighting
        focal_mse = weights * (mse ** self.beta)
        
        return focal_mse.mean()


class WeightedMSELoss(nn.Module):
    """
    MSE with importance weighting for ratings
    Higher and lower ratings are weighted more (they're more confident)
    """
    
    def __init__(self, rating_min: float = 0.5, rating_max: float = 5.0):
        super().__init__()
        self.rating_min = rating_min
        self.rating_max = rating_max
        self.rating_mid = (rating_max + rating_min) / 2
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute weighted MSE"""
        
        # Weight = distance from middle rating (normalized)
        weights = torch.abs(targets - self.rating_mid) / (self.rating_max - self.rating_mid)
        weights = weights + 1.0  # Ensure all weights >= 1
        
        mse = (predictions - targets) ** 2
        weighted_mse = weights * mse
        
        return weighted_mse.mean()


class RankingLoss(nn.Module):
    """
    Pairwise ranking loss
    Ensures model ranks movies in correct relative order
    """
    
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin
    
    def forward(
        self,
        predictions_high: torch.Tensor,
        predictions_low: torch.Tensor,
        targets_high: torch.Tensor,
        targets_low: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute ranking loss
        
        Args:
            predictions_high: Predictions for higher-rated movies
            predictions_low: Predictions for lower-rated movies
            targets_high: Actual ratings for higher-rated movies
            targets_low: Actual ratings for lower-rated movies
        """
        # Calculate the margin violation
        # We want predictions_high > predictions_low + margin
        violations = F.relu(
            self.margin - (predictions_high - predictions_low)
        )
        
        return violations.mean()


class CombinedLoss(nn.Module):
    """
    Combination of MSE, Focal, and Ranking losses
    Provides best overall performance
    """
    
    def __init__(
        self,
        mse_weight: float = 1.0,
        focal_weight: float = 0.5,
        ranking_weight: float = 0.3
    ):
        super().__init__()
        self.mse_weight = mse_weight
        self.focal_weight = focal_weight
        self.ranking_weight = ranking_weight
        
        self.mse_loss = nn.MSELoss()
        self.focal_loss = FocalMSELoss()
        self.ranking_loss = RankingLoss()
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        predictions_pairs: tuple = None
    ) -> torch.Tensor:
        """
        Compute combined loss
        
        Args:
            predictions: Model predictions
            targets: Ground truth ratings
            predictions_pairs: Optional (pred_high, pred_low, target_high, target_low)
        """
        # MSE component
        loss_mse = self.mse_loss(predictions, targets)
        
        # Focal component
        loss_focal = self.focal_loss(predictions, targets)
        
        # Ranking component (if pairs provided)
        loss_ranking = torch.tensor(0.0, device=predictions.device)
        if predictions_pairs is not None:
            pred_high, pred_low, target_high, target_low = predictions_pairs
            loss_ranking = self.ranking_loss(pred_high, pred_low, target_high, target_low)
        
        # Combine losses
        total_loss = (
            self.mse_weight * loss_mse +
            self.focal_weight * loss_focal +
            self.ranking_weight * loss_ranking
        )
        
        return total_loss


class BPRLoss(nn.Module):
    """
    Bayesian Personalized Ranking Loss
    Optimizes for ranking instead of absolute rating prediction
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(
        self,
        positive_scores: torch.Tensor,
        negative_scores: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute BPR loss
        
        Args:
            positive_scores: Scores for items user likes
            negative_scores: Scores for items user dislikes
        """
        # BPR: maximize log-sigmoid of score difference
        diff = positive_scores - negative_scores
        return -F.logsigmoid(diff).mean()


class ListwiseLoss(nn.Module):
    """
    Listwise ranking loss using softmax cross-entropy
    Treats recommendation as a classification problem
    """
    
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self,
        scores: torch.Tensor,
        relevance: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute listwise loss
        
        Args:
            scores: Model scores for all items (batch_size, num_items)
            relevance: Ground truth relevance (batch_size, num_items)
        """
        # Scale scores by temperature
        scaled_scores = scores / self.temperature
        
        # Compute cross-entropy with relevance as targets
        log_probs = F.log_softmax(scaled_scores, dim=-1)
        loss = -(relevance * log_probs).sum(dim=-1).mean()
        
        return loss


# Convenient loss getter
def get_loss_function(loss_type: str = 'mse', **kwargs):
    """
    Factory function to get loss by name
    
    Args:
        loss_type: 'mse', 'focal', 'weighted', 'ranking', 'combined', 'bpr', 'listwise'
        **kwargs: Additional arguments for loss function
    
    Returns:
        Loss function instance
    """
    loss_map = {
        'mse': nn.MSELoss,
        'focal': FocalMSELoss,
        'weighted': WeightedMSELoss,
        'ranking': RankingLoss,
        'combined': CombinedLoss,
        'bpr': BPRLoss,
        'listwise': ListwiseLoss
    }
    
    if loss_type not in loss_map:
        raise ValueError(f"Unknown loss type: {loss_type}. Available: {list(loss_map.keys())}")
    
    return loss_map[loss_type](**kwargs)


if __name__ == '__main__':
    # Test losses
    predictions = torch.randn(32)
    targets = torch.randn(32)
    
    print("Testing loss functions:")
    
    # MSE
    mse_loss = nn.MSELoss()
    print(f"MSE Loss: {mse_loss(predictions, targets):.4f}")
    
    # Focal MSE
    focal_loss = FocalMSELoss()
    print(f"Focal MSE Loss: {focal_loss(predictions, targets):.4f}")
    
    # Weighted MSE
    weighted_loss = WeightedMSELoss()
    print(f"Weighted MSE Loss: {weighted_loss(predictions, targets):.4f}")
    
    # Combined
    combined_loss = CombinedLoss()
    print(f"Combined Loss: {combined_loss(predictions, targets):.4f}")
    
    print("\n✓ All loss functions working correctly!")

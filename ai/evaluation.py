"""
Evaluation Metrics for Recommendation Systems
Industry-standard metrics: RMSE, Hit@K, NDCG@K
"""

import torch
import numpy as np
from sklearn.metrics import mean_squared_error
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def rmse(predictions, targets):
    """
    Root Mean Squared Error
    Lower is better
    """
    mse = mean_squared_error(targets, predictions)
    return np.sqrt(mse)


def hit_rate_at_k(recommended_items, true_items, k=10):
    """
    Hit Rate @ K
    
    For each user, check if any recommended item is in true items
    
    Args:
        recommended_items: List[List[int]] - recommended items for each user
        true_items: List[List[int]] - ground truth items for each user
        k: int - cutoff
    
    Returns:
        hit_rate: float - proportion of users with at least one hit
    """
    hits = 0
    
    for rec, true in zip(recommended_items, true_items):
        # Take top-k recommendations
        top_k_rec = rec[:k]
        
        # Check if any recommended item is in true items
        if any(item in true for item in top_k_rec):
            hits += 1
    
    return hits / len(recommended_items) if recommended_items else 0.0


def ndcg_at_k(recommended_items, true_items, k=10):
    """
    Normalized Discounted Cumulative Gain @ K
    
    Measures ranking quality - higher is better
    Gives more weight to items ranked higher
    
    Args:
        recommended_items: List[List[int]] - recommended items for each user
        true_items: List[List[int]] - ground truth items for each user
        k: int - cutoff
    
    Returns:
        ndcg: float - average NDCG across all users
    """
    ndcgs = []
    
    for rec, true in zip(recommended_items, true_items):
        # Take top-k recommendations
        top_k_rec = rec[:k]
        
        # Calculate DCG
        dcg = 0.0
        for i, item in enumerate(top_k_rec):
            if item in true:
                # Relevance = 1 if item is relevant, 0 otherwise
                relevance = 1
                # Position discount: 1/log2(i+2)
                dcg += relevance / np.log2(i + 2)
        
        # Calculate IDCG (ideal DCG)
        # Ideal ranking: all relevant items first
        num_relevant = min(len(true), k)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(num_relevant))
        
        # NDCG
        if idcg > 0:
            ndcgs.append(dcg / idcg)
        else:
            ndcgs.append(0.0)
    
    return np.mean(ndcgs) if ndcgs else 0.0


def precision_at_k(recommended_items, true_items, k=10):
    """
    Precision @ K
    
    What proportion of recommendations are relevant?
    
    Args:
        recommended_items: List[List[int]] - recommended items for each user
        true_items: List[List[int]] - ground truth items for each user
        k: int - cutoff
    
    Returns:
        precision: float
    """
    precisions = []
    
    for rec, true in zip(recommended_items, true_items):
        top_k_rec = rec[:k]
        
        # Count hits
        hits = sum(1 for item in top_k_rec if item in true)
        
        # Precision = hits / k
        precisions.append(hits / k if k > 0 else 0.0)
    
    return np.mean(precisions) if precisions else 0.0


def recall_at_k(recommended_items, true_items, k=10):
    """
    Recall @ K
    
    What proportion of relevant items did we recommend?
    
    Args:
        recommended_items: List[List[int]] - recommended items for each user
        true_items: List[List[int]] - ground truth items for each user
        k: int - cutoff
    
    Returns:
        recall: float
    """
    recalls = []
    
    for rec, true in zip(recommended_items, true_items):
        top_k_rec = rec[:k]
        
        if len(true) == 0:
            continue
        
        # Count hits
        hits = sum(1 for item in top_k_rec if item in true)
        
        # Recall = hits / total_relevant
        recalls.append(hits / len(true))
    
    return np.mean(recalls) if recalls else 0.0


def map_at_k(recommended_items, true_items, k=10):
    """
    Mean Average Precision @ K
    
    Considers both precision and position
    
    Args:
        recommended_items: List[List[int]] - recommended items for each user
        true_items: List[List[int]] - ground truth items for each user
        k: int - cutoff
    
    Returns:
        map_score: float
    """
    aps = []
    
    for rec, true in zip(recommended_items, true_items):
        top_k_rec = rec[:k]
        
        if len(true) == 0:
            continue
        
        # Calculate AP
        hits = 0
        precision_sum = 0.0
        
        for i, item in enumerate(top_k_rec):
            if item in true:
                hits += 1
                precision_at_i = hits / (i + 1)
                precision_sum += precision_at_i
        
        # AP = average of precisions at hit positions
        ap = precision_sum / min(len(true), k) if hits > 0 else 0.0
        aps.append(ap)
    
    return np.mean(aps) if aps else 0.0


class RecommenderEvaluator:
    """
    Comprehensive evaluator for recommendation models
    """
    
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
    
    def evaluate_ranking(self, test_loader, k_values=[5, 10, 20]):
        """
        Evaluate ranking metrics: Hit@K, NDCG@K, Precision@K, Recall@K
        
        Args:
            test_loader: DataLoader for test set
            k_values: List[int] - K values to evaluate
        
        Returns:
            metrics: dict
        """
        self.model.eval()
        
        logger.info("Evaluating ranking metrics...")
        
        # Collect predictions by user
        user_predictions = {}  # user_id -> [(movie_id, score)]
        user_ground_truth = {}  # user_id -> [movie_id]
        
        with torch.no_grad():
            for batch in test_loader:
                user_ids = batch['user_id'].to(self.device)
                movie_ids = batch['movie_id'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Get predictions
                scores = self.model(user_ids, movie_ids)
                
                # Group by user
                for i in range(len(user_ids)):
                    uid = user_ids[i].item()
                    mid = movie_ids[i].item()
                    score = scores[i].item()
                    label = labels[i].item()
                    
                    if uid not in user_predictions:
                        user_predictions[uid] = []
                        user_ground_truth[uid] = []
                    
                    user_predictions[uid].append((mid, score))
                    
                    # If label > 0.5 (positive), add to ground truth
                    if label > 0.5:
                        user_ground_truth[uid].append(mid)
        
        # Sort predictions by score for each user
        recommended_items = []
        true_items = []
        
        for uid in user_predictions.keys():
            # Sort by score descending
            sorted_preds = sorted(user_predictions[uid], key=lambda x: x[1], reverse=True)
            recommended = [mid for mid, score in sorted_preds]
            
            recommended_items.append(recommended)
            true_items.append(user_ground_truth[uid])
        
        # Calculate metrics for each K
        metrics = {}
        
        for k in k_values:
            metrics[f'Hit@{k}'] = hit_rate_at_k(recommended_items, true_items, k)
            metrics[f'NDCG@{k}'] = ndcg_at_k(recommended_items, true_items, k)
            metrics[f'Precision@{k}'] = precision_at_k(recommended_items, true_items, k)
            metrics[f'Recall@{k}'] = recall_at_k(recommended_items, true_items, k)
            metrics[f'MAP@{k}'] = map_at_k(recommended_items, true_items, k)
        
        return metrics
    
    def evaluate_rating_prediction(self, test_loader):
        """
        Evaluate rating prediction: RMSE, MAE
        
        Args:
            test_loader: DataLoader for test set
        
        Returns:
            metrics: dict
        """
        self.model.eval()
        
        logger.info("Evaluating rating prediction...")
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in test_loader:
                user_ids = batch['user_id'].to(self.device)
                movie_ids = batch['movie_id'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Get predictions
                scores = self.model(user_ids, movie_ids)
                
                all_predictions.extend(scores.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())
        
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        
        # Calculate metrics
        rmse_score = rmse(all_predictions, all_targets)
        mae_score = np.mean(np.abs(all_predictions - all_targets))
        
        return {
            'RMSE': rmse_score,
            'MAE': mae_score
        }
    
    def evaluate_full(self, test_loader, k_values=[5, 10, 20]):
        """
        Full evaluation: rating prediction + ranking
        
        Returns:
            metrics: dict
        """
        rating_metrics = self.evaluate_rating_prediction(test_loader)
        ranking_metrics = self.evaluate_ranking(test_loader, k_values)
        
        # Combine
        metrics = {**rating_metrics, **ranking_metrics}
        
        # Log results
        logger.info("="*60)
        logger.info("EVALUATION RESULTS")
        logger.info("="*60)
        
        logger.info("\nRating Prediction:")
        logger.info(f"  RMSE: {metrics['RMSE']:.4f}")
        logger.info(f"  MAE:  {metrics['MAE']:.4f}")
        
        logger.info("\nRanking Metrics:")
        for k in k_values:
            logger.info(f"  K={k}:")
            logger.info(f"    Hit@{k}:       {metrics[f'Hit@{k}']:.4f}")
            logger.info(f"    NDCG@{k}:      {metrics[f'NDCG@{k}']:.4f}")
            logger.info(f"    Precision@{k}: {metrics[f'Precision@{k}']:.4f}")
            logger.info(f"    Recall@{k}:    {metrics[f'Recall@{k}']:.4f}")
        
        logger.info("="*60)
        
        return metrics


if __name__ == '__main__':
    # Test metrics
    logger.info("Testing evaluation metrics...")
    
    # Sample data
    recommended = [[1, 2, 3, 4, 5], [10, 11, 12, 13, 14]]
    true = [[2, 5, 6], [10, 15]]
    
    hit = hit_rate_at_k(recommended, true, k=5)
    ndcg = ndcg_at_k(recommended, true, k=5)
    precision = precision_at_k(recommended, true, k=5)
    recall = recall_at_k(recommended, true, k=5)
    
    logger.info(f"Hit@5: {hit:.4f}")
    logger.info(f"NDCG@5: {ndcg:.4f}")
    logger.info(f"Precision@5: {precision:.4f}")
    logger.info(f"Recall@5: {recall:.4f}")
    
    logger.info("✓ Metrics test passed!")

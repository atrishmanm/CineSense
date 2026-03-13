"""
A/B Testing Framework for Recommendation Systems
Conduct experiments, track metrics, and analyze performance
"""

import hashlib
import random
import time
import json
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import logging
from scipy import stats
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Experiment:
    """Experiment configuration"""
    experiment_id: str
    name: str
    description: str
    variants: List[str]  # e.g., ['control', 'treatment_a', 'treatment_b']
    traffic_allocation: Dict[str, float]  # e.g., {'control': 0.5, 'treatment_a': 0.5}
    start_date: str
    end_date: str
    metrics: List[str]  # Metrics to track
    is_active: bool = True
    min_sample_size: int = 1000  # Minimum samples before statistical test
    
    def __post_init__(self):
        """Validate experiment configuration"""
        # Check traffic allocation sums to 1.0
        total = sum(self.traffic_allocation.values())
        if not (0.99 <= total <= 1.01):
            raise ValueError(f"Traffic allocation must sum to 1.0, got {total}")
        
        # Check all variants have allocation
        for variant in self.variants:
            if variant not in self.traffic_allocation:
                raise ValueError(f"Variant {variant} missing traffic allocation")


@dataclass
class ExperimentMetrics:
    """Metrics for an experiment variant"""
    variant: str
    experiment_id: str
    
    # User engagement
    impressions: int = 0
    clicks: int = 0
    conversions: int = 0
    
    # Recommendation quality
    ratings_given: int = 0
    avg_rating: float = 0.0
    positive_ratings: int = 0  # 4+ stars
    
    # Business metrics
    watch_time_minutes: float = 0.0
    user_retention: int = 0  # Users who return
    
    # Lists for statistical tests
    rating_values: List[float] = None
    watch_times: List[float] = None
    
    def __post_init__(self):
        if self.rating_values is None:
            self.rating_values = []
        if self.watch_times is None:
            self.watch_times = []
    
    @property
    def ctr(self) -> float:
        """Click-through rate"""
        return self.clicks / self.impressions if self.impressions > 0 else 0.0
    
    @property
    def conversion_rate(self) -> float:
        """Conversion rate"""
        return self.conversions / self.clicks if self.clicks > 0 else 0.0
    
    @property
    def positive_rating_rate(self) -> float:
        """Rate of positive ratings"""
        return self.positive_ratings / self.ratings_given if self.ratings_given > 0 else 0.0


class ABTestingFramework:
    """
    A/B testing system with:
    - Consistent user bucketing
    - Metric tracking
    - Statistical significance testing
    - Experiment management
    """
    
    def __init__(self, db_manager=None):
        """
        Initialize A/B testing framework
        
        Args:
            db_manager: Optional database manager for persistent storage
        """
        self.db_manager = db_manager
        self.experiments: Dict[str, Experiment] = {}
        self.metrics: Dict[str, Dict[str, ExperimentMetrics]] = {}
        
        # For demo: in-memory storage
        self.user_assignments: Dict[int, Dict[str, str]] = {}
    
    def create_experiment(
        self,
        experiment_id: str,
        name: str,
        variants: List[str],
        traffic_allocation: Dict[str, float],
        duration_days: int = 14,
        description: str = "",
        metrics: List[str] = None
    ) -> Experiment:
        """
        Create a new A/B test experiment
        
        Args:
            experiment_id: Unique ID
            name: Experiment name
            variants: List of variant names
            traffic_allocation: Traffic % for each variant
            duration_days: Experiment duration
            description: Description
            metrics: Metrics to track
            
        Returns:
            Experiment configuration
        """
        start_date = datetime.now()
        end_date = start_date + timedelta(days=duration_days)
        
        if metrics is None:
            metrics = ['ctr', 'conversion_rate', 'avg_rating', 'watch_time']
        
        experiment = Experiment(
            experiment_id=experiment_id,
            name=name,
            description=description,
            variants=variants,
            traffic_allocation=traffic_allocation,
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
            metrics=metrics
        )
        
        self.experiments[experiment_id] = experiment
        
        # Initialize metrics for each variant
        self.metrics[experiment_id] = {
            variant: ExperimentMetrics(
                variant=variant,
                experiment_id=experiment_id
            )
            for variant in variants
        }
        
        logger.info(f"✓ Created experiment: {name} ({experiment_id})")
        logger.info(f"  Variants: {variants}")
        logger.info(f"  Allocation: {traffic_allocation}")
        
        return experiment
    
    def assign_variant(self, user_id: int, experiment_id: str) -> str:
        """
        Consistently assign user to a variant
        
        Uses deterministic hashing for consistency across sessions
        
        Args:
            user_id: User ID
            experiment_id: Experiment ID
            
        Returns:
            Assigned variant name
        """
        # Check if already assigned
        if user_id in self.user_assignments:
            if experiment_id in self.user_assignments[user_id]:
                return self.user_assignments[user_id][experiment_id]
        
        experiment = self.experiments.get(experiment_id)
        if not experiment or not experiment.is_active:
            return 'control'  # Default variant
        
        # Deterministic hash
        hash_input = f"{user_id}:{experiment_id}".encode()
        hash_value = int(hashlib.md5(hash_input).hexdigest(), 16)
        bucket = (hash_value % 10000) / 10000.0  # 0.0 to 1.0
        
        # Assign to variant based on traffic allocation
        cumulative = 0.0
        for variant, allocation in experiment.traffic_allocation.items():
            cumulative += allocation
            if bucket < cumulative:
                assigned_variant = variant
                break
        else:
            assigned_variant = experiment.variants[0]
        
        # Store assignment
        if user_id not in self.user_assignments:
            self.user_assignments[user_id] = {}
        self.user_assignments[user_id][experiment_id] = assigned_variant
        
        return assigned_variant
    
    def track_impression(self, user_id: int, experiment_id: str):
        """Track recommendation impression"""
        variant = self.assign_variant(user_id, experiment_id)
        if experiment_id in self.metrics:
            self.metrics[experiment_id][variant].impressions += 1
    
    def track_click(self, user_id: int, experiment_id: str):
        """Track recommendation click"""
        variant = self.assign_variant(user_id, experiment_id)
        if experiment_id in self.metrics:
            self.metrics[experiment_id][variant].clicks += 1
    
    def track_conversion(self, user_id: int, experiment_id: str):
        """Track conversion (e.g., watched movie)"""
        variant = self.assign_variant(user_id, experiment_id)
        if experiment_id in self.metrics:
            self.metrics[experiment_id][variant].conversions += 1
    
    def track_rating(self, user_id: int, experiment_id: str, rating: float):
        """Track user rating"""
        variant = self.assign_variant(user_id, experiment_id)
        if experiment_id in self.metrics:
            metrics = self.metrics[experiment_id][variant]
            metrics.ratings_given += 1
            
            # Update average
            n = metrics.ratings_given
            metrics.avg_rating = (
                (metrics.avg_rating * (n - 1) + rating) / n
            )
            
            # Track positive ratings
            if rating >= 4.0:
                metrics.positive_ratings += 1
            
            # Store for statistical test
            metrics.rating_values.append(rating)
    
    def track_watch_time(self, user_id: int, experiment_id: str, minutes: float):
        """Track watch time"""
        variant = self.assign_variant(user_id, experiment_id)
        if experiment_id in self.metrics:
            metrics = self.metrics[experiment_id][variant]
            metrics.watch_time_minutes += minutes
            metrics.watch_times.append(minutes)
    
    def get_experiment_results(self, experiment_id: str) -> Dict[str, Any]:
        """
        Get experiment results with statistical significance
        
        Args:
            experiment_id: Experiment ID
            
        Returns:
            Results dictionary with metrics and p-values
        """
        if experiment_id not in self.experiments:
            return {'error': 'Experiment not found'}
        
        experiment = self.experiments[experiment_id]
        variant_metrics = self.metrics[experiment_id]
        
        results = {
            'experiment': asdict(experiment),
            'variants': {},
            'statistical_tests': {},
            'recommendation': None
        }
        
        # Collect metrics for each variant
        for variant, metrics in variant_metrics.items():
            results['variants'][variant] = {
                'impressions': metrics.impressions,
                'clicks': metrics.clicks,
                'ctr': round(metrics.ctr * 100, 2),
                'conversions': metrics.conversions,
                'conversion_rate': round(metrics.conversion_rate * 100, 2),
                'ratings_given': metrics.ratings_given,
                'avg_rating': round(metrics.avg_rating, 2),
                'positive_rating_rate': round(metrics.positive_rating_rate * 100, 2),
                'total_watch_time': round(metrics.watch_time_minutes, 1)
            }
        
        # Statistical significance tests
        control = variant_metrics.get('control')
        if control:
            for variant, metrics in variant_metrics.items():
                if variant == 'control':
                    continue
                
                tests = {}
                
                # Test CTR (chi-square test)
                if control.impressions >= experiment.min_sample_size:
                    contingency = [
                        [control.clicks, control.impressions - control.clicks],
                        [metrics.clicks, metrics.impressions - metrics.clicks]
                    ]
                    try:
                        chi2, p_value = stats.chi2_contingency(contingency)[:2]
                        tests['ctr_p_value'] = round(p_value, 4)
                        tests['ctr_significant'] = p_value < 0.05
                    except:
                        tests['ctr_p_value'] = None
                
                # Test average rating (t-test)
                if len(control.rating_values) >= 30 and len(metrics.rating_values) >= 30:
                    t_stat, p_value = stats.ttest_ind(
                        control.rating_values,
                        metrics.rating_values,
                        equal_var=False
                    )
                    tests['rating_p_value'] = round(p_value, 4)
                    tests['rating_significant'] = p_value < 0.05
                
                # Test watch time (t-test)
                if len(control.watch_times) >= 30 and len(metrics.watch_times) >= 30:
                    t_stat, p_value = stats.ttest_ind(
                        control.watch_times,
                        metrics.watch_times,
                        equal_var=False
                    )
                    tests['watch_time_p_value'] = round(p_value, 4)
                    tests['watch_time_significant'] = p_value < 0.05
                
                results['statistical_tests'][variant] = tests
        
        # Make recommendation
        results['recommendation'] = self._make_recommendation(results)
        
        return results
    
    def _make_recommendation(self, results: Dict) -> str:
        """Make recommendation based on results"""
        control_variant = results['variants'].get('control')
        if not control_variant:
            return "No control variant found"
        
        # Find best performing variant
        best_variant = 'control'
        best_ctr = control_variant['ctr']
        is_significant = False
        
        for variant, metrics in results['variants'].items():
            if variant == 'control':
                continue
            
            # Check if better and significant
            if metrics['ctr'] > best_ctr:
                tests = results['statistical_tests'].get(variant, {})
                if tests.get('ctr_significant', False):
                    best_variant = variant
                    best_ctr = metrics['ctr']
                    is_significant = True
        
        if best_variant == 'control':
            return "Keep control variant (no significant improvements)"
        elif is_significant:
            return f"✓ Deploy {best_variant} (statistically significant improvement)"
        else:
            return f"Consider {best_variant} but not yet significant (continue testing)"
    
    def stop_experiment(self, experiment_id: str):
        """Stop an experiment"""
        if experiment_id in self.experiments:
            self.experiments[experiment_id].is_active = False
            logger.info(f"✓ Stopped experiment: {experiment_id}")


# Example usage
if __name__ == '__main__':
    print("A/B Testing Framework Demo")
    print("=" * 60)
    
    # Create framework
    ab_test = ABTestingFramework()
    
    # Create experiment: Test new recommendation algorithm
    experiment = ab_test.create_experiment(
        experiment_id='rec_algo_v2',
        name='New Recommendation Algorithm',
        variants=['control', 'treatment'],
        traffic_allocation={'control': 0.5, 'treatment': 0.5},
        duration_days=14,
        description='Test transformer-based recommendations vs baseline'
    )
    
    print(f"\n✓ Created experiment: {experiment.name}")
    
    # Simulate user interactions
    print("\nSimulating 10,000 user interactions...")
    random.seed(42)
    
    for user_id in range(1, 10001):
        # Track impression
        ab_test.track_impression(user_id, 'rec_algo_v2')
        
        # 30% click rate (treatment slightly better)
        variant = ab_test.assign_variant(user_id, 'rec_algo_v2')
        click_prob = 0.28 if variant == 'control' else 0.32
        
        if random.random() < click_prob:
            ab_test.track_click(user_id, 'rec_algo_v2')
            
            # 40% conversion rate
            if random.random() < 0.4:
                ab_test.track_conversion(user_id, 'rec_algo_v2')
                
                # Give rating (treatment gets slightly higher ratings)
                if variant == 'control':
                    rating = random.gauss(3.5, 0.8)
                else:
                    rating = random.gauss(3.8, 0.7)
                rating = max(1.0, min(5.0, rating))
                ab_test.track_rating(user_id, 'rec_algo_v2', rating)
                
                # Watch time
                watch_time = random.expovariate(1/45)  # ~45 min average
                ab_test.track_watch_time(user_id, 'rec_algo_v2', watch_time)
    
    # Get results
    print("\n" + "=" * 60)
    print("EXPERIMENT RESULTS")
    print("=" * 60)
    
    results = ab_test.get_experiment_results('rec_algo_v2')
    
    for variant, metrics in results['variants'].items():
        print(f"\n{variant.upper()}:")
        print(f"  Impressions: {metrics['impressions']:,}")
        print(f"  CTR: {metrics['ctr']}%")
        print(f"  Conversions: {metrics['conversions']:,}")
        print(f"  Conversion Rate: {metrics['conversion_rate']}%")
        print(f"  Avg Rating: {metrics['avg_rating']}")
        print(f"  Positive Rating Rate: {metrics['positive_rating_rate']}%")
        print(f"  Total Watch Time: {metrics['total_watch_time']} min")
    
    print("\n" + "=" * 60)
    print("STATISTICAL SIGNIFICANCE")
    print("=" * 60)
    
    for variant, tests in results['statistical_tests'].items():
        print(f"\n{variant.upper()} vs CONTROL:")
        for test_name, value in tests.items():
            print(f"  {test_name}: {value}")
    
    print("\n" + "=" * 60)
    print("RECOMMENDATION")
    print("=" * 60)
    print(f"\n{results['recommendation']}")
    
    print("\n✓ A/B testing framework operational")

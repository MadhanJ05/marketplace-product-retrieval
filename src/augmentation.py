"""
Wholesale Query Augmentation Module

Transforms standard product queries into wholesale/B2B style queries
to simulate retailer search behavior on platforms like Faire.
"""

import random
from typing import List, Dict, Optional
import yaml


class WholesaleQueryAugmenter:
    """
    Augments product queries with wholesale/B2B signals.
    
    Example:
        >>> augmenter = WholesaleQueryAugmenter()
        >>> augmenter.augment("running shoes nike")
        ['running shoes nike bulk', 'wholesale running shoes nike']
    """
    
    # Default templates if config not provided
    DEFAULT_WHOLESALE_TEMPLATES = [
        "{query} bulk",
        "{query} wholesale",
        "{query} bulk pack",
        "{query} case pack",
        "{query} wholesale lot",
        "bulk {query}",
        "wholesale {query}",
        "{query} for resale",
        "{query} retail pack",
        "{query} multi pack",
    ]
    
    DEFAULT_QUANTITY_TEMPLATES = [
        "{query} set of 6",
        "{query} set of 12",
        "{query} pack of 24",
        "{query} 50 count",
        "{query} 100 pack",
    ]
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        wholesale_templates: Optional[List[str]] = None,
        quantity_templates: Optional[List[str]] = None,
        quantity_probability: float = 0.3,
        seed: int = 42
    ):
        """
        Initialize the augmenter.
        
        Args:
            config_path: Path to config.yaml (optional)
            wholesale_templates: List of wholesale query templates
            quantity_templates: List of quantity signal templates
            quantity_probability: Probability of adding quantity signal
            seed: Random seed for reproducibility
        """
        random.seed(seed)
        
        # Load from config if provided
        if config_path:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            aug_config = config.get('augmentation', {})
            
            self.wholesale_templates = aug_config.get('templates', {}).get(
                'wholesale', self.DEFAULT_WHOLESALE_TEMPLATES
            )
            self.quantity_templates = aug_config.get('templates', {}).get(
                'quantity', self.DEFAULT_QUANTITY_TEMPLATES
            )
            self.quantity_probability = aug_config.get(
                'quantity_probability', quantity_probability
            )
        else:
            self.wholesale_templates = wholesale_templates or self.DEFAULT_WHOLESALE_TEMPLATES
            self.quantity_templates = quantity_templates or self.DEFAULT_QUANTITY_TEMPLATES
            self.quantity_probability = quantity_probability
    
    def augment(
        self,
        query: str,
        n_augmentations: int = 2,
        include_quantity: bool = True
    ) -> List[str]:
        """
        Create wholesale-style variations of a query.
        
        Args:
            query: Original query string
            n_augmentations: Number of wholesale variations to create
            include_quantity: Whether to potentially add quantity signals
        
        Returns:
            List of augmented queries
        """
        augmented = []
        
        # Select random wholesale templates
        n_templates = min(n_augmentations, len(self.wholesale_templates))
        selected_templates = random.sample(self.wholesale_templates, n_templates)
        
        for template in selected_templates:
            aug_query = template.format(query=query)
            augmented.append(aug_query)
        
        # Optionally add quantity signal
        if include_quantity and random.random() < self.quantity_probability:
            qty_template = random.choice(self.quantity_templates)
            augmented.append(qty_template.format(query=query))
        
        return augmented
    
    def augment_batch(
        self,
        queries: List[str],
        n_augmentations: int = 2,
        augment_ratio: float = 0.25
    ) -> Dict[str, List[str]]:
        """
        Augment a batch of queries.
        
        Args:
            queries: List of original queries
            n_augmentations: Augmentations per query
            augment_ratio: Fraction of queries to augment
        
        Returns:
            Dictionary mapping original queries to their augmentations
        """
        n_to_augment = int(len(queries) * augment_ratio)
        selected_queries = random.sample(queries, n_to_augment)
        
        augmented_map = {}
        for query in selected_queries:
            augmented_map[query] = self.augment(query, n_augmentations)
        
        return augmented_map
    
    def get_augmentation_stats(self, augmented_map: Dict[str, List[str]]) -> Dict:
        """
        Get statistics about augmented queries.
        
        Args:
            augmented_map: Output from augment_batch()
        
        Returns:
            Dictionary of statistics
        """
        total_augmented = sum(len(v) for v in augmented_map.values())
        
        # Count template usage
        template_counts = {t: 0 for t in self.wholesale_templates + self.quantity_templates}
        
        for augs in augmented_map.values():
            for aug in augs:
                for template in self.wholesale_templates + self.quantity_templates:
                    # Check if this template was likely used
                    if 'bulk' in template and 'bulk' in aug:
                        template_counts[template] = template_counts.get(template, 0) + 1
                        break
        
        return {
            'n_original_queries': len(augmented_map),
            'n_augmented_queries': total_augmented,
            'avg_augmentations_per_query': total_augmented / len(augmented_map) if augmented_map else 0,
        }


# Convenience function for quick augmentation
def augment_query(query: str, n: int = 2) -> List[str]:
    """Quick function to augment a single query."""
    augmenter = WholesaleQueryAugmenter()
    return augmenter.augment(query, n)


if __name__ == "__main__":
    # Test the augmenter
    augmenter = WholesaleQueryAugmenter()
    
    test_queries = [
        "running shoes nike",
        "ceramic mugs blue",
        "organic cotton t-shirts",
        "boston celtics cap",
        "scented candles lavender"
    ]
    
    print("Wholesale Query Augmentation Examples")
    print("=" * 50)
    
    for query in test_queries:
        augmented = augmenter.augment(query, n_augmentations=3)
        print(f"\nOriginal: '{query}'")
        for aug in augmented:
            print(f"  → '{aug}'")
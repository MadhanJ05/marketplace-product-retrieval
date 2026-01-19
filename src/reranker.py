"""
Reranking Module
Uses XGBoost LambdaMART-style ranker to reorder search results.
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb


class FeatureExtractor:
    """Extract features for reranking."""
    
    def __init__(self, product_df: pd.DataFrame):
        """
        Args:
            product_df: DataFrame with product information
        """
        self.product_lookup = product_df.set_index('product_id').to_dict('index')
        
    def extract_features(
        self,
        query: str,
        results: List[Dict],
        query_analysis: Optional[Dict] = None
    ) -> np.ndarray:
        """
        Extract features for each query-product pair.
        
        Features:
        1. BM25 score (normalized)
        2. FAISS score (normalized)
        3. Source flags (bm25_only, faiss_only, both)
        4. Rank from BM25
        5. Rank from FAISS
        6. Title length
        7. Query-title word overlap
        8. Has brand match
        9. Intent match (wholesale signal in product)
        
        Returns:
            np.ndarray of shape (n_results, n_features)
        """
        features = []
        
        query_tokens = set(query.lower().split())
        is_wholesale_query = any(w in query.lower() for w in ['bulk', 'wholesale', 'pack', 'case'])
        
        for rank, result in enumerate(results):
            product = self.product_lookup.get(result['product_id'], {})
            product_title = str(product.get('product_title', '')).lower()
            product_brand = str(product.get('product_brand', '')).lower()
            
            title_tokens = set(product_title.split())
            
            # Feature extraction
            feat = []
            
            # 1. Hybrid score
            feat.append(result.get('score', 0))
            
            # 2. Rank position (normalized)
            feat.append(1.0 / (rank + 1))
            
            # 3-4. Source flags
            sources = result.get('sources', [])
            feat.append(1.0 if 'bm25' in sources else 0.0)
            feat.append(1.0 if 'faiss' in sources else 0.0)
            
            # 5. Both sources (stronger signal)
            feat.append(1.0 if ('bm25' in sources and 'faiss' in sources) else 0.0)
            
            # 6. Title length (normalized)
            feat.append(min(len(product_title) / 200.0, 1.0))
            
            # 7. Query-title word overlap (Jaccard-ish)
            overlap = len(query_tokens & title_tokens)
            feat.append(overlap / max(len(query_tokens), 1))
            
            # 8. Query coverage (what % of query words are in title)
            coverage = overlap / max(len(query_tokens), 1)
            feat.append(coverage)
            
            # 9. Brand in query
            brand_match = 1.0 if product_brand and product_brand in query.lower() else 0.0
            feat.append(brand_match)
            
            # 10. Wholesale signal match
            product_has_bulk = any(w in product_title for w in ['bulk', 'pack', 'set of', 'wholesale', 'case'])
            wholesale_match = 1.0 if (is_wholesale_query and product_has_bulk) else 0.0
            feat.append(wholesale_match)
            
            # 11. Price indicator (has $ or price-like patterns)
            has_price = 1.0 if '$' in product_title or 'oz' in product_title else 0.0
            feat.append(has_price)
            
            # 12. Product completeness (has description, brand, etc.)
            completeness = sum([
                1 if product.get('product_brand') else 0,
                1 if product.get('product_description') else 0,
                1 if product.get('product_bullet_point') else 0,
            ]) / 3.0
            feat.append(completeness)
            
            features.append(feat)
        
        return np.array(features, dtype=np.float32)
    
    @property
    def feature_names(self) -> List[str]:
        return [
            'hybrid_score',
            'rank_score',
            'has_bm25',
            'has_faiss',
            'has_both',
            'title_length',
            'word_overlap',
            'query_coverage',
            'brand_match',
            'wholesale_match',
            'has_price_indicator',
            'product_completeness'
        ]


class Reranker:
    """XGBoost-based reranker using LambdaMART objective."""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        product_df: Optional[pd.DataFrame] = None
    ):
        """
        Args:
            model_path: Path to saved model (optional)
            product_df: Product DataFrame for feature extraction
        """
        self.model = None
        self.feature_extractor = None
        
        if product_df is not None:
            self.feature_extractor = FeatureExtractor(product_df)
        
        if model_path and Path(model_path).exists():
            self.load(model_path)
    
    def train(
        self,
        train_data: List[Dict],
        val_data: Optional[List[Dict]] = None,
        params: Optional[Dict] = None
    ):
        """
        Train the reranker.
        
        Args:
            train_data: List of {'query', 'results', 'labels', 'query_analysis'}
            val_data: Optional validation data in same format
            params: XGBoost parameters
        """
        if params is None:
            params = {
                'objective': 'rank:ndcg',
                'learning_rate': 0.1,
                'max_depth': 6,
                'n_estimators': 100,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'n_jobs': -1
            }
        
        # Prepare training data
        X_train, y_train, groups_train = self._prepare_data(train_data)
        
        print(f"Training data: {X_train.shape[0]} samples, {len(groups_train)} queries")
        
        # Create model
        self.model = xgb.XGBRanker(**params)
        
        # Train
        if val_data:
            X_val, y_val, groups_val = self._prepare_data(val_data)
            self.model.fit(
                X_train, y_train,
                group=groups_train,
                eval_set=[(X_val, y_val)],
                eval_group=[groups_val],
                verbose=True
            )
        else:
            self.model.fit(X_train, y_train, group=groups_train, verbose=True)
        
        print("✓ Reranker trained")
        
        # Feature importance
        self._print_feature_importance()
    
    def _prepare_data(
        self,
        data: List[Dict]
    ) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """Prepare data for XGBoost ranker."""
        X_list = []
        y_list = []
        groups = []
        
        for item in data:
            query = item['query']
            results = item['results']
            labels = item['labels']
            query_analysis = item.get('query_analysis')
            
            if len(results) == 0:
                continue
            
            # Extract features
            features = self.feature_extractor.extract_features(
                query, results, query_analysis
            )
            
            X_list.append(features)
            y_list.extend(labels)
            groups.append(len(results))
        
        X = np.vstack(X_list)
        y = np.array(y_list, dtype=np.float32)
        
        return X, y, groups
    
    def _print_feature_importance(self):
        """Print feature importance."""
        if self.model is None:
            return
        
        importance = self.model.feature_importances_
        names = self.feature_extractor.feature_names
        
        print("\nFeature Importance:")
        print("-" * 40)
        for name, imp in sorted(zip(names, importance), key=lambda x: -x[1]):
            print(f"  {name}: {imp:.4f}")
    
    def rerank(
        self,
        query: str,
        results: List[Dict],
        query_analysis: Optional[Dict] = None,
        top_k: Optional[int] = None
    ) -> List[Dict]:
        """
        Rerank search results.
        
        Args:
            query: Search query
            results: List of search results from hybrid retriever
            query_analysis: Optional query analysis from QueryUnderstanding
            top_k: Number of results to return (default: all)
        
        Returns:
            Reranked results with updated scores
        """
        if len(results) == 0:
            return results
        
        if self.model is None:
            # No model trained, return original order
            return results[:top_k] if top_k else results
        
        # Extract features
        features = self.feature_extractor.extract_features(
            query, results, query_analysis
        )
        
        # Predict scores
        scores = self.model.predict(features)
        
        # Sort by predicted score
        sorted_indices = np.argsort(scores)[::-1]
        
        reranked = []
        for new_rank, old_idx in enumerate(sorted_indices):
            result = results[old_idx].copy()
            result['rerank_score'] = float(scores[old_idx])
            result['original_rank'] = old_idx + 1
            result['new_rank'] = new_rank + 1
            reranked.append(result)
        
        return reranked[:top_k] if top_k else reranked
    
    def save(self, path: str):
        """Save model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'feature_names': self.feature_extractor.feature_names if self.feature_extractor else None
            }, f)
        print(f"✓ Model saved to {path}")
    
    def load(self, path: str):
        """Load model from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.model = data['model']
        print(f"✓ Model loaded from {path}")


def create_training_data_from_labels(
    labels_df: pd.DataFrame,
    retriever,
    query_understanding=None,
    max_queries: int = 1000,
    candidates_per_query: int = 50
) -> List[Dict]:
    """
    Create training data from labeled query-product pairs.
    
    Args:
        labels_df: DataFrame with 'query', 'product_id', 'label' columns
                   Labels: E=3 (Exact), S=2 (Substitute), C=1 (Complement), I=0 (Irrelevant)
        retriever: HybridRetriever instance
        query_understanding: QueryUnderstanding instance (optional)
        max_queries: Maximum number of queries to use
        candidates_per_query: Number of candidates to retrieve per query
    
    Returns:
        List of training examples
    """
    # Label mapping
    label_map = {'E': 3, 'S': 2, 'C': 1, 'I': 0}
    
    # Get unique queries
    queries = labels_df['query'].unique()[:max_queries]
    
    training_data = []
    
    for query in queries:
        # Get labeled products for this query
        query_labels = labels_df[labels_df['query'] == query]
        label_dict = dict(zip(query_labels['product_id'], query_labels['esci_label']))
        
        # Get query analysis if available
        query_analysis = None
        if query_understanding:
            query_analysis = query_understanding.analyze(query)
        
        # Retrieve candidates
        results = retriever.search(query, method="hybrid", top_k=candidates_per_query)
        
        if len(results) == 0:
            continue
        
        # Assign labels to results
        labels = []
        for r in results:
            pid = r['product_id']
            if pid in label_dict:
                labels.append(label_map.get(label_dict[pid], 0))
            else:
                labels.append(0)  # Unknown = irrelevant
        
        training_data.append({
            'query': query,
            'results': results,
            'labels': labels,
            'query_analysis': query_analysis
        })
    
    return training_data


# Quick test
if __name__ == "__main__":
    print("Reranker module loaded successfully")
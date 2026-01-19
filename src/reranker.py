"""
CrossEncoder Reranker for Wholesale Product Retrieval.

Supports multiple models:
- cross-encoder/ms-marco-MiniLM-L-12-v2 (default, balanced)
- cross-encoder/ms-marco-MiniLM-L-6-v2 (faster)
- BAAI/bge-reranker-base (best quality, slower)
"""

from typing import List, Dict, Optional
from sentence_transformers import CrossEncoder
import numpy as np
from pathlib import Path
import pickle


class CrossEncoderReranker:
    """Reranker using CrossEncoder models."""
    
    # Available models
    MODELS = {
        "balanced": "cross-encoder/ms-marco-MiniLM-L-12-v2",
        "fast": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "quality": "BAAI/bge-reranker-base",
    }
    
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2",
        device: Optional[str] = None
    ):
        """
        Initialize the reranker.
        
        Args:
            model_name: Model name or shortcut ("balanced", "fast", "quality")
            device: "cpu", "cuda", or None (auto-detect)
        """
        # Allow shortcuts
        if model_name in self.MODELS:
            model_name = self.MODELS[model_name]
        
        self.model_name = model_name
        self.model = CrossEncoder(model_name, device=device)
        print(f"✓ Loaded CrossEncoder: {model_name}")
    
    # Wholesale/bulk quantity signals
    BULK_KEYWORDS = [
        "set of 6", "set of 8", "set of 12", "set of 24",
        "pack of", "case of", "dozen", "bulk", "wholesale",
        "6 pack", "12 pack", "24 pack", "multipack", "multi-pack"
    ]
    
    def _is_bulk_query(self, query: str) -> bool:
        """Check if query has wholesale/bulk intent."""
        bulk_signals = ["bulk", "wholesale", "case", "pack", "set of", "dozen"]
        query_lower = query.lower()
        return any(signal in query_lower for signal in bulk_signals)
    
    def _get_bulk_boost(self, title: str) -> float:
        """Calculate bulk boost based on product title."""
        title_lower = title.lower()
        for kw in self.BULK_KEYWORDS:
            if kw in title_lower:
                return 0.5
        return 0.0
    
    def rerank(
        self,
        query: str,
        results: List[Dict],
        top_k: Optional[int] = None,
        text_field: str = "product_title",
        apply_bulk_boost: bool = True
    ) -> List[Dict]:
        """
        Rerank search results using CrossEncoder.
        
        Args:
            query: Search query
            results: List of search results with product info
            top_k: Number of results to return (None = all)
            text_field: Field to use for reranking
            apply_bulk_boost: Whether to boost bulk products for bulk queries
        
        Returns:
            Reranked results with scores
        """
        if not results:
            return []
        
        # Check if this is a bulk query
        is_bulk_query = self._is_bulk_query(query) if apply_bulk_boost else False
        
        # Build query-document pairs
        pairs = []
        for r in results:
            text = r.get(text_field, "")
            if not text and "product" in r:
                text = r["product"].get(text_field, "")
            pairs.append([query, str(text)])
        
        # Score all pairs
        scores = self.model.predict(pairs)
        
        # Add scores and sort
        reranked = []
        for i, (result, score) in enumerate(zip(results, scores)):
            result_copy = result.copy()
            
            # Base score from CrossEncoder
            final_score = float(score)
            result_copy["crossencoder_score"] = final_score
            
            # Apply bulk boost if bulk query
            bulk_boost = 0.0
            if is_bulk_query:
                title = result_copy.get(text_field, "")
                if not title and "product" in result_copy:
                    title = result_copy["product"].get(text_field, "")
                bulk_boost = self._get_bulk_boost(title)
                final_score += bulk_boost
            
            result_copy["bulk_boost"] = bulk_boost
            result_copy["rerank_score"] = final_score
            result_copy["original_rank"] = i + 1
            reranked.append(result_copy)
        
        # Sort by rerank score (descending)
        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
        
        # Add new rank
        for i, r in enumerate(reranked):
            r["new_rank"] = i + 1
        
        return reranked[:top_k] if top_k else reranked
    
    def rerank_batch(
        self,
        queries: List[str],
        results_batch: List[List[Dict]],
        top_k: Optional[int] = None,
        text_field: str = "product_title"
    ) -> List[List[Dict]]:
        """
        Rerank multiple queries efficiently.
        
        Args:
            queries: List of search queries
            results_batch: List of result lists
            top_k: Number of results per query
            text_field: Field to use for reranking
        
        Returns:
            List of reranked results
        """
        return [
            self.rerank(q, r, top_k, text_field)
            for q, r in zip(queries, results_batch)
        ]
    
    def compare_rankings(
        self,
        query: str,
        results: List[Dict],
        top_k: int = 10,
        text_field: str = "product_title"
    ) -> Dict:
        """
        Show before/after comparison for analysis.
        
        Args:
            query: Search query
            results: Original results
            top_k: Number to compare
            text_field: Field for display
        
        Returns:
            Comparison dict with before/after
        """
        reranked = self.rerank(query, results, top_k, text_field)
        
        before = []
        after = []
        
        for i, r in enumerate(results[:top_k]):
            text = r.get(text_field, "")
            if not text and "product" in r:
                text = r["product"].get(text_field, "")
            before.append(f"{i+1}. {text[:60]}")
        
        for r in reranked:
            text = r.get(text_field, "")
            if not text and "product" in r:
                text = r["product"].get(text_field, "")
            move = r["original_rank"] - r["new_rank"]
            arrow = "↑" if move > 0 else "↓" if move < 0 else "="
            after.append(f"{r['new_rank']}. {text[:50]} {arrow}{abs(move)}")
        
        return {
            "query": query,
            "before": before,
            "after": after
        }


# Convenience function
def load_reranker(
    model: str = "balanced",
    device: Optional[str] = None
) -> CrossEncoderReranker:
    """
    Load a reranker with preset.
    
    Args:
        model: "balanced", "fast", "quality", or full model name
        device: "cpu", "cuda", or None
    
    Returns:
        CrossEncoderReranker instance
    """
    return CrossEncoderReranker(model_name=model, device=device)
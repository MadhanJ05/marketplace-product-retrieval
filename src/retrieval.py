"""
Hybrid Retrieval Module
Combines BM25 (keyword) + FAISS (semantic) search with RRF fusion.
"""

import pickle
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import faiss
from sentence_transformers import SentenceTransformer


class HybridRetriever:
    """Hybrid search combining BM25 and FAISS with Reciprocal Rank Fusion."""
    
    def __init__(
        self,
        indices_dir: str = "data/indices",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        self.indices_dir = Path(indices_dir)
        
        # Load indices
        self._load_bm25()
        self._load_faiss()
        self._load_product_ids()
        
        # Load embedding model for query encoding
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer(embedding_model)
        print(f"✓ HybridRetriever initialized")
        print(f"  Products: {len(self.product_ids):,}")
    
    def _load_bm25(self):
        """Load BM25 index."""
        bm25_path = self.indices_dir / "bm25_index.pkl"
        print(f"Loading BM25 from {bm25_path}...")
        with open(bm25_path, 'rb') as f:
            data = pickle.load(f)
        self.bm25 = data['bm25']
        self.tokenized_corpus = data['tokenized_corpus']
        print(f"✓ BM25 loaded: {len(self.tokenized_corpus):,} documents")
    
    def _load_faiss(self):
        """Load FAISS index."""
        faiss_path = self.indices_dir / "faiss_index.bin"
        print(f"Loading FAISS from {faiss_path}...")
        self.faiss_index = faiss.read_index(str(faiss_path))
        print(f"✓ FAISS loaded: {self.faiss_index.ntotal:,} vectors")
    
    def _load_product_ids(self):
        """Load product ID mapping."""
        ids_path = self.indices_dir / "product_ids.json"
        with open(ids_path, 'r') as f:
            self.product_ids = json.load(f)
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenizer matching BM25 index."""
        import re
        text = text.lower()
        return re.findall(r'\b\w+\b', text)
    
    def search_bm25(self, query: str, top_k: int = 100) -> List[Dict]:
        """
        Search using BM25 (keyword matching).
        
        Returns:
            List of {index, product_id, score}
        """
        query_tokens = self._tokenize(query)
        scores = self.bm25.get_scores(query_tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include matches
                results.append({
                    'index': int(idx),
                    'product_id': self.product_ids[idx],
                    'score': float(scores[idx]),
                    'source': 'bm25'
                })
        return results
    
    def search_faiss(self, query: str, top_k: int = 100) -> List[Dict]:
        """
        Search using FAISS (semantic similarity).
        
        Returns:
            List of {index, product_id, score}
        """
        # Encode query
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.faiss_index.search(query_embedding, top_k)
        
        results = []
        for i, (idx, score) in enumerate(zip(indices[0], scores[0])):
            results.append({
                'index': int(idx),
                'product_id': self.product_ids[idx],
                'score': float(score),
                'source': 'faiss'
            })
        return results
    
    def search_hybrid(
        self,
        query: str,
        top_k: int = 20,
        bm25_weight: float = 0.5,
        faiss_weight: float = 0.5,
        bm25_candidates: int = 100,
        faiss_candidates: int = 100,
        fusion_method: str = "rrf"
    ) -> List[Dict]:
        """
        Hybrid search combining BM25 and FAISS results.
        
        Args:
            query: Search query
            top_k: Number of final results
            bm25_weight: Weight for BM25 scores (used in weighted fusion)
            faiss_weight: Weight for FAISS scores (used in weighted fusion)
            bm25_candidates: Number of BM25 candidates to retrieve
            faiss_candidates: Number of FAISS candidates to retrieve
            fusion_method: "rrf" (Reciprocal Rank Fusion) or "weighted"
        
        Returns:
            List of {index, product_id, score, sources}
        """
        # Get candidates from both sources
        bm25_results = self.search_bm25(query, top_k=bm25_candidates)
        faiss_results = self.search_faiss(query, top_k=faiss_candidates)
        
        if fusion_method == "rrf":
            return self._fuse_rrf(bm25_results, faiss_results, top_k)
        else:
            return self._fuse_weighted(
                bm25_results, faiss_results, 
                bm25_weight, faiss_weight, top_k
            )
    
    def _fuse_rrf(
        self,
        bm25_results: List[Dict],
        faiss_results: List[Dict],
        top_k: int,
        k: int = 60
    ) -> List[Dict]:
        """
        Reciprocal Rank Fusion (RRF).
        
        RRF score = sum(1 / (k + rank)) for each source
        
        This method is robust and doesn't require score normalization.
        """
        scores = {}
        sources = {}
        
        # Score from BM25 rankings
        for rank, result in enumerate(bm25_results):
            idx = result['index']
            rrf_score = 1.0 / (k + rank + 1)
            scores[idx] = scores.get(idx, 0) + rrf_score
            sources[idx] = sources.get(idx, []) + ['bm25']
        
        # Score from FAISS rankings
        for rank, result in enumerate(faiss_results):
            idx = result['index']
            rrf_score = 1.0 / (k + rank + 1)
            scores[idx] = scores.get(idx, 0) + rrf_score
            sources[idx] = sources.get(idx, []) + ['faiss']
        
        # Sort by combined score
        sorted_indices = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        
        results = []
        for idx in sorted_indices[:top_k]:
            results.append({
                'index': idx,
                'product_id': self.product_ids[idx],
                'score': scores[idx],
                'sources': sources[idx]
            })
        
        return results
    
    def _fuse_weighted(
        self,
        bm25_results: List[Dict],
        faiss_results: List[Dict],
        bm25_weight: float,
        faiss_weight: float,
        top_k: int
    ) -> List[Dict]:
        """
        Weighted score fusion with min-max normalization.
        """
        scores = {}
        sources = {}
        
        # Normalize and add BM25 scores
        if bm25_results:
            bm25_scores = [r['score'] for r in bm25_results]
            min_s, max_s = min(bm25_scores), max(bm25_scores)
            range_s = max_s - min_s if max_s > min_s else 1
            
            for result in bm25_results:
                idx = result['index']
                norm_score = (result['score'] - min_s) / range_s
                scores[idx] = scores.get(idx, 0) + bm25_weight * norm_score
                sources[idx] = sources.get(idx, []) + ['bm25']
        
        # Normalize and add FAISS scores
        if faiss_results:
            faiss_scores = [r['score'] for r in faiss_results]
            min_s, max_s = min(faiss_scores), max(faiss_scores)
            range_s = max_s - min_s if max_s > min_s else 1
            
            for result in faiss_results:
                idx = result['index']
                norm_score = (result['score'] - min_s) / range_s
                scores[idx] = scores.get(idx, 0) + faiss_weight * norm_score
                sources[idx] = sources.get(idx, []) + ['faiss']
        
        # Sort by combined score
        sorted_indices = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        
        results = []
        for idx in sorted_indices[:top_k]:
            results.append({
                'index': idx,
                'product_id': self.product_ids[idx],
                'score': scores[idx],
                'sources': sources[idx]
            })
        
        return results
    
    def search(
        self,
        query: str,
        top_k: int = 20,
        method: str = "hybrid",
        **kwargs
    ) -> List[Dict]:
        """
        Main search interface.
        
        Args:
            query: Search query
            top_k: Number of results
            method: "hybrid", "bm25", or "faiss"
        
        Returns:
            List of search results
        """
        if method == "bm25":
            return self.search_bm25(query, top_k)
        elif method == "faiss":
            return self.search_faiss(query, top_k)
        else:
            return self.search_hybrid(query, top_k, **kwargs)


# Quick test
if __name__ == "__main__":
    retriever = HybridRetriever()
    
    test_queries = [
        "ceramic mugs bulk",
        "running shoes nike",
        "organic candles lavender"
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: '{query}'")
        print('='*60)
        
        results = retriever.search(query, top_k=5)
        for r in results:
            sources = ', '.join(r['sources'])
            print(f"  [{sources}] {r['product_id']} (score: {r['score']:.4f})")
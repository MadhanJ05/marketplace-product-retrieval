"""
Query Understanding Module using Groq API (Free Tier)
Extracts product type, attributes, quantity signals, intent, and expanded terms.
"""

import os
import json
import re
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from groq import Groq

load_dotenv()


class QueryUnderstanding:
    """LLM-powered query understanding using Groq (free tier)."""
    
    def __init__(
        self,
        model: str = "llama-3.1-8b-instant",  # Fast & free
        use_cache: bool = True,
        cache_dir: str = "data/cache/query_understanding"
    ):
        self.model = model
        self.use_cache = use_cache
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "cache.json"
        self.cache = self._load_cache()
        
        # Initialize Groq client
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        self.client = Groq(api_key=api_key)
        
        # System prompt for query analysis
        self.system_prompt = """You are a search query analyzer for a wholesale B2B marketplace (like Faire).
Your job is to extract structured information from product search queries.

For each query, extract:
1. product_type: The main product being searched (e.g., "mugs", "candles", "t-shirts")
2. attributes: List of descriptors like color, material, style, brand (e.g., ["ceramic", "blue", "minimalist"])
3. quantity_signal: Any bulk/wholesale indicators (e.g., "bulk", "wholesale", "pack of 50", "case pack", or null if none)
4. intent: Either "wholesale_purchase" (if bulk signals present) or "product_search" (regular search)
5. expanded_terms: 3-5 synonyms or related terms that could help find relevant products

IMPORTANT: Respond ONLY with valid JSON, no markdown, no explanation. Example:
{"product_type": "mugs", "attributes": ["ceramic", "blue"], "quantity_signal": "bulk", "intent": "wholesale_purchase", "expanded_terms": ["cups", "drinkware", "coffee mugs"]}"""

    def _load_cache(self) -> dict:
        """Load cache from disk."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def save_cache(self):
        """Save cache to disk."""
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f, indent=2)
        print(f"Cache saved: {len(self.cache)} entries")
    
    def _get_default_response(self, query: str) -> dict:
        """Return default response when API fails."""
        return {
            "product_type": query,
            "attributes": [],
            "quantity_signal": None,
            "intent": "product_search",
            "expanded_terms": []
        }
    
    def _parse_response(self, text: str, query: str) -> dict:
        """Parse LLM response to structured dict."""
        try:
            # Clean up response - remove markdown code blocks if present
            text = text.strip()
            text = re.sub(r'^```json\s*', '', text)
            text = re.sub(r'^```\s*', '', text)
            text = re.sub(r'\s*```$', '', text)
            
            result = json.loads(text)
            
            # Validate required fields
            required = ["product_type", "attributes", "quantity_signal", "intent", "expanded_terms"]
            for field in required:
                if field not in result:
                    result[field] = self._get_default_response(query)[field]
            
            return result
        except json.JSONDecodeError:
            return self._get_default_response(query)
    
    def analyze(self, query: str) -> dict:
        """
        Analyze a search query and extract structured information.
        
        Args:
            query: The search query string
            
        Returns:
            dict with keys: product_type, attributes, quantity_signal, intent, expanded_terms
        """
        query = query.strip().lower()
        
        # Check cache first
        if self.use_cache and query in self.cache:
            return self.cache[query]
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"Analyze this search query: {query}"}
                ],
                temperature=0.1,
                max_tokens=300
            )
            
            result = self._parse_response(response.choices[0].message.content, query)
            
            # Cache the result
            if self.use_cache:
                self.cache[query] = result
            
            return result
            
        except Exception as e:
            print(f"Error analyzing query '{query}': {e}")
            return self._get_default_response(query)
    
    def get_expanded_query(self, query: str) -> str:
        """
        Get an expanded version of the query with synonyms.
        
        Args:
            query: Original search query
            
        Returns:
            Expanded query string
        """
        analysis = self.analyze(query)
        
        # Build expanded query
        parts = [query]
        if analysis.get("expanded_terms"):
            parts.extend(analysis["expanded_terms"][:3])  # Add top 3 synonyms
        
        return " ".join(parts)
    
    def batch_analyze(self, queries: list, show_progress: bool = True) -> list:
        """
        Analyze multiple queries.
        
        Args:
            queries: List of query strings
            show_progress: Whether to show progress bar
            
        Returns:
            List of analysis dicts
        """
        results = []
        
        if show_progress:
            from tqdm import tqdm
            queries = tqdm(queries, desc="Analyzing queries")
        
        for query in queries:
            results.append(self.analyze(query))
        
        return results


# Quick test
if __name__ == "__main__":
    qu = QueryUnderstanding()
    
    test_queries = [
        "ceramic mugs bulk",
        "organic cotton t-shirts",
        "wholesale candles lavender"
    ]
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        result = qu.analyze(query)
        print(f"  Product type: {result['product_type']}")
        print(f"  Attributes: {result['attributes']}")
        print(f"  Quantity signal: {result['quantity_signal']}")
        print(f"  Intent: {result['intent']}")
        print(f"  Expanded terms: {result['expanded_terms']}")
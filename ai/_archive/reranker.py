"""
Cross-encoder re-ranker for semantic search
Provides better accuracy by jointly encoding query + movie
"""

from sentence_transformers import CrossEncoder
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


class SemanticReranker:
    def __init__(self):
        """Initialize cross-encoder for re-ranking"""
        logger.info("Loading cross-encoder model...")
        try:
            # Cross-encoder for re-ranking top candidates
            self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            logger.info("✓ Cross-encoder loaded: ms-marco-MiniLM-L-6-v2")
        except Exception as e:
            logger.error(f"Failed to load cross-encoder: {e}")
            self.reranker = None
    
    def rerank(self, query: str, candidates: List[Dict], top_k: int = 10) -> List[Dict]:
        """
        Re-rank candidates using cross-encoder
        
        Args:
            query: Search query string
            candidates: List of candidate movies from initial search
            top_k: Number of top results to return
            
        Returns:
            Re-ranked list of movies with updated scores
        """
        if self.reranker is None:
            logger.warning("Cross-encoder not loaded, returning original candidates")
            return candidates[:top_k]
        
        if not candidates:
            return []
        
        # Prepare query-movie pairs
        pairs = []
        for movie in candidates:
            # Combine title and overview for matching
            text = f"{movie.get('title', '')}. {movie.get('overview', '')}"
            pairs.append([query, text])
        
        # Get cross-encoder scores
        try:
            scores = self.reranker.predict(pairs)
            
            # Add scores to candidates
            for i, movie in enumerate(candidates):
                movie['rerank_score'] = float(scores[i])
            
            # Sort by rerank score
            candidates.sort(key=lambda x: x['rerank_score'], reverse=True)
            
            logger.info(f"✓ Re-ranked {len(candidates)} candidates")
            
        except Exception as e:
            logger.error(f"Re-ranking failed: {e}")
            return candidates[:top_k]
        
        return candidates[:top_k]
    
    def rerank_with_context(
        self, 
        query: str, 
        candidates: List[Dict], 
        user_context: Dict = None,
        top_k: int = 10
    ) -> List[Dict]:
        """
        Re-rank with additional user context (preferences, history)
        
        Args:
            query: Search query
            candidates: Candidate movies
            user_context: Optional dict with user preferences, history
            top_k: Number of results
        """
        if self.reranker is None:
            return candidates[:top_k]
        
        # Enhance query with user context
        enhanced_query = query
        if user_context:
            if user_context.get('favorite_genres'):
                genres = ', '.join(user_context['favorite_genres'][:3])
                enhanced_query += f" (User likes: {genres})"
            
            if user_context.get('mood'):
                enhanced_query += f" (Current mood: {user_context['mood']})"
        
        # Rerank with enhanced query
        return self.rerank(enhanced_query, candidates, top_k)


# Example usage
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    # Sample candidates
    sample_candidates = [
        {
            'movie_id': 1,
            'title': 'Inception',
            'overview': 'A thief who enters the dreams of others to steal secrets.',
            'relevance_score': 0.85
        },
        {
            'movie_id': 2,
            'title': 'The Matrix',
            'overview': 'A hacker discovers the truth about reality and his role in the war.',
            'relevance_score': 0.82
        }
    ]
    
    # Test re-ranking
    reranker = SemanticReranker()
    query = "mind-bending sci-fi about dreams"
    results = reranker.rerank(query, sample_candidates, top_k=5)
    
    print(f"\nRe-ranked results for: '{query}'")
    for i, movie in enumerate(results, 1):
        print(f"{i}. {movie['title']} (score: {movie.get('rerank_score', 0):.3f})")

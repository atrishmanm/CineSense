"""
Semantic movie search using advanced embeddings
Handles natural language queries like "indian spy thriller" or "time loop movie"
"""

from sentence_transformers import SentenceTransformer, util
import numpy as np
from typing import List, Dict, Tuple
import torch
import json
import os
import pickle
import logging

logger = logging.getLogger(__name__)


class SemanticMovieSearch:
    def __init__(self, cache_dir='model/semantic_cache'):
        """Initialize semantic search engine"""
        self.local_only = os.getenv('SEMANTIC_LOCAL_ONLY', '0').lower() in {'1', 'true', 'yes', 'on'}

        # Use local cache first to avoid unnecessary Hugging Face network checks.
        logger.info("Loading semantic search model...")
        try:
            self.model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', local_files_only=True)
            logger.info("✓ Model loaded from local cache: all-mpnet-base-v2 (768-dim)")
        except Exception as e_local_mpnet:
            if self.local_only:
                logger.warning(f"Local-only semantic mode: all-mpnet-base-v2 unavailable: {e_local_mpnet}")
                self.model = SentenceTransformer('all-MiniLM-L6-v2', local_files_only=True)
                logger.info("✓ Model loaded from local cache: all-MiniLM-L6-v2")
            else:
                try:
                    self.model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
                    logger.info("✓ Model downloaded: all-mpnet-base-v2 (768-dim)")
                except Exception as e_remote_mpnet:
                    logger.warning(f"Failed to load all-mpnet-base-v2, falling back to all-MiniLM-L6-v2: {e_remote_mpnet}")
                    self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        self.movie_embeddings = None
        self.movie_data = None
        self.faiss_index = None
        self.cross_encoder = None
        self.enable_cross_encoder = os.getenv('ENABLE_CROSS_ENCODER', '0').lower() in {'1', 'true', 'yes', 'on'}
        self.cross_encoder_local_only = os.getenv('CROSS_ENCODER_LOCAL_ONLY', '1').lower() in {'1', 'true', 'yes', 'on'}
        
    def build_index(self, movies: List[Dict]):
        """
        Pre-compute embeddings for all movies
        Call this once during startup or when adding new movies
        """
        logger.info(f"Building semantic index for {len(movies)} movies...")
        
        # Create rich text corpus combining all movie info
        corpus = []
        for movie in movies:
            # Combine multiple fields for better matching
            text_parts = [
                f"Title: {movie.get('title', '')}",
            ]
            
            # Include original title for non-English movies
            if movie.get('original_title') and movie.get('original_title') != movie.get('title'):
                text_parts.append(f"Original Title: {movie['original_title']}")
            
            text_parts.append(f"Plot: {movie.get('overview', '')}")
            
            if isinstance(movie.get('genres'), list):
                text_parts.append(f"Genres: {', '.join(movie.get('genres', []))}")
            else:
                text_parts.append(f"Genres: {movie.get('genres', '')}")
            
            # Add optional fields if available
            if movie.get('keywords'):
                keywords = movie['keywords'][:10] if isinstance(movie['keywords'], list) else str(movie['keywords']).split(',')[:10]
                text_parts.append(f"Keywords: {', '.join(keywords)}")
            
            if movie.get('director'):
                text_parts.append(f"Director: {movie['director']}")
                
            if movie.get('cast'):
                cast = movie['cast'][:5] if isinstance(movie['cast'], list) else str(movie['cast']).split(',')[:5]
                text_parts.append(f"Cast: {', '.join(cast)}")
            
            # Add language/country info for regional search
            if movie.get('original_language'):
                lang = movie['original_language']
                lang_names = {
                    'hi': 'Hindi Indian Bollywood', 'ta': 'Tamil Indian',
                    'te': 'Telugu Indian', 'ml': 'Malayalam Indian',
                    'kn': 'Kannada Indian', 'bn': 'Bengali Indian',
                    'mr': 'Marathi Indian', 'pa': 'Punjabi Indian',
                    'ko': 'Korean', 'ja': 'Japanese', 'zh': 'Chinese Mandarin',
                    'fr': 'French', 'es': 'Spanish', 'de': 'German',
                    'it': 'Italian', 'pt': 'Portuguese', 'ru': 'Russian',
                    'ar': 'Arabic', 'tr': 'Turkish', 'th': 'Thai',
                    'en': 'English',
                }
                lang_desc = lang_names.get(lang, lang)
                text_parts.append(f"Language: {lang_desc}")
            
            corpus_text = ' | '.join(text_parts)
            corpus.append(corpus_text)
        
        # Generate embeddings (batch processing for speed)
        self.movie_embeddings = self.model.encode(
            corpus,
            convert_to_tensor=True,
            show_progress_bar=True,
            batch_size=64,
            normalize_embeddings=True  # For cosine similarity
        )
        
        self.movie_data = movies
        
        # Build FAISS index from new embeddings
        self._build_faiss_index(self.movie_embeddings.cpu().numpy())
        
        # Save to cache
        self._save_cache()
        
        logger.info(f"✓ Index built successfully! Embeddings shape: {self.movie_embeddings.shape}")
    
    def search(self, query: str, top_k: int = 20, min_score: float = 0.3) -> List[Dict]:
        """
        Search movies using natural language query
        
        Args:
            query: Natural language description (e.g., "thriller about spies in india")
            top_k: Number of results to return
            min_score: Minimum similarity score (0-1)
            
        Returns:
            List of movies with relevance scores
        """
        if self.movie_embeddings is None:
            raise ValueError("Index not built! Call build_index() first")
        
        # Encode query
        query_embedding = self.model.encode(
            query,
            convert_to_tensor=True,
            normalize_embeddings=True
        )
        
        # Fix device mismatch: ensure query embedding is on same device as movie embeddings
        query_embedding = query_embedding.to(self.movie_embeddings.device)
        
        results = []
        fallback_candidates = []
        if self.faiss_index is not None:
            # FAISS vector search (fast approximate nearest neighbour)
            query_np = query_embedding.cpu().numpy().astype(np.float32).reshape(1, -1)
            scores, indices = self.faiss_index.search(query_np, min(top_k * 2, len(self.movie_data)))
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0:
                    continue
                score_val = float(score)
                fallback_candidates.append((score_val, int(idx)))
                if score_val >= min_score:
                    movie = self.movie_data[int(idx)].copy()
                    movie['relevance_score'] = score_val
                    movie['match_type'] = 'semantic'
                    results.append(movie)
        else:
            # Fallback: full-matrix cosine similarity
            cos_scores = util.cos_sim(query_embedding, self.movie_embeddings)[0]
            top_results = torch.topk(cos_scores, k=min(top_k * 2, len(cos_scores)))
            for score, idx in zip(top_results[0], top_results[1]):
                score_val = float(score)
                fallback_candidates.append((score_val, int(idx)))
                if score_val >= min_score:
                    movie = self.movie_data[int(idx)].copy()
                    movie['relevance_score'] = score_val
                    movie['match_type'] = 'semantic'
                    results.append(movie)

        # If strict threshold removes all items, return best semantic candidates anyway.
        if not results and fallback_candidates:
            for score_val, idx in fallback_candidates[:top_k]:
                movie = self.movie_data[idx].copy()
                movie['relevance_score'] = score_val
                movie['match_type'] = 'semantic_fallback'
                results.append(movie)
        
        return results[:top_k]
    
    def hybrid_search(
        self, 
        query: str, 
        top_k: int = 20,
        semantic_weight: float = 0.7
    ) -> List[Dict]:
        """
        Combine semantic search with keyword matching for best results
        
        Args:
            query: Search query
            top_k: Number of results
            semantic_weight: Weight for semantic vs keyword (0-1)
        """
        # Stage 1: Semantic search (get more candidates)
        semantic_results = self.search(query, top_k=50, min_score=0.2)
        
        # Stage 2: Keyword boosting
        keywords = self._extract_keywords(query)
        
        for result in semantic_results:
            # Calculate keyword match score
            keyword_score = self._calculate_keyword_match(keywords, result)
            
            # Combine scores
            result['keyword_score'] = keyword_score
            result['final_score'] = (
                semantic_weight * result['relevance_score'] +
                (1 - semantic_weight) * keyword_score
            )
        
        # Stage 3: Cross-encoder reranking on top candidates (optional)
        if self.enable_cross_encoder and self._load_cross_encoder():
            try:
                top_candidates = sorted(semantic_results, key=lambda x: x['final_score'], reverse=True)[:20]
                pairs = [
                    [query, f"{r.get('title', '')} {r.get('overview', '')[:200]}"]
                    for r in top_candidates
                ]
                rerank_scores = self.cross_encoder.predict(pairs)
                min_s = float(rerank_scores.min())
                max_s = float(rerank_scores.max())
                score_range = max_s - min_s if max_s > min_s else 1.0
                for result, rs in zip(top_candidates, rerank_scores):
                    normalized = (float(rs) - min_s) / score_range
                    result['rerank_score'] = normalized
                    result['final_score'] = (
                        0.5 * result['relevance_score'] +
                        0.2 * result['keyword_score'] +
                        0.3 * normalized
                    )
            except Exception as e:
                logger.warning(f"Cross-encoder reranking failed: {e}")
        
        # Sort by final score
        semantic_results.sort(key=lambda x: x['final_score'], reverse=True)
        
        return semantic_results[:top_k]
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract important keywords from query with region/language awareness"""
        # Common stop words
        stopwords = {
            'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been',
            'about', 'who', 'what', 'where', 'when', 'why', 'how',
            'movie', 'film', 'show', 'like', 'want', 'find', 'search',
            'some', 'good', 'best', 'great', 'nice', 'top', 'new',
            'give', 'me', 'for', 'with', 'and', 'but', 'or', 'not',
            'very', 'really', 'just', 'also', 'can', 'you', 'please'
        }
        
        # Region/country -> related keywords for better matching
        region_expansions = {
            'indian': ['india', 'indian', 'bollywood', 'hindi', 'tamil', 'telugu', 'desi', 'mumbai', 'delhi'],
            'bollywood': ['india', 'indian', 'bollywood', 'hindi'],
            'korean': ['korea', 'korean', 'k-drama', 'seoul'],
            'japanese': ['japan', 'japanese', 'anime', 'tokyo'],
            'chinese': ['china', 'chinese', 'mandarin', 'hong kong'],
            'french': ['france', 'french', 'paris'],
            'british': ['british', 'uk', 'england', 'london'],
            'spanish': ['spain', 'spanish', 'mexican', 'latin'],
            'thai': ['thai', 'thailand'],
            'turkish': ['turkish', 'turkey'],
            'italian': ['italian', 'italy'],
            'german': ['german', 'germany'],
            'arabic': ['arabic', 'arab', 'middle east'],
            'nigerian': ['nigerian', 'nollywood'],
        }
        
        # Tokenize and filter
        words = query.lower().split()
        keywords = [
            word for word in words 
            if word not in stopwords and len(word) > 2
        ]
        
        # Add expanded keywords for region terms
        expanded = []
        for word in keywords:
            expanded.append(word)
            if word in region_expansions:
                for expansion in region_expansions[word]:
                    if expansion not in expanded:
                        expanded.append(expansion)
        
        return expanded
    
    def _calculate_keyword_match(self, keywords: List[str], movie: Dict) -> float:
        """Calculate keyword overlap score with weighted matching"""
        # Combine all searchable text
        searchable_parts = [
            str(movie.get('title', '')),
            str(movie.get('overview', '')),
            str(movie.get('original_title', '')),
        ]
        
        # Handle genres
        if isinstance(movie.get('genres'), list):
            searchable_parts.append(' '.join(movie.get('genres', [])))
        else:
            searchable_parts.append(str(movie.get('genres', '')))
        
        # Handle keywords
        if isinstance(movie.get('keywords'), list):
            searchable_parts.append(' '.join(movie.get('keywords', [])[:20]))
        else:
            searchable_parts.append(str(movie.get('keywords', ''))[:200])
        
        # Handle cast and director
        if movie.get('cast'):
            if isinstance(movie['cast'], list):
                searchable_parts.append(' '.join(movie['cast']))
            else:
                searchable_parts.append(str(movie['cast']))
        
        if movie.get('director'):
            searchable_parts.append(str(movie['director']))
        
        # Handle origin country / production info
        if movie.get('original_language'):
            searchable_parts.append(str(movie['original_language']))
        if movie.get('production_countries'):
            searchable_parts.append(str(movie['production_countries']))
        
        searchable_text = ' '.join(searchable_parts).lower()
        
        if not keywords:
            return 0.0
        
        # Weighted matching: title matches worth more
        title_text = str(movie.get('title', '')).lower() + ' ' + str(movie.get('original_title', '')).lower()
        genre_text = ''
        if isinstance(movie.get('genres'), list):
            genre_text = ' '.join(movie.get('genres', [])).lower()
        else:
            genre_text = str(movie.get('genres', '')).lower()
        
        score = 0.0
        for kw in keywords:
            if kw in title_text:
                score += 2.0  # Title match is high value
            elif kw in genre_text:
                score += 1.5  # Genre match is medium-high value
            elif kw in searchable_text:
                score += 1.0  # Content match is standard value
        
        # Normalize by number of keywords
        return score / (len(keywords) * 2.0)  # Normalize to 0-1 range roughly
    
    def _save_cache(self):
        """Save embeddings and movie data to disk"""
        try:
            cache_file = os.path.join(self.cache_dir, 'embeddings.pkl')
            movie_data_file = os.path.join(self.cache_dir, 'movie_data.pkl')
            metadata_file = os.path.join(self.cache_dir, 'metadata.json')
            
            # Save embeddings
            with open(cache_file, 'wb') as f:
                pickle.dump(self.movie_embeddings.cpu().numpy(), f)
            
            # Save movie data (needed for returning results)
            with open(movie_data_file, 'wb') as f:
                pickle.dump(self.movie_data, f)
            
            # Save metadata
            movie_ids = [m.get('movie_id') for m in self.movie_data]
            with open(metadata_file, 'w') as f:
                json.dump({'movie_ids': movie_ids, 'count': len(movie_ids)}, f)
            
            logger.info(f"✓ Cache saved to {self.cache_dir}")
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")
    
    def _build_faiss_index(self, embeddings_np: np.ndarray):
        """Build a FAISS IndexFlatIP from an L2-normalised numpy embedding array."""
        try:
            import faiss
            embeddings_f32 = embeddings_np.astype(np.float32)
            d = embeddings_f32.shape[1]
            index = faiss.IndexFlatIP(d)
            index.add(embeddings_f32)
            self.faiss_index = index
            logger.info(f"✓ FAISS index built: {index.ntotal} vectors, {d}-dim")
        except ImportError:
            logger.warning("faiss-cpu not installed — using torch cosine similarity fallback")
            self.faiss_index = None
        except Exception as e:
            logger.warning(f"FAISS index build failed: {e}")
            self.faiss_index = None

    def _load_cross_encoder(self) -> bool:
        """Lazy-load the cross-encoder reranker on first call."""
        if self.cross_encoder is not None:
            return True
        if not self.enable_cross_encoder:
            return False
        try:
            from sentence_transformers import CrossEncoder
            self.cross_encoder = CrossEncoder(
                'cross-encoder/ms-marco-MiniLM-L-6-v2',
                local_files_only=self.cross_encoder_local_only,
            )
            logger.info("✓ Cross-encoder loaded: ms-marco-MiniLM-L-6-v2")
            return True
        except Exception as e:
            logger.warning(f"Cross-encoder unavailable: {e}")
            self.cross_encoder = None
            return False

    def load_cache(self):
        """Load pre-computed embeddings and movie data from disk"""
        try:
            cache_file = os.path.join(self.cache_dir, 'embeddings.pkl')
            movie_data_file = os.path.join(self.cache_dir, 'movie_data.pkl')
            
            if os.path.exists(cache_file) and os.path.exists(movie_data_file):
                logger.info("Loading cached embeddings and movie data...")
                with open(cache_file, 'rb') as f:
                    embeddings_np = pickle.load(f)
                    self.movie_embeddings = torch.from_numpy(embeddings_np)
                with open(movie_data_file, 'rb') as f:
                    self.movie_data = pickle.load(f)
                # Rebuild FAISS index from cached embeddings (no separate file needed)
                self._build_faiss_index(embeddings_np)
                logger.info(f"✓ Cache loaded: {len(self.movie_data)} movies")
                return True
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
        
        return False


# Example usage for testing
if __name__ == '__main__':
    # Sample movies
    sample_movies = [
        {
            'movie_id': 1,
            'title': 'Inception',
            'overview': 'A thief who enters the dreams of others to steal secrets from their subconscious.',
            'genres': ['Action', 'Sci-Fi', 'Thriller'],
            'director': 'Christopher Nolan',
            'cast': ['Leonardo DiCaprio', 'Tom Hardy']
        },
        {
            'movie_id': 2,
            'title': 'The Prestige',
            'overview': 'Two magicians engage in a competitive rivalry involving illusions and obsession.',
            'genres': ['Drama', 'Mystery', 'Thriller'],
            'director': 'Christopher Nolan',
            'cast': ['Christian Bale', 'Hugh Jackman']
        }
    ]
    
    # Initialize and test
    searcher = SemanticMovieSearch()
    searcher.build_index(sample_movies)
    
    # Test queries
    queries = [
        "mind-bending thriller about dreams",
        "magic and competition",
        "christopher nolan movies"
    ]
    
    for query in queries:
        print(f"\nQuery: '{query}'")
        results = searcher.hybrid_search(query, top_k=5)
        for i, movie in enumerate(results, 1):
            print(f"  {i}. {movie['title']} (score: {movie['final_score']:.3f})")

"""
Query understanding and expansion using LLMs
Enhances user queries with relevant keywords and context
"""

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import Dict, List
import json
import logging
import re

logger = logging.getLogger(__name__)


class QueryEnhancer:
    def __init__(self, model_name: str = "google/flan-t5-small"):
        """
        Initialize query enhancement model
        
        Args:
            model_name: HuggingFace model to use for query understanding
        """
        logger.info(f"Loading query enhancement model: {model_name}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            self.model.eval()  # Set to evaluation mode
            logger.info(f"✓ Query enhancement model loaded")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.tokenizer = None
            self.model = None
        
    def expand_query(self, query: str, max_length: int = 256) -> str:
        """
        Expand user query with synonyms and relevant keywords
        
        Args:
            query: Original user query
            max_length: Maximum length for expanded output
            
        Returns:
            Expanded query string
        """
        if self.model is None:
            return query
        
        prompt = f"Expand this movie search query with relevant keywords: {query}"
        
        try:
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                max_length=128, 
                truncation=True
            )
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs, 
                    max_length=max_length,
                    num_beams=4,
                    early_stopping=True
                )
            
            expanded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.debug(f"Query expanded: '{query}' -> '{expanded}'")
            return expanded
            
        except Exception as e:
            logger.error(f"Query expansion failed: {e}")
            return query
    
    def extract_entities(self, query: str) -> Dict[str, List[str]]:
        """
        Extract entities from query (genres, actors, themes)
        
        Args:
            query: User search query
            
        Returns:
            Dictionary with extracted entities
        """
        if self.model is None:
            return self._fallback_entity_extraction(query)
        
        prompt = f"""Extract from this movie query and return as JSON:
Query: {query}

Return format:
{{"genres": [], "actors": [], "directors": [], "themes": [], "keywords": []}}
"""
        
        try:
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                max_length=256, 
                truncation=True
            )
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs, 
                    max_length=512,
                    num_beams=4
                )
            
            result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Try to parse JSON
            try:
                entities = json.loads(result)
                return entities
            except json.JSONDecodeError:
                logger.warning("Failed to parse JSON, using fallback")
                return self._fallback_entity_extraction(query)
                
        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            return self._fallback_entity_extraction(query)
    
    def _fallback_entity_extraction(self, query: str) -> Dict[str, List[str]]:
        """
        Simple rule-based entity extraction as fallback
        
        Args:
            query: Search query
            
        Returns:
            Dictionary with basic entity extraction
        """
        query_lower = query.lower()
        
        # Common genre keywords
        genre_keywords = {
            'action': ['action', 'fight', 'martial arts', 'explosion', 'chase'],
            'thriller': ['thriller', 'suspense', 'mystery', 'detective', 'spy'],
            'comedy': ['comedy', 'funny', 'humor', 'laugh', 'hilarious'],
            'horror': ['horror', 'scary', 'ghost', 'zombie', 'demon'],
            'romance': ['romance', 'love', 'romantic', 'relationship'],
            'sci-fi': ['sci-fi', 'science fiction', 'space', 'alien', 'future', 'robot'],
            'drama': ['drama', 'emotional', 'serious'],
            'fantasy': ['fantasy', 'magic', 'wizard', 'dragon', 'medieval'],
            'animation': ['animated', 'animation', 'cartoon'],
            'documentary': ['documentary', 'true story', 'real', 'biography']
        }
        
        # Detect genres
        detected_genres = []
        for genre, keywords in genre_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                detected_genres.append(genre)
        
        # Extract basic themes
        theme_keywords = {
            'time travel': ['time travel', 'time loop', 'time machine'],
            'superhero': ['superhero', 'super hero', 'superpowers'],
            'war': ['war', 'battle', 'military', 'soldier'],
            'crime': ['crime', 'criminal', 'heist', 'robbery'],
            'survival': ['survival', 'stranded', 'lost'],
            'revenge': ['revenge', 'vengeance'],
            'coming of age': ['coming of age', 'teenager', 'growing up']
        }
        
        detected_themes = []
        for theme, keywords in theme_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                detected_themes.append(theme)
        
        # Extract keywords (simple tokenization)
        stopwords = {
            'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been',
            'about', 'who', 'what', 'where', 'when', 'why', 'how',
            'movie', 'film', 'show', 'like', 'want', 'find', 'search'
        }
        
        words = re.findall(r'\b\w+\b', query_lower)
        keywords = [w for w in words if w not in stopwords and len(w) > 2]
        
        return {
            'genres': detected_genres,
            'actors': [],  # Can't reliably extract without NER
            'directors': [],  # Can't reliably extract without NER
            'themes': detected_themes,
            'keywords': keywords[:10]
        }
    
    def detect_mood(self, query: str) -> str:
        """
        Detect user mood from query
        
        Args:
            query: User input
            
        Returns:
            Detected mood string
        """
        query_lower = query.lower()
        
        mood_indicators = {
            'happy': ['happy', 'cheerful', 'upbeat', 'fun', 'light-hearted'],
            'sad': ['sad', 'depressing', 'tearjerker', 'emotional', 'cry'],
            'excited': ['exciting', 'thrilling', 'intense', 'action-packed'],
            'relaxed': ['calm', 'peaceful', 'slow', 'quiet', 'relaxing'],
            'stressed': ['stressed', 'tense', 'need to unwind'],
            'bored': ['bored', 'something interesting', 'surprise me'],
            'romantic': ['romantic', 'date night', 'love story'],
            'scared': ['scary', 'horror', 'frightening', 'creepy']
        }
        
        for mood, indicators in mood_indicators.items():
            if any(indicator in query_lower for indicator in indicators):
                return mood
        
        return 'neutral'
    
    def enhance_query_with_context(
        self, 
        query: str, 
        user_history: List[str] = None,
        user_preferences: Dict = None
    ) -> str:
        """
        Enhance query with user context
        
        Args:
            query: Original query
            user_history: List of movies user has enjoyed
            user_preferences: Dict of user preferences
            
        Returns:
            Context-enhanced query
        """
        enhanced = query
        
        if user_preferences:
            if user_preferences.get('favorite_genres'):
                genres = ', '.join(user_preferences['favorite_genres'][:2])
                enhanced += f" (user enjoys {genres} movies)"
        
        if user_history and len(user_history) > 0:
            recent = ', '.join(user_history[:3])
            enhanced += f" (recently watched: {recent})"
        
        return enhanced


# Example usage
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    enhancer = QueryEnhancer()
    
    # Test queries
    test_queries = [
        "indian spy thriller",
        "time loop movie",
        "funny animated movie for kids",
        "sad movie about war"
    ]
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        entities = enhancer.extract_entities(query)
        mood = enhancer.detect_mood(query)
        print(f"  Genres: {entities['genres']}")
        print(f"  Themes: {entities['themes']}")
        print(f"  Keywords: {entities['keywords']}")
        print(f"  Mood: {mood}")

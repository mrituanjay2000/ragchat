from mistralai import Mistral
import numpy as np
from typing import List
import os
from .config import settings
from .utils import setup_logger
import time
logger = setup_logger("embeddings")

class MistralEmbeddings:
    def __init__(self):
        """Initialize the Mistral client"""
        try:
            self.client = Mistral(api_key=settings.MISTRAL_API_KEY)
            logger.info("Initialized Mistral client successfully")
        except Exception as e:
            logger.error(f"Error initializing Mistral client: {str(e)}")
            raise

    def get_text_embedding(self, input_text: str) -> np.ndarray:
        """Get embedding for a single text using the latest Mistral API"""
        try:
            if not input_text:
                logger.warning("Input text is empty. Cannot generate embedding.")
                return np.array([])
            time.sleep(2)
            embeddings_response = self.client.embeddings.create(
                model="mistral-embed",
                inputs=input_text
            )
            return embeddings_response.data[0].embedding
        except Exception as e:
            logger.error(f"Error getting text embedding: {str(e)}")
            raise

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts"""
        try:
            if not texts:
                logger.warning("No texts provided for embedding")
                return np.array([])

            logger.info(f"Generating embeddings for {len(texts)} texts")
            # Use list comprehension for better performance
            embeddings = [self.get_text_embedding(text) for text in texts]
            return np.array(embeddings)
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise

    def embed_query(self, text: str) -> np.ndarray:
        """Generate embedding for a single query text"""
        try:
            logger.debug(f"Generating embedding for query: {text[:100]}...")
            embedding = self.get_text_embedding(text)
            return np.array([embedding])
        except Exception as e:
            logger.error(f"Error generating query embedding: {str(e)}")
            raise

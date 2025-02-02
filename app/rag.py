import faiss
import numpy as np
from typing import List, Dict, Tuple
import os
from mistralai import Mistral
from .config import settings
from .embeddings import MistralEmbeddings
from .utils import setup_logger
import time
logger = setup_logger("rag")

class RAGSystem:
    def __init__(self, index_path: str = None):
        # Initialize Mistral clients
        self.embeddings = MistralEmbeddings()
        self.llm_client = Mistral(api_key=settings.MISTRAL_API_KEY)
        self.embedding_dim = settings.EMBEDDING_DIMENSION
        logger.info(f"Initialized RAG system with embedding dimension {self.embedding_dim}")
        
        # Initialize Faiss index and load chunks
        self.index_path = index_path or settings.VECTOR_STORE_PATH
        self.chunks_path = os.path.join(os.path.dirname(self.index_path), "chunks.npy")
        
        if os.path.exists(self.index_path) and os.path.exists(self.chunks_path):
            logger.info(f"Loading existing index from {self.index_path}")
            self.index = faiss.read_index(self.index_path)
            self.chunks = np.load(self.chunks_path, allow_pickle=True).tolist()
            logger.info(f"Loaded index with {self.index.ntotal} vectors and {len(self.chunks)} chunks")
        else:
            logger.info("Creating new FAISS index")
            self.index = faiss.IndexFlatL2(self.embedding_dim)
            self.chunks = []
        
    def add_documents(self, texts: List[str], filename: str):
        """Add new documents to the index"""
        try:
            if not texts:
                logger.warning("No texts provided to add_documents")
                return
                
            logger.info(f"Adding {len(texts)} documents to index")
            
            # Get embeddings for all texts at once
            embeddings = self.embeddings.embed_texts(texts)
            
            # Add to Faiss index
            self.index.add(embeddings)
            
            # Store text chunks with metadata
            self.chunks.extend([(text, filename) for text in texts])
            logger.info(f"Index now contains {self.index.ntotal} vectors")
            
            # Save both the index and chunks
            self.save_index()
        except Exception as e:
            logger.error(f"Error adding documents to index: {str(e)}")
            raise
        
    def retrieve_relevant_chunks(self, query: str, k: int = 3) -> List[Tuple[str, str]]:
        """Retrieve k most relevant chunks for the query"""
        try:
            if not self.chunks:
                logger.warning("No chunks available in the index")
                return []
                
            logger.debug(f"Retrieving {k} chunks for query: {query[:100]}...")
            
            # Get query embedding
            query_embedding = self.embeddings.embed_query(query)
            
            # Search the index
            distances, indices = self.index.search(query_embedding, k)
            logger.debug(f"Search distances: {distances[0]}")
            
            # Get the chunks and their metadata
            retrieved_chunks = []
            for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
                if idx < len(self.chunks):  # Ensure index is valid
                    chunk_text, filename = self.chunks[idx]
                    logger.debug(f"Retrieved chunk {i+1}/{k} with distance {dist:.4f} from {filename}")
                    retrieved_chunks.append((chunk_text, filename))
                
            return retrieved_chunks
        except Exception as e:
            logger.error(f"Error retrieving chunks: {str(e)}")
            raise
    
    def generate_response(self, query: str, max_tokens: int = None) -> Tuple[str, List[Dict[str, str]]]:
        """Generate a response using RAG"""
        try:
            # Retrieve relevant chunks
            relevant_chunks = self.retrieve_relevant_chunks(query)
            logger.info(f"Retrieved {len(relevant_chunks)} relevant chunks")
            
            if not relevant_chunks:
                return "I apologize, but I don't have any relevant information in my knowledge base to answer your question about that topic. Could you try asking about something else, or rephrase your question?", []
            
            # Construct prompt with context
            context = "\n\n".join([chunk[0] for chunk in relevant_chunks])
            prompt = f"""Context information is below.
---------------------
{context}
---------------------
Given the context information and not prior knowledge, answer the query.
Query: {query}
Answer: """
            
            logger.debug(f"Generated prompt with length {len(prompt)}")
            time.sleep(2)  # Rate limiting
            try:
                # Generate response using Mistral chat API
                messages = [{"role": "user", "content": prompt}]
                chat_response = self.llm_client.chat.complete(
                    model=settings.LLM_MODEL,
                    messages=messages
                )
                
                answer = chat_response.choices[0].message.content
                logger.info("Generated response successfully")
                
                # Prepare source information
                sources = [{"filename": filename, "text": text[:200]} for text, filename in relevant_chunks]
                
                return answer, sources
                
            except Exception as e:
                logger.error(f"Error in chat completion: {str(e)}")
                raise
                
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise
    
    def save_index(self):
        """Save both the Faiss index and chunks to disk"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
            
            # Save the Faiss index
            logger.info(f"Saving index with {self.index.ntotal} vectors to {self.index_path}")
            faiss.write_index(self.index, self.index_path)
            
            # Save the chunks
            logger.info(f"Saving {len(self.chunks)} chunks to {self.chunks_path}")
            np.save(self.chunks_path, np.array(self.chunks, dtype=object))
            
            logger.info("Index and chunks saved successfully")
        except Exception as e:
            logger.error(f"Error saving index and chunks: {str(e)}")
            raise

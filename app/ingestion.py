import markdown
from typing import List, Dict
import re
import os
import traceback
from pathlib import Path
from .config import settings
from .utils import setup_logger

logger = setup_logger("ingestion")

class DocumentProcessor:
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
        logger.info(f"Initialized DocumentProcessor with chunk_size={self.chunk_size}, chunk_overlap={self.chunk_overlap}")
        
    def preprocess_markdown(self, content: str) -> str:
        """Convert markdown to plain text and clean it"""
        try:
            logger.debug(f"Starting markdown preprocessing. Content length: {len(content)}")
            
            # Convert markdown to HTML
            html = markdown.markdown(content)
            logger.debug(f"Converted to HTML. Length: {len(html)}")
            
            # Remove HTML tags
            text = re.sub('<[^<]+?>', '', html)
            logger.debug(f"Removed HTML tags. Length: {len(text)}")
            
            # Clean up whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            logger.debug(f"Cleaned whitespace. Final length: {len(text)}")
            return text
        except Exception as e:
            logger.error(f"Error in preprocess_markdown: {str(e)}\n{traceback.format_exc()}")
            raise

    def split_by_sentences(self, text: str) -> List[str]:
        """Split text into sentences using regex"""
        # Pattern matches sentence endings (., !, ?) followed by spaces and capital letters
        pattern = r'(?<=[.!?])\s+(?=[A-Z])'
        sentences = re.split(pattern, text)
        return [s.strip() for s in sentences if s.strip()]

    def create_chunks(self, text: str) -> List[Dict[str, str]]:
        """Split text into overlapping chunks with metadata"""
        try:
            logger.debug(f"Starting text chunking. Text length: {len(text)}")
            chunks = []
            sentences = self.split_by_sentences(text)
            current_chunk = []
            current_size = 0
            
            for sentence in sentences:
                sentence_size = len(sentence)
                
                # If adding this sentence would exceed chunk_size, finalize current chunk
                if current_size + sentence_size > self.chunk_size and current_chunk:
                    # Join sentences and add metadata
                    chunk_text = ' '.join(current_chunk)
                    chunks.append({
                        'content': chunk_text,
                        'size': len(chunk_text),
                        'num_sentences': len(current_chunk)
                    })
                    
                    # Start new chunk with overlap
                    # Take the last few sentences that fit within overlap_size
                    overlap_size = 0
                    overlap_sentences = []
                    for s in reversed(current_chunk):
                        if overlap_size + len(s) <= self.chunk_overlap:
                            overlap_sentences.insert(0, s)
                            overlap_size += len(s)
                        else:
                            break
                    
                    current_chunk = overlap_sentences
                    current_size = overlap_size
                
                current_chunk.append(sentence)
                current_size += sentence_size
            
            # Add the last chunk if it exists
            if current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunks.append({
                    'content': chunk_text,
                    'size': len(chunk_text),
                    'num_sentences': len(current_chunk)
                })
            
            logger.info(f"Created {len(chunks)} chunks")
            for i, chunk in enumerate(chunks):
                logger.debug(f"Chunk {i+1}: {chunk['size']} chars, {chunk['num_sentences']} sentences")
            
            return chunks
        except Exception as e:
            logger.error(f"Error in create_chunks: {str(e)}\n{traceback.format_exc()}")
            raise
    
    def find_markdown_files(self, directory: str = None) -> List[Path]:
        """Recursively find all .md files in the given directory"""
        directory = directory or settings.DOCS_DIRECTORY
        logger.info(f"Searching for markdown files in: {directory}")
        
        markdown_files = []
        try:
            for path in Path(directory).rglob("*.md"):
                markdown_files.append(path)
                logger.debug(f"Found markdown file: {path}")
        except Exception as e:
            logger.error(f"Error searching for markdown files: {str(e)}\n{traceback.format_exc()}")
            raise
            
        logger.info(f"Found {len(markdown_files)} markdown files")
        return markdown_files
    
    def process_file(self, file_path: Path, rag_system) -> int:
        """Process a single markdown file and add it to the RAG system"""
        logger.info(f"Processing file: {file_path}")
        try:
            # Read file with explicit UTF-8 encoding and error handling
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                logger.debug(f"Successfully read file. Content length: {len(content)}")
            except UnicodeDecodeError:
                logger.warning(f"UTF-8 decode failed, trying with error handling")
                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                    content = f.read()
                logger.debug(f"Read file with error handling. Content length: {len(content)}")
            
            # Add file path as metadata to help with source tracking
            processed_content = f"Source: {file_path}\n\n{content}"
            logger.debug(f"Added source metadata. Total content length: {len(processed_content)}")
            
            # Process and index the content
            chunks = self.process_and_index(processed_content, rag_system)
            
            logger.info(f"Successfully processed {file_path} into {len(chunks)} chunks")
            return len(chunks)
        except Exception as e:
            error_msg = f"Error processing file {file_path}: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            raise Exception(error_msg)
    
    def batch_process_directory(self, directory: str = None, rag_system = None) -> dict:
        """Process all markdown files in a directory and its subdirectories"""
        logger.info("Starting batch processing of documentation")
        directory = directory or settings.DOCS_DIRECTORY
        
        try:
            markdown_files = self.find_markdown_files(directory)
            results = {
                "total_files": len(markdown_files),
                "processed_files": 0,
                "total_chunks": 0,
                "failed_files": []
            }
            
            logger.info(f"Found {len(markdown_files)} files to process")
            
            for file_path in markdown_files:
                try:
                    logger.info(f"Processing [{results['processed_files'] + 1}/{results['total_files']}]: {file_path}")
                    num_chunks = self.process_file(file_path, rag_system)
                    results["processed_files"] += 1
                    results["total_chunks"] += num_chunks
                    logger.info(f"Successfully processed {file_path} into {num_chunks} chunks")
                except Exception as e:
                    error_msg = f"Failed to process {file_path}: {str(e)}"
                    logger.error(error_msg)
                    results["failed_files"].append({"path": str(file_path), "error": str(e)})
            
            logger.info("Batch processing completed:")
            logger.info(f"- Processed {results['processed_files']}/{results['total_files']} files")
            logger.info(f"- Generated {results['total_chunks']} total chunks")
            logger.info(f"- Failed files: {len(results['failed_files'])}")
            if results["failed_files"]:
                logger.info("Failed files details:")
                for failed in results["failed_files"]:
                    logger.info(f"  - {failed['path']}: {failed['error']}")
            
            return results
        except Exception as e:
            error_msg = f"Error during batch processing: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            raise Exception(error_msg)
    
    def process_and_index(self, content: str, rag_system=None) -> List[str]:
        """Process document and add to RAG system"""
        try:
            logger.debug(f"Starting document processing. Content length: {len(content)}")
            
            # Preprocess the markdown content
            clean_text = self.preprocess_markdown(content)
            logger.debug(f"Preprocessed text length: {len(clean_text)}")
            
            # Create chunks
            chunks_with_metadata = self.create_chunks(clean_text)
            chunks = [chunk['content'] for chunk in chunks_with_metadata]
            logger.debug(f"Created {len(chunks)} chunks")
            
            # If RAG system is provided, add chunks to it
            if rag_system is not None:
                logger.info(f"Adding {len(chunks)} chunks to RAG system")
                rag_system.add_documents(chunks)
            
            return chunks
        except Exception as e:
            error_msg = f"Error in process_and_index: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            raise Exception(error_msg)

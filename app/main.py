from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import os
from pathlib import Path
from .rag import RAGSystem
from .config import settings
from .utils import setup_logger

logger = setup_logger("api")

app = FastAPI()
rag_system = RAGSystem()

class ChatRequest(BaseModel):
    query: str
    max_tokens: int = None

class Source(BaseModel):
    filename: str
    text: str

class ChatResponse(BaseModel):
    answer: str
    sources: List[Source]

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat endpoint that uses RAG to generate responses"""
    try:
        logger.info(f"Received chat request: {request.query[:100]}...")
        answer, sources = rag_system.generate_response(request.query, request.max_tokens)
        return ChatResponse(answer=answer, sources=sources)
        
    except Exception as e:
        logger.error(f"Error generating chat response: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process-documentation")
async def process_documentation():
    """Process all markdown files in the documentation directory"""
    try:
        docs_dir = Path(settings.DOCS_DIRECTORY)
        logger.info(f"Documentation directory: {docs_dir}")
        
        if not docs_dir.exists():
            raise HTTPException(status_code=404, detail=f"Documentation directory not found: {docs_dir}")
        
        # Walk through all files in docs directory
        results = []
        
        for file_path in docs_dir.rglob("*.md"):
            relative_path = file_path.relative_to(docs_dir)
            logger.info(f"Processing file: {relative_path}")
            
            # Read the markdown file
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                if not content:
                    logger.warning(f"File {relative_path} is empty. Skipping.")
                    continue
            
            # Split into chunks with overlap
            chunk_size = settings.CHUNK_SIZE
            overlap = settings.CHUNK_OVERLAP
            
            # Split content into lines first
            lines = content.split('\n')
            chunks = []
            
            for i in range(0, len(lines), chunk_size - overlap):
                # Get chunk lines with overlap
                chunk_lines = lines[i:i + chunk_size]
                chunk_text = '\n'.join(chunk_lines)
                chunks.append(chunk_text)
            
            # Add chunks to RAG system
            rag_system.add_documents(chunks, str(relative_path))
            results.append({"file": str(relative_path), "chunks": len(chunks)})
        
        # Save the updated index
        logger.info("Saving updated vector index")
        rag_system.save_index()
        
        logger.info("Documentation processing completed successfully")
        return {
            "message": "Documentation processed successfully",
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Error processing documentation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

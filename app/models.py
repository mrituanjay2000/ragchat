from pydantic import BaseModel
from typing import List, Optional, Dict

class ChatRequest(BaseModel):
    query: str
    max_tokens: Optional[int] = 512

class ChatResponse(BaseModel):
    answer: str
    sources: List[str]

class IngestResponse(BaseModel):
    message: str
    num_chunks: int

class ProcessingResults(BaseModel):
    total_files: int
    processed_files: int
    total_chunks: int
    failed_files: List[str]

class DocumentationProcessingResponse(BaseModel):
    message: str
    results: ProcessingResults

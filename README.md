# RAG Chatbot with Mixtral and Faiss

A Retrieval-Augmented Generation (RAG) chatbot that uses Faiss for vector storage and Mixtral 8x7B for generating responses. The system processes Ubuntu documentation and provides accurate, context-aware responses to user queries.

## Features

- Document processing and chunking of Ubuntu documentation
- Vector embeddings using Mistral-Embed
- Efficient similarity search using Faiss
- Context-aware responses using Mixtral 8x7B
- FastAPI backend with Swagger UI

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
Create a `.env` file with (use env.example as a reference):
```
MIXTRAL_API_KEY=your_api_key_here
```

3. Run the application:
```bash
uvicorn app.main:app --reload
```

## API Endpoints

- POST `/chat`: Submit a query and get a context-aware response
- POST `/process-documentation`: Ingest new documentation into the vector store

## Project Structure

- `app/`: Main application directory
  - `main.py`: FastAPI application and endpoints
  - `ingestion.py`: Document processing and indexing
  - `rag.py`: RAG implementation with Faiss and Mixtral
  - `models.py`: Pydantic models for request/response
- `data/`: Directory for storing documentation and vector indices

from pydantic_settings import BaseSettings
from pathlib import Path
from functools import lru_cache
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get the project root directory
ROOT_DIR = Path(__file__).parent.parent

class Settings(BaseSettings):
    # API Configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    
    # Model Configuration
    MISTRAL_API_KEY: str = os.getenv("MISTRAL_API_KEY")
    EMBEDDING_MODEL: str = "mistral-embed"
    EMBEDDING_DIMENSION: int = 1024
    LLM_MODEL: str = "mistral-large-latest"
    
    # Vector Store Configuration
    VECTOR_STORE_PATH: str = str(ROOT_DIR / "vector_store" / "faiss_index")
    
    # Document Processing
    CHUNK_SIZE: int = 10
    CHUNK_OVERLAP: int = 2
    DOCS_DIRECTORY: str = str(ROOT_DIR / "demo_bot_data")
    
    # Generation Configuration
    MAX_NEW_TOKENS: int = 512
    TEMPERATURE: float = 0.7
    TOP_P: float = 0.9
    
    class Config:
        env_file = ROOT_DIR / ".env"
        env_file_encoding = "utf-8"

@lru_cache()
def get_settings() -> Settings:
    return Settings()

# Create a global settings instance
settings = get_settings()

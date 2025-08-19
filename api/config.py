"""
Configuration management for the Medical Graph Extraction API
"""
import os
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Neo4j Configuration
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password123"
    
    # Model Configuration (MLX-based)
    gemma_model: str = "google/gemma-3-4b-it"
    ner_model: str = "mlx-ner-medical"  # MLX-based NER model
    scispacy_model: str = "en_core_sci_lg"
    model_cache_dir: Optional[str] = "/app/models"
    
    # API Configuration
    api_port: int = 8000
    api_host: str = "0.0.0.0"
    max_request_size: str = "10MB"
    request_timeout: int = 300
    
    # Cache Configuration
    cache_size: int = 10000
    cache_ttl: int = 3600
    
    # Processing Configuration
    default_entity_threshold: float = 0.5
    default_max_tokens: int = 512
    max_workers: int = 2
    
    # Logging
    log_level: str = "INFO"
    
    model_config = {
        "env_file": ".env",
        "case_sensitive": False,
        "extra": "ignore"
    }


# Global settings instance
settings = Settings()
# backend/app/config.py

import os
from pathlib import Path
from typing import Optional

class Settings:
    """
    Enhanced application configuration with validation and defaults.
    """
    
    # API Keys
    GOOGLE_API_KEY: str = os.getenv(
        "GOOGLE_API_KEY",
        "AIzaSyAEesyJ5kz10ooIG5O3nWXqxiHxmkfDJBI"
    )
    
    TAVILY_API_KEY: str = os.getenv(
        "TAVILY_API_KEY", 
        "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"
    )
    
    # Database Configuration
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        "sqlite+aiosqlite:///./math_agent.db"
    )
    
    # LanceDB Configuration
    LANCEDB_PATH: str = os.getenv(
        "LANCEDB_PATH",
        "./data/lancedb_math"
    )
    
    LANCEDB_TABLE: str = os.getenv(
        "LANCEDB_TABLE",
        "math_qa"
    )
    
    LANCEDB_VECTOR_COLUMN: str = os.getenv(
        "LANCEDB_VECTOR_COLUMN",
        "embedding"
    )
    
    # Knowledge Base Configuration
    KB_THRESHOLD: float = float(os.getenv("KB_THRESHOLD", "0.75"))
    KB_TOP_K: int = int(os.getenv("KB_TOP_K", "5"))
    
    # Embedding Configuration
    EMBEDDING_MODEL: str = os.getenv(
        "EMBEDDING_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2"
    )
    EMBEDDING_DIMENSION: int = int(os.getenv("EMBEDDING_DIMENSION", "384"))
    
    # MCP Configuration
    MCP_URL: str = os.getenv(
        "MCP_URL",
        "https://mcp.tavily.com/mcp/"
    )
    
    MCP_MAX_RESULTS: int = int(os.getenv("MCP_MAX_RESULTS", "5"))
    MCP_TIMEOUT: int = int(os.getenv("MCP_TIMEOUT", "30"))
    
    # DSPy Configuration
    DSPY_MODEL: str = os.getenv("DSPY_MODEL", "gemini-1.5-flash")
    DSPY_TEMPERATURE: float = float(os.getenv("DSPY_TEMPERATURE", "0.2"))
    DSPY_MAX_TOKENS: int = int(os.getenv("DSPY_MAX_TOKENS", "1024"))
    
    # Application Configuration
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    
    # Performance Configuration
    MAX_RESPONSE_TIME: float = float(os.getenv("MAX_RESPONSE_TIME", "30.0"))
    CACHE_TTL: int = int(os.getenv("CACHE_TTL", "3600"))  # 1 hour
    
    # Educational Content Configuration
    STUDENT_LEVELS: list = ["beginner", "intermediate", "advanced"]
    DEFAULT_STUDENT_LEVEL: str = os.getenv("DEFAULT_STUDENT_LEVEL", "intermediate")
    
    # Guardrails Configuration
    ENABLE_INPUT_VALIDATION: bool = os.getenv("ENABLE_INPUT_VALIDATION", "True").lower() == "true"
    ENABLE_OUTPUT_VALIDATION: bool = os.getenv("ENABLE_OUTPUT_VALIDATION", "True").lower() == "true"
    ENABLE_CONTENT_FILTERING: bool = os.getenv("ENABLE_CONTENT_FILTERING", "True").lower() == "true"
    
    # Feedback Configuration
    ENABLE_FEEDBACK_LEARNING: bool = os.getenv("ENABLE_FEEDBACK_LEARNING", "True").lower() == "true"
    FEEDBACK_THRESHOLD: float = float(os.getenv("FEEDBACK_THRESHOLD", "4.0"))
    
    # File Paths
    BASE_DIR: Path = Path(__file__).parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    LOGS_DIR: Path = BASE_DIR / "logs"
    SCRIPTS_DIR: Path = BASE_DIR / "scripts"
    
    # Knowledge Base Files
    KB_PARQUET_PATH: str = os.getenv(
        "KB_PARQUET_PATH",
        str(BASE_DIR / "knowledge_base.parquet")
    )
    
    # Benchmark Configuration
    JEE_BENCHMARK_ENABLED: bool = os.getenv("JEE_BENCHMARK_ENABLED", "True").lower() == "true"
    BENCHMARK_OUTPUT_DIR: Path = BASE_DIR / "benchmark_results"
    
    def __post_init__(self):
        """Create necessary directories on initialization."""
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        self.BENCHMARK_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    def validate_config(self) -> dict:
        """
        Validate configuration and return validation results.
        """
        issues = []
        warnings = []
        
        # Validate API keys
        if not self.GOOGLE_API_KEY or self.GOOGLE_API_KEY == "your-google-api-key":
            issues.append("GOOGLE_API_KEY not properly configured")
        
        if not self.TAVILY_API_KEY or self.TAVILY_API_KEY == "your-tavily-api-key":
            warnings.append("TAVILY_API_KEY not configured - web search will be limited")
        
        # Validate paths
        try:
            Path(self.LANCEDB_PATH).parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            issues.append(f"Cannot create LanceDB directory: {e}")
        
        # Validate numeric configurations
        if self.KB_THRESHOLD < 0 or self.KB_THRESHOLD > 1:
            issues.append("KB_THRESHOLD must be between 0 and 1")
        
        if self.DSPY_TEMPERATURE < 0 or self.DSPY_TEMPERATURE > 2:
            warnings.append("DSPY_TEMPERATURE outside recommended range (0-2)")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings
        }
    
    def get_database_url(self) -> str:
        """Get the properly formatted database URL."""
        return self.DATABASE_URL
    
    def get_lancedb_config(self) -> dict:
        """Get LanceDB configuration as a dictionary."""
        return {
            "db_path": self.LANCEDB_PATH,
            "table_name": self.LANCEDB_TABLE,
            "vector_column": self.LANCEDB_VECTOR_COLUMN
        }
    
    def get_dspy_config(self) -> dict:
        """Get DSPy configuration as a dictionary."""
        return {
            "model": self.DSPY_MODEL,
            "api_key": self.GOOGLE_API_KEY,
            "temperature": self.DSPY_TEMPERATURE,
            "max_tokens": self.DSPY_MAX_TOKENS
        }
    
    def get_mcp_config(self) -> dict:
        """Get MCP configuration as a dictionary."""
        return {
            "url": self.MCP_URL,
            "api_key": self.TAVILY_API_KEY,
            "max_results": self.MCP_MAX_RESULTS,
            "timeout": self.MCP_TIMEOUT
        }

# Instantiate settings object
settings = Settings()

# Validate configuration on import
validation_result = settings.validate_config()
if not validation_result["valid"]:
    import logging
    logger = logging.getLogger(__name__)
    logger.error("❌ Configuration validation failed:")
    for issue in validation_result["issues"]:
        logger.error(f"  - {issue}")

if validation_result["warnings"]:
    import logging
    logger = logging.getLogger(__name__)
    for warning in validation_result["warnings"]:
        logger.warning(f"⚠️ {warning}")

# Create directories
settings.__post_init__()
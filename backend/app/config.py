# backend/app/config.py

import os

class Settings:
    """
    Application configuration loaded from environment variables.
    """
    # Google Gemini API key
    GOOGLE_API_KEY: str = os.getenv(
        "GOOGLE_API_KEY",
        "AIzaSyAEesyJ5kz10ooIG5O3nWXqxiHxmkfDJBI"
    )
    # Tavily MCP URL & key
    MCP_URL: str = os.getenv(
        "MCP_URL",
        "https://mcp.tavily.com/mcp/"
    )
    TAVILY_API_KEY: str = os.getenv(
        "TAVILY_API_KEY",
        "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"
    )
    # LanceDB path
    LANCEDB_PATH: str = os.getenv(
        "LANCEDB_PATH",
        "./lancedb_math"
    )

# Instantiate a single settings object for import everywhere
settings = Settings()

"""
Configuration management for the Cold Email Generator application.
"""
import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Application configuration class."""

    # API Configuration
    GROQ_API_KEY: str = os.getenv("API_KEY", "")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0"))

    # Application Settings
    APP_TITLE: str = "Student Resume → Cold Email Generator"
    APP_ICON: str = "📧"
    PAGE_LAYOUT: str = "wide"

    # User Information (can be overridden by user input in future)
    DEFAULT_USER_NAME: str = os.getenv("USER_NAME", "Srivatsav")
    DEFAULT_UNIVERSITY: str = os.getenv("UNIVERSITY", "Vellore Institute of Technology, Chennai")
    DEFAULT_YEAR: str = os.getenv("YEAR", "4th year")

    # File Upload Settings
    MAX_UPLOAD_SIZE_MB: int = 10
    ALLOWED_RESUME_FORMATS: list = ["pdf", "docx"]

    # Job Matching Settings
    MIN_SKILL_MATCH_THRESHOLD: float = 0.3
    MAX_CLOSE_MATCHES: int = 3

    # Email Generation Settings
    EMAIL_TONES: list = ["professional", "enthusiastic", "casual"]
    DEFAULT_EMAIL_TONE: str = "professional"

    # Caching Settings
    ENABLE_CACHE: bool = os.getenv("ENABLE_CACHE", "true").lower() == "true"
    CACHE_TTL_SECONDS: int = int(os.getenv("CACHE_TTL_SECONDS", "3600"))

    # Logging Settings
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: Optional[str] = os.getenv("LOG_FILE", None)

    @classmethod
    def validate(cls) -> bool:
        """Validate required configuration."""
        if not cls.GROQ_API_KEY:
            raise ValueError("API_KEY environment variable is required")
        return True


# Validate configuration on import
Config.validate()

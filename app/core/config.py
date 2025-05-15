"""
Config
-----
Modul ini berisi konfigurasi aplikasi.
"""

import os
from pydantic import BaseSettings
from typing import Optional, Dict, Any, List


class Settings(BaseSettings):
    """
    Konfigurasi aplikasi.
    """

    APP_NAME: str = "IR-System-BE"
    APP_VERSION: str = "0.1.0"

    # Lokasi data
    DATA_DIR: str = "app/data"
    UPLOAD_DIR: str = "app/data/uploads"

    # Word2Vec
    WORD2VEC_MODEL_PATH: Optional[str] = None

    # Stopwords dan Stemming
    STOPWORDS_FILE: str = "app/data/stopwords.txt"
    USE_STEMMING_DEFAULT: bool = True
    USE_STOPWORD_REMOVAL_DEFAULT: bool = True

    # Logging
    LOG_LEVEL: str = "INFO"

    class Config:
        env_file = ".env"


settings = Settings()

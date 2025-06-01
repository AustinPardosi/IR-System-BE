from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging
import nltk
import uvicorn
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global Query Expansion Service
qe_service = None

app = FastAPI(
    title="IR-System-BE",
    description="Backend for Information Retrieval System with Word2Vec Query Expansion",
    version="0.1.0",
)

# Konfigurasi CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Untuk development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import router (akan diimplementasikan nanti)
from app.routers import documents, retrieval, api

# Daftarkan router
app.include_router(documents.router)
app.include_router(retrieval.router)
app.include_router(api.router)


@app.on_event("startup")
async def startup_event():
    """Download NLTK data dan training Word2Vec model saat startup."""
    global qe_service

    logger.info("=== Starting IR-System-BE ===")

    # 1. Download NLTK data
    logger.info("Downloading NLTK data...")
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    nltk.download("stopwords", quiet=True)
    logger.info("✅ NLTK data downloaded successfully")

    # 2. Training Word2Vec model
    try:
        logger.info("🚀 Training Word2Vec model...")

        # Import di sini setelah NLTK data didownload
        from app.services.query_expansion_service import QueryExpansionService

        # Coba beberapa path yang mungkin
        possible_paths = [
            "app/data/parsing/cisi.all",
            "IRTestCollection/cisi.all",
            "/app/data/parsing/cisi.all",
            "/app/IRTestCollection/cisi.all",
        ]

        document_path = None
        for path in possible_paths:
            if os.path.exists(path):
                document_path = path
                logger.info(f"Found CISI dataset at: {path}")
                break

        if document_path:
            qe_service = await QueryExpansionService.create(document_path)
            logger.info("✅ Word2Vec model trained and ready!")
        else:
            logger.warning("⚠️ CISI dataset not found in any expected location")
            qe_service = None

    except Exception as e:
        logger.error(f"❌ Error training Word2Vec: {e}")
        qe_service = None

    logger.info("=== IR-System-BE ready to serve! ===")


def get_query_expansion_service():
    """Dependency untuk mendapatkan QueryExpansionService yang sudah dilatih."""
    if qe_service is None:
        raise HTTPException(
            status_code=503,
            detail="QueryExpansionService not available. Model training failed or CISI dataset not found.",
        )
    return qe_service


@app.get("/")
async def read_root():
    return {"message": "IR-System-BE API is running!"}


@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "IR-System-BE"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)

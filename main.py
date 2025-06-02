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

# Import router
from app.routers import documents, retrieval, query

# Daftarkan router dengan prefix /api
app.include_router(documents.router, prefix="/api")
app.include_router(retrieval.router, prefix="/api")
app.include_router(query.router, prefix="/api")


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

        # Gunakan parsing_docs.json sebagai dataset training
        document_path = "app/data/parsing/parsing_docs.json"

        if os.path.exists(document_path):
            logger.info(f"Found parsing documents at: {document_path}")
            qe_service = await QueryExpansionService.create(document_path)
            logger.info("✅ Word2Vec model trained and ready!")
        else:
            logger.warning(
                "⚠️ parsing_docs.json not found at app/data/parsing/parsing_docs.json"
            )
            qe_service = None

    except Exception as e:
        logger.error(f"❌ Error training Word2Vec: {e}")
        qe_service = None

    logger.info("=== IR-System-BE ready to serve on PORT 8080! ===")


def get_query_expansion_service():
    """Dependency untuk mendapatkan QueryExpansionService yang sudah dilatih."""
    if qe_service is None:
        raise HTTPException(
            status_code=503,
            detail="QueryExpansionService not available. Model training failed or parsing_docs.json not found.",
        )
    return qe_service


@app.get("/api")
async def api_status():
    """Main API endpoint - Health check and system status."""
    return {
        "message": "IR-System-BE API is running!",
        "status": "active",
        "port": 8080,
        "version": "0.1.0",
        "endpoints": {
            "documents": "/api/documents/*",
            "retrieval": "/api/retrieval/*",
            "query": "/api/query/*",
        },
    }


@app.get("/")
async def read_root():
    return {"message": "IR-System-BE API is running! Visit /api for API status"}


@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "IR-System-BE", "port": 8080}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging
import nltk
import uvicorn
from app.services.query_expansion_service import QueryExpansionService

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Query Expansion Service
qe_service: QueryExpansionService = None

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
    nltk.download("punkt")
    nltk.download("punkt_tab")
    nltk.download("stopwords")

    global qe_service
    qe_service = await QueryExpansionService.create(
        document_path="IRTestCollection/cisi.all"
    )
    logger.info("QueryExpansionService initialized on startup")


@app.get("/")
async def root():
    return {"message": "Selamat datang di IR-System-BE"}


@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "IR-System-BE"}


def get_query_expansion_service():
    if not qe_service or not qe_service.model:
        raise HTTPException(
            status_code=500, detail="QueryExpansionService not initialized."
        )
    return qe_service


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
import uvicorn

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
from app.routers import documents, retrieval

# Daftarkan router
app.include_router(documents.router)
app.include_router(retrieval.router)


@app.get("/")
async def root():
    return {"message": "Selamat datang di IR-System-BE"}


@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "IR-System-BE"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

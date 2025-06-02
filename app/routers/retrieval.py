"""
Retrieval Router
-------------
Router untuk endpoint terkait retrieval:
1. Interactive query
2. Batch query
3. Inverted file
4. Model status
"""

from fastapi import (
    APIRouter,
    UploadFile,
    File,
    Form,
    HTTPException,
    Query,
)
from typing import List, Dict, Any, Optional
import logging
import json
import os

from app.services.retrieval_service import RetrievalService
from app.services.query_expansion_service import QueryExpansionService
from app.models.query_models import (
    InteractiveQueryInput,
    BatchQueryInput,
    RetrievalResult,
)

logger = logging.getLogger(__name__)

# Simple cache untuk inverted file
_cached_inverted_file = None
_cache_key = None

router = APIRouter(
    prefix="/retrieval",
    tags=["retrieval"],
    responses={404: {"description": "Not found"}},
)


@router.post("/query/interactive", response_model=RetrievalResult)
async def interactive_query(query_input: InteractiveQueryInput):
    """
    Endpoint untuk query interaktif.
    """
    # Placeholder for implementation
    return {"message": "Interactive query placeholder"}


@router.post("/query/batch")
async def batch_query(
    query_file: UploadFile = File(...),
    relevance_judgement_file: Optional[UploadFile] = File(None),
    use_stemming: bool = Form(True),
    use_stopword_removal: bool = Form(True),
    query_expansion_threshold: float = Form(0.7),
):
    """
    Endpoint untuk batch query.
    """
    # Placeholder for implementation
    return {"message": "Batch query placeholder"}


@router.get("/inverted-file")
async def get_inverted_file(
    use_stemming: bool = Query(...),
    use_stopword_removal: bool = Query(...),
    tf_raw: bool = Query(...),
    tf_log: bool = Query(...),
    tf_binary: bool = Query(...),
    tf_augmented: bool = Query(...),
    use_idf: bool = Query(...),
    use_normalization: bool = Query(...),
):
    """
    Endpoint untuk mendapatkan inverted file dari dokumen yang tersimpan di parsing_docs.json.
    Menggunakan caching untuk meningkatkan performa.

    Parameters (semua parameter wajib diisi):
    - use_stemming: Gunakan stemming atau tidak
    - use_stopword_removal: Hapus stopwords atau tidak
    - tf_raw: Gunakan raw term frequency
    - tf_log: Gunakan log term frequency
    - tf_binary: Gunakan binary term frequency
    - tf_augmented: Gunakan augmented term frequency
    - use_idf: Gunakan inverse document frequency
    - use_normalization: Gunakan normalisasi
    """
    global _cached_inverted_file, _cache_key

    try:
        # Buat cache key dari semua parameter
        current_cache_key = f"{use_stemming}_{use_stopword_removal}_{tf_raw}_{tf_log}_{tf_binary}_{tf_augmented}_{use_idf}_{use_normalization}"

        # Cek apakah sudah ada di cache dengan parameter yang sama
        if _cached_inverted_file and _cache_key == current_cache_key:
            logger.info("Returning cached inverted file")
            return _cached_inverted_file

        logger.info("Generating new inverted file...")

        # Baca file JSON
        json_path = os.path.join("app", "data", "parsing", "parsing_docs.json")
        with open(json_path, "r", encoding="utf-8") as f:
            documents = json.load(f)

        # Buat weighting method dari input user
        document_weighting_method = {
            "tf_raw": tf_raw,
            "tf_log": tf_log,
            "tf_binary": tf_binary,
            "tf_augmented": tf_augmented,
            "use_idf": use_idf,
            "use_normalization": use_normalization,
        }

        # Buat inverted file
        retrieval_service = RetrievalService()
        inverted_file = await retrieval_service.create_inverted_file(
            documents, use_stemming, use_stopword_removal, document_weighting_method
        )

        result = {
            "status": "success",
            "total_documents": len(documents),
            "total_terms": len(inverted_file),
            "inverted_file": inverted_file,
            "parameters": {
                "use_stemming": use_stemming,
                "use_stopword_removal": use_stopword_removal,
                "document_weighting_method": document_weighting_method,
            },
            "cached": False,
        }

        # Simpan ke cache
        _cached_inverted_file = result
        _cache_key = current_cache_key

        logger.info(
            f"Inverted file generated and cached with {len(inverted_file)} terms"
        )
        return result

    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="File parsing_docs.json tidak ditemukan. Pastikan file sudah ada di app/data/parsing/",
        )
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=500,
            detail="Error membaca file JSON. Pastikan format file valid.",
        )
    except Exception as e:
        logger.exception("Error saat membuat inverted file")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/inverted-file/cache")
async def clear_inverted_file_cache():
    """
    Endpoint untuk menghapus cache inverted file.
    """
    global _cached_inverted_file, _cache_key

    _cached_inverted_file = None
    _cache_key = None

    logger.info("Inverted file cache cleared")
    return {"status": "success", "message": "Cache cleared successfully"}


@router.get("/model-status")
async def get_model_status():
    """
    Endpoint untuk mengecek status Word2Vec model.
    """
    try:
        from main import get_query_expansion_service

        qe_service = get_query_expansion_service()

        return {
            "status": "ready",
            "model_trained": True,
            "vocabulary_size": (
                len(qe_service.model.wv.key_to_index) if qe_service.model else 0
            ),
            "message": "Word2Vec model is trained and ready for query expansion",
        }
    except HTTPException as e:
        return {
            "status": "not_ready",
            "model_trained": False,
            "vocabulary_size": 0,
            "message": str(e.detail),
        }
    except Exception as e:
        return {
            "status": "error",
            "model_trained": False,
            "vocabulary_size": 0,
            "message": f"Error checking model status: {str(e)}",
        }

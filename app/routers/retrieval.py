"""
Retrieval Router - Endpoint untuk operasi retrieval dokumen
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
    DocumentRetrievalInput,
    DocumentRetrievalResult,
    DocumentRetrievalInputSimple,
    BatchRetrievalInput,
    BatchRetrievalResult,
)
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Cache untuk inverted file
_cached_inverted_file = None
_cache_key = None
_inverted_file_cache = {"inverted_file": None, "parameters": None, "is_cached": False}

router = APIRouter(
    prefix="/retrieval",
    tags=["retrieval"],
    responses={404: {"description": "Not found"}},
)


class DocumentWeightResponse(BaseModel):
    status: str
    document_id: str
    weights: Dict[str, float]
    total_terms: int
    message: str


@router.post("/query/interactive", response_model=RetrievalResult)
async def interactive_query(query_input: InteractiveQueryInput):
    return {"message": "Interactive query placeholder"}


@router.post("/query/batch")
async def batch_query(
    query_file: UploadFile = File(...),
    relevance_judgement_file: Optional[UploadFile] = File(None),
    use_stemming: bool = Form(True),
    use_stopword_removal: bool = Form(True),
    query_expansion_threshold: float = Form(0.7),
):
    return {"message": "Batch query placeholder"}


@router.post("/retrieve", response_model=DocumentRetrievalResult)
async def retrieve_documents(request: DocumentRetrievalInputSimple):
    """
    Endpoint untuk melakukan retrieval dokumen menggunakan cached inverted file.
    Pastikan sudah memanggil GET /inverted-file terlebih dahulu.
    """
    global _inverted_file_cache

    try:
        if (
            not _inverted_file_cache["is_cached"]
            or _inverted_file_cache["inverted_file"] is None
        ):
            raise HTTPException(
                status_code=400,
                detail="Inverted file cache tidak tersedia. Silakan panggil GET /api/retrieval/inverted-file terlebih dahulu.",
            )

        logger.info(f"Processing document retrieval for query: '{request.query}'")
        retrieval_service = RetrievalService()
        cached_inverted_file = _inverted_file_cache["inverted_file"]

        similarity_results, average_precision = (
            await retrieval_service.retrieve_document_single_query(
                query=request.query,
                inverted_file=cached_inverted_file,
                weighting_method=request.weighting_method,
                relevant_doc=request.relevant_doc,
            )
        )

        ranked_docs = list(similarity_results.keys()) if similarity_results else []
        logger.info(
            f"Retrieved {len(ranked_docs)} documents with AP: {average_precision}"
        )

        return DocumentRetrievalResult(
            status="success",
            ranked_documents=ranked_docs,
            average_precision=average_precision,
            total_retrieved=len(ranked_docs),
            query_used=request.query,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error saat melakukan document retrieval")
        raise HTTPException(
            status_code=500, detail=f"Error during document retrieval: {str(e)}"
        )


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
    Inverted file akan disimpan ke global cache untuk digunakan oleh endpoint retrieve.
    """
    global _cached_inverted_file, _cache_key, _inverted_file_cache

    try:
        current_cache_key = f"{use_stemming}_{use_stopword_removal}_{tf_raw}_{tf_log}_{tf_binary}_{tf_augmented}_{use_idf}_{use_normalization}"

        if _cached_inverted_file and _cache_key == current_cache_key:
            logger.info("Returning cached inverted file")
            return _cached_inverted_file

        logger.info("Generating new inverted file...")
        json_path = os.path.join("app", "data", "parsing", "parsing_docs.json")
        with open(json_path, "r", encoding="utf-8") as f:
            documents = json.load(f)

        document_weighting_method = {
            "tf_raw": tf_raw,
            "tf_log": tf_log,
            "tf_binary": tf_binary,
            "tf_augmented": tf_augmented,
            "use_idf": use_idf,
            "use_normalization": use_normalization,
        }

        retrieval_service = RetrievalService()
        inverted_file = await retrieval_service.create_inverted_file(
            documents, use_stemming, use_stopword_removal, document_weighting_method
        )

        _inverted_file_cache["inverted_file"] = inverted_file
        _inverted_file_cache["parameters"] = {
            "use_stemming": use_stemming,
            "use_stopword_removal": use_stopword_removal,
            "document_weighting_method": document_weighting_method,
        }
        _inverted_file_cache["is_cached"] = True

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
            "cache_info": "Inverted file berhasil disimpan ke cache untuk endpoint retrieve",
        }

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


@router.get("/cache/status")
async def get_cache_status():
    """Endpoint untuk mengecek status cache inverted file."""
    global _inverted_file_cache

    if (
        _inverted_file_cache["is_cached"]
        and _inverted_file_cache["inverted_file"] is not None
    ):
        total_terms = len(_inverted_file_cache["inverted_file"])
        return {
            "status": "Cache tersedia",
            "is_cached": True,
            "total_terms": total_terms,
            "parameters": _inverted_file_cache["parameters"],
            "message": "Inverted file sudah tersedia di cache. Anda bisa menggunakan endpoint /retrieve.",
        }
    else:
        return {
            "status": "Cache tidak tersedia",
            "is_cached": False,
            "total_terms": 0,
            "parameters": None,
            "message": "Silakan panggil GET /inverted-file terlebih dahulu untuk generate cache.",
        }


@router.delete("/inverted-file/cache")
async def clear_inverted_file_cache():
    """Endpoint untuk menghapus cache inverted file."""
    global _cached_inverted_file, _cache_key, _inverted_file_cache

    _cached_inverted_file = None
    _cache_key = None
    _inverted_file_cache["inverted_file"] = None
    _inverted_file_cache["parameters"] = None
    _inverted_file_cache["is_cached"] = False

    logger.info("All caches cleared")
    return {
        "status": "success",
        "message": "Semua cache berhasil dihapus",
    }


@router.get("/model-status")
async def get_model_status():
    """Endpoint untuk mengecek status Word2Vec model."""
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


@router.post("/retrieve-batch", response_model=BatchRetrievalResult)
async def batch_retrieve_documents(request: BatchRetrievalInput):
    """
    Endpoint untuk batch retrieval menggunakan cached inverted file.
    Pastikan sudah memanggil GET /inverted-file terlebih dahulu.
    """
    global _inverted_file_cache

    try:
        if (
            not _inverted_file_cache["is_cached"]
            or _inverted_file_cache["inverted_file"] is None
        ):
            raise HTTPException(
                status_code=400,
                detail="Inverted file cache tidak tersedia. Silakan panggil GET /api/retrieval/inverted-file terlebih dahulu.",
            )

        logger.info(f"Processing batch retrieval using cached inverted file")
        logger.info(f"Query file: {request.query_file}")
        logger.info(f"Relevant doc file: {request.relevant_doc_filename}")

        if not os.path.exists(request.query_file):
            raise HTTPException(
                status_code=400,
                detail=f"Query file tidak ditemukan: {request.query_file}",
            )

        if not os.path.exists(request.relevant_doc_filename):
            raise HTTPException(
                status_code=400,
                detail=f"Relevant document file tidak ditemukan: {request.relevant_doc_filename}",
            )

        retrieval_service = RetrievalService()
        cached_inverted_file = _inverted_file_cache["inverted_file"]

        batch_results, mean_average_precision = (
            await retrieval_service.retrieve_document_batch_query(
                filename=request.query_file,
                inverted_file=cached_inverted_file,
                weighting_method=request.weighting_method,
                relevant_doc_filename=request.relevant_doc_filename,
            )
        )

        query_results = []
        for i, (similarity_results, average_precision) in enumerate(batch_results):
            query_results.append(
                {
                    "query_index": i + 1,
                    "average_precision": average_precision,
                    "total_retrieved": len(similarity_results),
                    "top_documents": (
                        list(similarity_results.keys())[:10]
                        if similarity_results
                        else []
                    ),
                }
            )

        logger.info(
            f"Batch retrieval completed: {len(batch_results)} queries processed, MAP: {mean_average_precision:.4f}"
        )

        return BatchRetrievalResult(
            status="success",
            total_queries=len(batch_results),
            mean_average_precision=mean_average_precision,
            query_results=query_results,
            processing_info={
                "query_file_path": request.query_file,
                "relevant_doc_file_path": request.relevant_doc_filename,
                "weighting_method": request.weighting_method,
                "cache_terms_count": len(cached_inverted_file),
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error saat melakukan batch retrieval")
        raise HTTPException(
            status_code=500, detail=f"Error during batch retrieval: {str(e)}"
        )


@router.get("/document-weights/{document_id}", response_model=DocumentWeightResponse)
async def get_document_weights(document_id: str):
    """
    Endpoint untuk mengambil bobot setiap term dalam dokumen tertentu.
    Pastikan sudah memanggil GET /inverted-file terlebih dahulu.
    """
    global _inverted_file_cache

    try:
        if (
            not _inverted_file_cache["is_cached"]
            or _inverted_file_cache["inverted_file"] is None
        ):
            raise HTTPException(
                status_code=400,
                detail="Inverted file cache tidak tersedia. Silakan panggil GET /api/retrieval/inverted-file terlebih dahulu.",
            )

        logger.info(f"Getting weights for document ID: {document_id}")
        retrieval_service = RetrievalService()
        cached_inverted_file = _inverted_file_cache["inverted_file"]

        weights = await retrieval_service.get_weight_by_document_id(
            document_id=document_id, inverted_file=cached_inverted_file
        )
        if not weights:
            raise HTTPException(
                status_code=404,
                detail=f"Document dengan ID '{document_id}' tidak ditemukan dalam inverted file atau tidak memiliki term apapun.",
            )

        logger.info(f"Found {len(weights)} terms for document {document_id}")

        return DocumentWeightResponse(
            status="success",
            document_id=document_id,
            weights=weights,
            total_terms=len(weights),
            message=f"Berhasil mengambil bobot {len(weights)} term untuk dokumen {document_id}",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error saat mengambil bobot untuk dokumen {document_id}")
        raise HTTPException(
            status_code=500, detail=f"Error getting document weights: {str(e)}"
        )

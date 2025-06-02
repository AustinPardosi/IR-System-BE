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
    DocumentRetrievalInput,
    DocumentRetrievalResult,
    DocumentRetrievalInputSimple,
    BatchRetrievalInput,
    BatchRetrievalResult,
)

logger = logging.getLogger(__name__)

# Simple cache untuk inverted file
_cached_inverted_file = None
_cache_key = None

# Global cache untuk inverted file data
_inverted_file_cache = {"inverted_file": None, "parameters": None, "is_cached": False}

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


@router.post("/retrieve", response_model=DocumentRetrievalResult)
async def retrieve_documents(request: DocumentRetrievalInputSimple):
    """
    Endpoint untuk melakukan retrieval dokumen menggunakan cached inverted file.

    ⭐ PENTING: Endpoint ini menggunakan inverted file yang sudah di-cache dari endpoint GET /inverted-file.
    Pastikan Anda sudah memanggil GET /inverted-file terlebih dahulu sebelum menggunakan endpoint ini.

    **Contoh Request Body (lebih simple, tanpa inverted_file):**
    ```json
    {
        "query": "information retrieval test",
        "weighting_method": {
            "tf_raw": true,
            "tf_log": false,
            "tf_binary": false,
            "tf_augmented": false,
            "use_idf": true,
            "use_normalization": true
        },
        "relevant_doc": [1, 2, 3]
    }
    ```

    **Response:**
    - status: Status operasi ("success" atau "error")
    - ranked_documents: List ID dokumen yang diurutkan berdasarkan similarity
    - average_precision: Nilai average precision untuk evaluasi
    - total_retrieved: Total dokumen yang ditemukan
    - query_used: Query text yang digunakan

    Args:
        request: DocumentRetrievalInputSimple berisi query, weighting_method, dan relevant_doc

    Returns:
        DocumentRetrievalResult berisi ranked documents dan average precision

    Raises:
        HTTPException: Jika terjadi error dalam proses retrieval atau cache belum tersedia
    """
    global _inverted_file_cache

    try:
        # ✅ CEK APAKAH CACHE TERSEDIA
        if (
            not _inverted_file_cache["is_cached"]
            or _inverted_file_cache["inverted_file"] is None
        ):
            raise HTTPException(
                status_code=400,
                detail="❌ Inverted file cache tidak tersedia. Silakan panggil GET /api/retrieval/inverted-file terlebih dahulu untuk generate dan cache inverted file.",
            )

        logger.info(f"Processing document retrieval for query: '{request.query}'")
        logger.info("Using cached inverted file")

        # Inisialisasi service
        retrieval_service = RetrievalService()

        # ⭐ GUNAKAN INVERTED FILE DARI CACHE
        cached_inverted_file = _inverted_file_cache["inverted_file"]

        # Panggil method retrieve_document dengan inverted file dari cache
        similarity_results, average_precision = (
            await retrieval_service.retrieve_document_single_query(
                query=request.query,
                inverted_file=cached_inverted_file,
                weighting_method=request.weighting_method,
                relevant_doc=request.relevant_doc,
            )
        )

        # Konversi dictionary similarity menjadi list document IDs yang ter-ranking
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
        # Re-raise HTTPException yang sudah ada
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
    Menggunakan caching untuk meningkatkan performa.

    ⭐ PENTING: Inverted file akan disimpan ke global cache untuk digunakan oleh endpoint retrieve.

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
    global _cached_inverted_file, _cache_key, _inverted_file_cache

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

        # ⭐ SIMPAN KE GLOBAL CACHE untuk endpoint retrieve
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
            "cache_info": "✅ Inverted file berhasil disimpan ke cache untuk endpoint retrieve",
        }

        # Simpan ke cache
        _cached_inverted_file = result
        _cache_key = current_cache_key

        logger.info(
            f"Inverted file generated and cached with {len(inverted_file)} terms"
        )
        logger.info("✅ Inverted file saved to global cache for retrieve endpoint")
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
    """
    Endpoint untuk mengecek status cache inverted file.
    """
    global _inverted_file_cache

    if (
        _inverted_file_cache["is_cached"]
        and _inverted_file_cache["inverted_file"] is not None
    ):
        total_terms = len(_inverted_file_cache["inverted_file"])
        return {
            "status": "✅ Cache tersedia",
            "is_cached": True,
            "total_terms": total_terms,
            "parameters": _inverted_file_cache["parameters"],
            "message": "Inverted file sudah tersedia di cache. Anda bisa menggunakan endpoint /retrieve.",
        }
    else:
        return {
            "status": "❌ Cache tidak tersedia",
            "is_cached": False,
            "total_terms": 0,
            "parameters": None,
            "message": "Silakan panggil GET /inverted-file terlebih dahulu untuk generate cache.",
        }


@router.delete("/inverted-file/cache")
async def clear_inverted_file_cache():
    """
    Endpoint untuk menghapus cache inverted file.
    """
    global _cached_inverted_file, _cache_key, _inverted_file_cache

    _cached_inverted_file = None
    _cache_key = None

    # Clear global cache untuk retrieve
    _inverted_file_cache["inverted_file"] = None
    _inverted_file_cache["parameters"] = None
    _inverted_file_cache["is_cached"] = False

    logger.info("All caches cleared")
    return {
        "status": "success",
        "message": "✅ Semua cache berhasil dihapus (response cache + inverted file cache)",
    }


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


@router.post("/retrieve-batch", response_model=BatchRetrievalResult)
async def batch_retrieve_documents(request: BatchRetrievalInput):
    """
    Endpoint untuk batch retrieval menggunakan cached inverted file.

    ⭐ PENTING: Endpoint ini menggunakan inverted file yang sudah di-cache dari endpoint GET /inverted-file.
    Pastikan Anda sudah memanggil GET /inverted-file terlebih dahulu sebelum menggunakan endpoint ini.

    **Contoh Request Body:**
    ```json
    {
        "query_file": "D://path/to/queries.xml",
        "relevant_doc": {
            "1": ["1", "4", "5", "6"],
            "2": ["2", "4", "3"]
        },
        "weighting_method": {
            "tf_raw": true,
            "tf_log": false,
            "tf_binary": false,
            "tf_augmented": false,
            "use_idf": true,
            "use_normalization": true
        }
    }
    ```

    **Response:**
    - status: Status operasi batch retrieval
    - total_queries: Total query yang diproses
    - mean_average_precision: MAP untuk semua query
    - query_results: Detail hasil untuk setiap query
    - processing_info: Info tambahan proses

    Args:
        request: BatchRetrievalInput berisi query_file, relevant_doc, dan weighting_method

    Returns:
        BatchRetrievalResult berisi hasil batch retrieval dan MAP

    Raises:
        HTTPException: Jika terjadi error atau cache belum tersedia
    """
    global _inverted_file_cache

    try:
        # ✅ CEK APAKAH CACHE TERSEDIA
        if (
            not _inverted_file_cache["is_cached"]
            or _inverted_file_cache["inverted_file"] is None
        ):
            raise HTTPException(
                status_code=400,
                detail="❌ Inverted file cache tidak tersedia. Silakan panggil GET /api/retrieval/inverted-file terlebih dahulu untuk generate dan cache inverted file.",
            )

        logger.info(f"Processing batch retrieval using cached inverted file")
        logger.info(f"Query file: {request.query_file}")
        logger.info(f"Total queries in relevant_doc: {len(request.relevant_doc)}")

        # Validasi file path exists
        import os

        if not os.path.exists(request.query_file):
            raise HTTPException(
                status_code=400,
                detail=f"❌ Query file tidak ditemukan: {request.query_file}",
            )

        # Inisialisasi service
        retrieval_service = RetrievalService()

        # ⭐ GUNAKAN INVERTED FILE DARI CACHE
        cached_inverted_file = _inverted_file_cache["inverted_file"]

        # Panggil batch retrieval function dengan filepath dan relevant_doc
        batch_results, mean_average_precision = (
            await retrieval_service.retrieve_document_batch_query(
                filename=request.query_file,
                inverted_file=cached_inverted_file,
                weighting_method=request.weighting_method,
                relevant_doc=request.relevant_doc,
            )
        )

        # Format hasil untuk response
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
                "total_relevant_queries": len(request.relevant_doc),
                "weighting_method": request.weighting_method,
                "cache_terms_count": len(cached_inverted_file),
            },
        )

    except HTTPException:
        # Re-raise HTTPException yang sudah ada
        raise
    except Exception as e:
        logger.exception("Error saat melakukan batch retrieval")
        raise HTTPException(
            status_code=500, detail=f"Error during batch retrieval: {str(e)}"
        )

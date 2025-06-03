"""
Query Router - Endpoint untuk query expansion menggunakan Word2Vec
"""

from fastapi import APIRouter, HTTPException
from typing import Optional
import logging
import os
from pydantic import BaseModel
from app.models.query_models import (
    BatchQueryExpansionInput,
    BatchQueryExpansionResult,
    Word2VecRetrainingInput,
    Word2VecRetrainingResult,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/query",
    tags=["query"],
    responses={404: {"description": "Not found"}},
)


class QueryRequest(BaseModel):
    query: str
    threshold: float = 0.7
    limit: int = -1


@router.post("/expand")
async def expand_query(request: QueryRequest):
    """
    Endpoint untuk melakukan query expansion menggunakan Word2Vec.
    """
    try:
        # Import di sini untuk menghindari circular import
        from main import get_query_expansion_service

        qe_service = get_query_expansion_service()
        result = await qe_service.expand_query(
            request.query, request.threshold, request.limit
        )

        return {
            "status": "success",
            "original_query": result["original_query"],
            "original_terms": result["original_terms"],
            "expansion_terms": result["expansion_terms"],
            "expanded_terms": result["expanded_terms"],
            "total_original_terms": len(result["original_terms"]),
            "total_expanded_terms": len(result["expanded_terms"]),
            "parameters": {
                "threshold": request.threshold,
                "limit": request.limit if request.limit > -1 else "unlimited",
            },
        }

    except Exception as e:
        logger.exception("Error during query expansion")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/expand-batch", response_model=BatchQueryExpansionResult)
async def expand_query_batch(request: BatchQueryExpansionInput):
    """
    Endpoint untuk melakukan batch query expansion menggunakan Word2Vec.
    """
    try:
        # Import di sini untuk menghindari circular import
        from main import get_query_expansion_service
        from app.data.parsing.func_parser import parser_query

        # Validasi file exists
        if not os.path.exists(request.query_file):
            raise HTTPException(
                status_code=400,
                detail=f"Query file tidak ditemukan: {request.query_file}",
            )

        logger.info(f"Processing batch query expansion from file: {request.query_file}")

        qe_service = get_query_expansion_service()

        # Parse query file
        list_query = parser_query(request.query_file)

        query_results = []

        for query_id, query_content in list_query.items():
            # Gabungkan title dan words untuk query lengkap
            full_query = str(query_content["title"] + " " + query_content["words"])

            try:
                # Lakukan query expansion untuk setiap query
                result = await qe_service.expand_query(
                    full_query, request.threshold, request.limit
                )

                query_results.append(
                    {
                        "query_id": query_id,
                        "original_query": result["original_query"],
                        "original_terms": result["original_terms"],
                        "expansion_terms": result["expansion_terms"],
                        "expanded_terms": result["expanded_terms"],
                        "total_original_terms": len(result["original_terms"]),
                        "total_expanded_terms": len(result["expanded_terms"]),
                    }
                )

            except Exception as e:
                logger.warning(f"Error expanding query {query_id}: {str(e)}")
                query_results.append(
                    {
                        "query_id": query_id,
                        "original_query": full_query,
                        "original_terms": [],
                        "expansion_terms": {},
                        "expanded_terms": [],
                        "total_original_terms": 0,
                        "total_expanded_terms": 0,
                        "error": str(e),
                    }
                )

        logger.info(
            f"Batch query expansion completed: {len(query_results)} queries processed"
        )

        return BatchQueryExpansionResult(
            status="success",
            total_queries=len(query_results),
            query_results=query_results,
            parameters={
                "threshold": request.threshold,
                "limit": request.limit if request.limit > -1 else "unlimited",
            },
            processing_info={
                "query_file_path": request.query_file,
                "successful_expansions": len(
                    [r for r in query_results if "error" not in r]
                ),
                "failed_expansions": len([r for r in query_results if "error" in r]),
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error during batch query expansion")
        raise HTTPException(status_code=500, detail=str(e))


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
            "preprocessing_config": qe_service.get_current_preprocessing_config(),
            "message": "Word2Vec model is trained and ready for query expansion",
        }
    except HTTPException as e:
        return {
            "status": "not_ready",
            "model_trained": False,
            "vocabulary_size": 0,
            "preprocessing_config": None,
            "message": str(e.detail),
        }
    except Exception as e:
        return {
            "status": "error",
            "model_trained": False,
            "vocabulary_size": 0,
            "preprocessing_config": None,
            "message": f"Error checking model status: {str(e)}",
        }


@router.post("/retrain-model", response_model=Word2VecRetrainingResult)
async def retrain_word2vec_model(request: Word2VecRetrainingInput):
    """
    Endpoint untuk melatih ulang Word2Vec model dengan konfigurasi preprocessing yang disesuaikan.

    Setelah retraining, semua service lain akan menggunakan model Word2Vec yang baru.
    """
    try:
        from main import get_query_expansion_service
        import os

        qe_service = get_query_expansion_service()

        # Simpan konfigurasi sebelumnya
        previous_config = qe_service.get_current_preprocessing_config()

        logger.info(f"Starting Word2Vec retraining with config: {request.dict()}")

        # Load dokumen yang sama dengan startup
        document_path = "app/data/parsing/parsing_docs.json"

        if not os.path.exists(document_path):
            raise HTTPException(
                status_code=404,
                detail=f"Document file tidak ditemukan: {document_path}",
            )

        # Baca dokumen
        documents = qe_service.read_json_collection(file_path=document_path)
        logger.info(f"Loaded {len(documents)} documents for retraining")

        # Lakukan retraining dengan konfigurasi baru
        training_info = await qe_service.retrain_word2vec_model(
            documents=documents,
            use_stemming=request.use_stemming,
            use_stopword_removal=request.use_stopword_removal,
        )

        logger.info("Word2Vec model retraining completed successfully")

        return Word2VecRetrainingResult(
            status="success",
            message="Word2Vec model berhasil dilatih ulang dengan konfigurasi baru",
            training_info=training_info,
            previous_config=previous_config,
            new_config=qe_service.get_current_preprocessing_config(),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error during Word2Vec model retraining")
        raise HTTPException(
            status_code=500, detail=f"Error melatih ulang Word2Vec model: {str(e)}"
        )


@router.get("/preprocessing-config")
async def get_preprocessing_config():
    """
    Endpoint untuk mendapatkan konfigurasi preprocessing yang sedang digunakan oleh Word2Vec model.
    """
    try:
        from main import get_query_expansion_service

        qe_service = get_query_expansion_service()
        config = qe_service.get_current_preprocessing_config()

        return {
            "status": "success",
            "preprocessing_config": config,
            "message": "Konfigurasi preprocessing Word2Vec model saat ini",
        }

    except HTTPException as e:
        return {
            "status": "not_ready",
            "preprocessing_config": None,
            "message": str(e.detail),
        }
    except Exception as e:
        return {
            "status": "error",
            "preprocessing_config": None,
            "message": f"Error getting preprocessing config: {str(e)}",
        }

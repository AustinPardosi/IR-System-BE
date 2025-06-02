"""
Query Router
-----------
Router untuk endpoint terkait query expansion menggunakan Word2Vec.
"""

from fastapi import APIRouter, HTTPException
from typing import Optional
import logging
from pydantic import BaseModel

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

    Args:
        request: QueryRequest model berisi query, threshold, dan limit

    Returns:
        Dict berisi hasil query expansion dengan kata-kata yang direkomendasikan
    """
    try:
        # Import di sini untuk menghindari circular import
        from main import get_query_expansion_service

        # Dapatkan service yang sudah dilatih saat startup
        qe_service = get_query_expansion_service()

        # Lakukan query expansion
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

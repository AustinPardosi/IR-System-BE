"""
Retrieval Router
-------------
Router untuk endpoint terkait retrieval:
1. Interactive query
2. Batch query
3. Inverted file
"""

from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException, Body
from typing import List, Dict, Any, Optional
import logging

from app.services.retrieval_service import RetrievalService
from app.services.query_expansion_service import QueryExpansionService
from app.models.query_models import (
    InteractiveQueryInput,
    BatchQueryInput,
    RetrievalResult,
)

logger = logging.getLogger(__name__)

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


@router.post("/inverted-file")
async def get_inverted_file(
    documents: Dict[str, Any],
    use_stemming: bool,
    use_stopword_removal: bool,
    document_weighting_method: Dict[str, bool],
):
    """
    Endpoint untuk mendapatkan inverted file.
    """
    return RetrievalService.create_inverted_file(
        documents, use_stemming, use_stopword_removal, document_weighting_method
    )


@router.get("/expand-query")
async def expand_query(
    q: str = "retrieval system",
    threshold: float = 0.7,
    limit: int = -1,
    qe_service: QueryExpansionService = Depends(get_query_expansion_service),
):
    try:
        result = await qe_service.expand_query(q, threshold, limit)
        return result
    except Exception as e:
        logger.exception("Error during query expansion")
        raise HTTPException(status_code=500, detail=str(e))

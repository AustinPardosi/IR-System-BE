"""
Documents Router
-------------
Router untuk endpoint terkait dokumen:
1. Upload dokumen collection
2. Parsing dokumen
"""

from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException
from typing import List, Dict, Any, Optional
import logging

from app.services.document_service import DocumentService

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/documents",
    tags=["documents"],
    responses={404: {"description": "Not found"}},
)


@router.post("/upload")
async def upload_documents(file: UploadFile = File(...)):
    """
    Endpoint untuk upload dokumen.
    """
    # Placeholder for implementation
    return {"message": "Upload documents placeholder"}


@router.post("/parse")
async def parse_documents(directory: str = Form(...)):
    """
    Endpoint untuk parsing dokumen dari direktori.
    """
    # Placeholder for implementation
    return {"message": "Parse documents placeholder"}

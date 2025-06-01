"""
Documents Router
-------------
Router untuk endpoint terkait dokumen:
1. Upload dokumen collection
2. Parsing dokumen
3. Get list document ID
"""

from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException
from typing import List, Dict, Any, Optional
import logging
import json
import os

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


@router.get("/list")
async def get_document_list():
    """
    Endpoint untuk mendapatkan list document ID dari file parsing_docs.json.
    Returns:
        List[Dict]: List berisi dictionary dengan format {"id": str, "label": str}
    """
    try:
        # Baca file parsing_docs.json
        file_path = os.path.join("app", "data", "parsing", "parsing_docs.json")
        with open(file_path, "r", encoding="utf-8") as f:
            docs = json.load(f)

        # Format response
        document_list = [
            {"id": str(doc_id), "label": f"Dokumen {doc_id}"} for doc_id in docs.keys()
        ]

        return document_list
    except FileNotFoundError:
        raise HTTPException(
            status_code=404, detail="File parsing_docs.json tidak ditemukan"
        )
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Error parsing file JSON")
    except Exception as e:
        logger.error(f"Error getting document list: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

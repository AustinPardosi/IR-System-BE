"""
Documents Router - Endpoint untuk operasi dokumen
"""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import List, Dict, Any, Optional
import logging
import json
import os

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/documents",
    tags=["documents"],
    responses={404: {"description": "Not found"}},
)


@router.get("/list")
async def get_document_list():
    """
    Endpoint untuk mendapatkan list document ID dari file parsing_docs.json.
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

        return {
            "status": "success",
            "total_documents": len(document_list),
            "documents": document_list,
        }
    except FileNotFoundError:
        raise HTTPException(
            status_code=404, detail="File parsing_docs.json tidak ditemukan"
        )
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Error parsing file JSON")
    except Exception as e:
        logger.error(f"Error getting document list: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/upload")
async def upload_documents(file: UploadFile = File(...)):
    """Endpoint untuk upload dokumen."""
    return {"message": "Upload documents placeholder", "filename": file.filename}


@router.post("/parse")
async def parse_documents(directory: str = Form(...)):
    """Endpoint untuk parsing dokumen dari direktori."""
    return {"message": "Parse documents placeholder", "directory": directory}

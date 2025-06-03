"""
Documents Router - Endpoint untuk operasi dokumen
"""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import List, Dict, Any, Optional
import logging
import json
import os
from app.models.query_models import (
    RetrieveDocumentsByIdsInput,
    RetrieveDocumentsByIdsResult,
)
from app.services.retrieval_service import RetrievalService

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


@router.post("/retrieve-by-ids", response_model=RetrieveDocumentsByIdsResult)
async def retrieve_documents_by_ids(request: RetrieveDocumentsByIdsInput):
    """
    Endpoint untuk mengambil dokumen berdasarkan list ID.
    Documents diambil dari file parsing_docs_with_field.json.
    """
    try:
        logger.info(f"Retrieving documents for {len(request.ids)} IDs")

        # Baca file documents
        file_path = os.path.join(
            "app", "data", "parsing", "parsing_docs_with_field.json"
        )
        if not os.path.exists(file_path):
            raise HTTPException(
                status_code=404,
                detail="File parsing_docs_with_field.json tidak ditemukan",
            )

        with open(file_path, "r", encoding="utf-8") as f:
            documents_data = json.load(f)

        # Convert documents to list format yang dibutuhkan retrieve_document_by_ids
        documents_list = []
        for doc_id, doc_content in documents_data.items():
            if isinstance(doc_content, dict):
                doc_dict = {
                    "id": doc_id,
                    "title": doc_content.get("title", ""),
                    "author": doc_content.get("author", ""),
                    "content": doc_content.get("words", ""),  # Map 'words' to 'content'
                    "bibliographic": doc_content.get("bibliographic", ""),
                }
                documents_list.append(doc_dict)
            else:
                # Jika format berbeda, buat struktur default
                documents_list.append(
                    {
                        "id": doc_id,
                        "content": str(doc_content),
                        "author": "",
                        "title": "",
                    }
                )

        # Implementasi langsung retrieve by IDs tanpa memanggil function
        found_documents = []
        found_ids = []

        for doc_id in request.ids:
            for doc in documents_list:
                if doc["id"] == doc_id:
                    found_documents.append(
                        {
                            "id": doc["id"],
                            "title": doc["title"],
                            "author": doc["author"],
                            "content": doc["content"],
                            "bibliographic": doc.get("bibliographic", ""),
                        }
                    )
                    found_ids.append(doc_id)
                    break

        # Hitung not found IDs
        not_found_ids = [doc_id for doc_id in request.ids if doc_id not in found_ids]

        logger.info(
            f"Found {len(found_documents)} documents, {len(not_found_ids)} not found"
        )

        return RetrieveDocumentsByIdsResult(
            status="success",
            total_requested=len(request.ids),
            total_found=len(found_documents),
            documents=found_documents,
            not_found_ids=not_found_ids,
            message=f"Berhasil mengambil {len(found_documents)} dari {len(request.ids)} dokumen yang diminta",
        )

    except FileNotFoundError:
        raise HTTPException(
            status_code=404, detail="File parsing_docs_with_field.json tidak ditemukan"
        )
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Error parsing file JSON")
    except Exception as e:
        logger.exception("Error saat retrieve documents by IDs")
        raise HTTPException(
            status_code=500, detail=f"Error retrieving documents: {str(e)}"
        )

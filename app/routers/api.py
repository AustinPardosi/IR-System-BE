"""
API Router
----------
Router untuk endpoint API yang memanfaatkan semua function yang sudah ada:
1. Hasil retrieval
2. MAP (Mean Average Precision)
3. Query expansion
4. Inverted file (all documents)
5. List document ID (untuk dropdown)
6. Get bobot by document ID
"""

from fastapi import APIRouter, HTTPException, Body
from typing import List, Dict, Any, Optional
import logging
import json

from app.services.retrieval_service import RetrievalService
from app.services.query_expansion_service import QueryExpansionService
from app.utils.evaluation import calculate_map, calculate_average_precision

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api",
    tags=["api"],
    responses={404: {"description": "Not found"}},
)

# Global variables untuk menyimpan data yang sudah diproses
cached_documents = None
cached_inverted_file = None
cached_document_list = None
cached_qe_service = None


async def get_query_expansion_service():
    """Get or create QueryExpansionService."""
    global cached_qe_service

    if cached_qe_service is None:
        try:
            cached_qe_service = await QueryExpansionService.create(
                document_path="IRTestCollection/cisi.all"
            )
            logger.info("QueryExpansionService initialized")
        except Exception as e:
            logger.error(f"Error initializing QueryExpansionService: {str(e)}")
            raise HTTPException(
                status_code=500, detail="QueryExpansionService not available"
            )

    return cached_qe_service


def load_cisi_documents():
    """Load dan cache dokumen CISI dari parsing hasil yang sudah ada."""
    global cached_documents, cached_document_list

    if cached_documents is not None:
        return cached_documents, cached_document_list

    try:
        # Load dari file parsing yang sudah ada
        with open("parsing/parsing_docs.json", "r", encoding="utf-8") as f:
            parsed_docs = json.load(f)

        # Convert ke format yang dibutuhkan services
        documents = {}
        document_list = []

        for doc_id, doc_data in parsed_docs.items():
            # Gabungkan semua content untuk IR processing
            content = f"{doc_data.get('title', '')} {doc_data.get('author', '')} {doc_data.get('words', '')} {doc_data.get('bibliographic', '')}"
            documents[doc_id] = content.strip()

            # List untuk dropdown
            document_list.append(
                {
                    "id": doc_id,
                    "title": doc_data.get("title", f"Document {doc_id}"),
                    "author": doc_data.get("author", "Unknown"),
                }
            )

        cached_documents = documents
        cached_document_list = document_list

        return documents, document_list

    except FileNotFoundError:
        raise HTTPException(
            status_code=500,
            detail="Parsed documents file not found. Please run document parsing first.",
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error loading documents: {str(e)}"
        )


async def get_or_create_inverted_file(
    use_stemming: bool = True,
    use_stopword_removal: bool = True,
    weighting_method: Dict[str, bool] = None,
):
    """Get atau create inverted file dengan caching."""
    global cached_inverted_file

    if weighting_method is None:
        weighting_method = {
            "tf_raw": True,
            "tf_log": False,
            "tf_binary": False,
            "tf_augmented": False,
            "use_idf": True,
            "use_normalization": True,
        }

    # Simple caching strategy - recreate if parameters change
    cache_key = (
        f"{use_stemming}_{use_stopword_removal}_{str(sorted(weighting_method.items()))}"
    )

    if (
        cached_inverted_file is None
        or getattr(cached_inverted_file, "cache_key", None) != cache_key
    ):
        documents, _ = load_cisi_documents()

        retrieval_service = RetrievalService()
        inverted_file = await retrieval_service.create_inverted_file(
            documents, use_stemming, use_stopword_removal, weighting_method
        )

        # Store with cache key
        inverted_file.cache_key = cache_key
        cached_inverted_file = inverted_file

    return cached_inverted_file


@router.post("/documents/list")
async def get_document_list():
    """
    Endpoint untuk mendapatkan list semua document ID beserta title dan author.
    Untuk digunakan dalam dropdown selection.
    """
    try:
        _, document_list = load_cisi_documents()
        return {
            "status": "success",
            "total_documents": len(document_list),
            "documents": document_list,
        }
    except Exception as e:
        logger.exception("Error getting document list")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/retrieval/search")
async def search_documents(
    query: str = Body(...),
    use_stemming: bool = Body(True),
    use_stopword_removal: bool = Body(True),
    weighting_method: Optional[Dict[str, bool]] = Body(None),
    relevant_docs: Optional[List[int]] = Body([]),
):
    """
    Endpoint untuk melakukan document retrieval berdasarkan query.
    Mengembalikan hasil retrieval dengan ranking dan average precision.
    """
    try:
        if weighting_method is None:
            weighting_method = {
                "tf_raw": True,
                "tf_log": False,
                "tf_binary": False,
                "tf_augmented": False,
                "use_idf": True,
                "use_normalization": True,
            }

        # Get inverted file
        inverted_file = await get_or_create_inverted_file(
            use_stemming, use_stopword_removal, weighting_method
        )

        # Perform retrieval
        retrieval_service = RetrievalService()
        ranked_docs, avg_precision = await retrieval_service.retrieve_document(
            query, inverted_file, weighting_method, relevant_docs or []
        )

        return {
            "status": "success",
            "query": query,
            "total_retrieved": len(ranked_docs),
            "ranked_documents": ranked_docs,
            "average_precision": avg_precision,
            "parameters": {
                "use_stemming": use_stemming,
                "use_stopword_removal": use_stopword_removal,
                "weighting_method": weighting_method,
            },
        }

    except Exception as e:
        logger.exception("Error during document retrieval")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/evaluation/map")
async def calculate_mean_average_precision(
    queries_results: Dict[str, List[str]] = Body(...),
    relevant_documents: Dict[str, List[str]] = Body(...),
):
    """
    Endpoint untuk menghitung Mean Average Precision (MAP) dari hasil retrieval.

    Args:
        queries_results: Dict mapping query ID ke list dokumen yang di-retrieve
        relevant_documents: Dict mapping query ID ke list dokumen yang relevan
    """
    try:
        map_score = calculate_map(queries_results, relevant_documents)

        return {
            "status": "success",
            "map_score": map_score,
            "total_queries": len(queries_results),
            "evaluation_details": {
                "queries_evaluated": list(queries_results.keys()),
                "average_precision_per_query": {
                    query_id: calculate_average_precision(
                        queries_results[query_id], relevant_documents.get(query_id, [])
                    )
                    for query_id in queries_results.keys()
                },
            },
        }

    except Exception as e:
        logger.exception("Error calculating MAP")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query/expand")
async def expand_query(
    query: str = Body(...), threshold: float = Body(0.7), limit: int = Body(-1)
):
    """
    Endpoint untuk melakukan query expansion menggunakan Word2Vec.
    """
    try:
        qe_service = await get_query_expansion_service()
        result = await qe_service.expand_query(query, threshold, limit)

        return {
            "status": "success",
            "original_query": result["original_query"],
            "original_terms": result["original_terms"],
            "expansion_terms": result["expansion_terms"],
            "expanded_terms": result["expanded_terms"],
            "total_original_terms": len(result["original_terms"]),
            "total_expanded_terms": len(result["expanded_terms"]),
            "parameters": {
                "threshold": threshold,
                "limit": limit if limit > -1 else "unlimited",
            },
        }

    except Exception as e:
        logger.exception("Error during query expansion")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/inverted-file")
async def get_inverted_file(
    use_stemming: bool = Body(True),
    use_stopword_removal: bool = Body(True),
    weighting_method: Optional[Dict[str, bool]] = Body(None),
):
    """
    Endpoint untuk mendapatkan inverted file dari semua dokumen.
    """
    try:
        if weighting_method is None:
            weighting_method = {
                "tf_raw": True,
                "tf_log": False,
                "tf_binary": False,
                "tf_augmented": False,
                "use_idf": True,
                "use_normalization": True,
            }

        inverted_file = await get_or_create_inverted_file(
            use_stemming, use_stopword_removal, weighting_method
        )

        # Remove cache_key attribute untuk response
        clean_inverted_file = {
            k: v for k, v in inverted_file.items() if k != "cache_key"
        }

        return {
            "status": "success",
            "total_terms": len(clean_inverted_file),
            "inverted_file": clean_inverted_file,
            "parameters": {
                "use_stemming": use_stemming,
                "use_stopword_removal": use_stopword_removal,
                "weighting_method": weighting_method,
            },
        }

    except Exception as e:
        logger.exception("Error creating inverted file")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/documents/weights")
async def get_document_weights(
    document_id: str = Body(...),
    use_stemming: bool = Body(True),
    use_stopword_removal: bool = Body(True),
    weighting_method: Optional[Dict[str, bool]] = Body(None),
):
    """
    Endpoint untuk mendapatkan bobot setiap term dalam dokumen tertentu.
    """
    try:
        if weighting_method is None:
            weighting_method = {
                "tf_raw": True,
                "tf_log": False,
                "tf_binary": False,
                "tf_augmented": False,
                "use_idf": True,
                "use_normalization": True,
            }

        # Get inverted file
        inverted_file = await get_or_create_inverted_file(
            use_stemming, use_stopword_removal, weighting_method
        )

        # Get document weights
        retrieval_service = RetrievalService()
        doc_weights = await retrieval_service.get_weight_by_document_id(
            document_id, inverted_file
        )

        if not doc_weights:
            raise HTTPException(
                status_code=404, detail=f"Document {document_id} not found"
            )

        # Load document info untuk context
        _, document_list = load_cisi_documents()
        doc_info = next(
            (doc for doc in document_list if doc["id"] == document_id), None
        )

        return {
            "status": "success",
            "document_id": document_id,
            "document_info": doc_info,
            "total_terms": len(doc_weights),
            "term_weights": doc_weights,
            "parameters": {
                "use_stemming": use_stemming,
                "use_stopword_removal": use_stopword_removal,
                "weighting_method": weighting_method,
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error getting document weights")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/retrieval/batch")
async def batch_retrieval_with_evaluation(
    queries: Dict[str, str] = Body(...),
    relevant_docs: Dict[str, List[str]] = Body(...),
    use_stemming: bool = Body(True),
    use_stopword_removal: bool = Body(True),
    weighting_method: Optional[Dict[str, bool]] = Body(None),
):
    """
    Endpoint untuk batch retrieval dengan evaluasi MAP.

    Args:
        queries: Dict mapping query ID ke query string
        relevant_docs: Dict mapping query ID ke list dokumen yang relevan
    """
    try:
        if weighting_method is None:
            weighting_method = {
                "tf_raw": True,
                "tf_log": False,
                "tf_binary": False,
                "tf_augmented": False,
                "use_idf": True,
                "use_normalization": True,
            }

        # Get inverted file
        inverted_file = await get_or_create_inverted_file(
            use_stemming, use_stopword_removal, weighting_method
        )

        retrieval_service = RetrievalService()

        # Process each query
        all_results = {}
        all_avg_precisions = {}

        for query_id, query_text in queries.items():
            relevant_for_query = [
                int(doc_id) for doc_id in relevant_docs.get(query_id, [])
            ]

            ranked_docs, avg_precision = await retrieval_service.retrieve_document(
                query_text, inverted_file, weighting_method, relevant_for_query
            )

            all_results[query_id] = ranked_docs
            all_avg_precisions[query_id] = avg_precision

        # Calculate MAP
        map_score = calculate_map(all_results, relevant_docs)

        return {
            "status": "success",
            "total_queries": len(queries),
            "map_score": map_score,
            "results": all_results,
            "average_precisions": all_avg_precisions,
            "parameters": {
                "use_stemming": use_stemming,
                "use_stopword_removal": use_stopword_removal,
                "weighting_method": weighting_method,
            },
        }

    except Exception as e:
        logger.exception("Error during batch retrieval")
        raise HTTPException(status_code=500, detail=str(e))

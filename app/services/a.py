

import asyncio
from app.services.retrieval_service import RetrievalService


query = "information retrieval system"

inverted_file = {
    "information": {"1": 0.3, "2": 0.5},
    "retrieval": {"1": 0.4},
    "system": {"2": 0.6, "3": 0.2},
}

documents = {
    "1": {"information": 2, "retrieval": 3},
    "2": {"information": 5, "system": 4},
    "3": {"system": 1},
}

relevant_doc = [1, 2]  # Dokumen 1 dan 2 dianggap relevan

weighting_method = {
    "tf_raw": False,
    "tf_logarithmic": True,
    "tf_binary": False,
    "tf_augmented": False,
    "use_idf": True,
    "use_normalization": True,
}

async def aw():
    service = RetrievalService()
    docs, ap = await service.retrieve_document(query,inverted_file,weighting_method,documents,relevant_doc)
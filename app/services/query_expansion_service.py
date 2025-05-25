"""
Query Expansion Service
---------------------
Modul ini bertanggung jawab untuk:
1. Melakukan query expansion menggunakan Word2Vec
2. Melatih model Word2Vec dari dokumen
3. Memilih term-term yang relevan untuk query expansion
"""

from typing import List, Dict, Any, Optional
import logging
import numpy as np
from gensim.models import Word2Vec
from ..utils.text_preprocessing import preprocess_text

logger = logging.getLogger(__name__)


class QueryExpansionService:
    """
    Service untuk melakukan query expansion dengan Word2Vec.
    """

    def __init__(self):
        """
        Inisialisasi QueryExpansionService.
        """
        self.model = None

    async def train_word2vec_model(self, documents: Dict[str, str]) -> None:
        """
        Melatih model Word2Vec dari dokumen.

        Args:
            documents: Dictionary berisi dokumen dengan format {doc_id: content}.
        """
        # Preprocess semua dokumen
        processed_docs = []
        for doc_id, content in documents.items():
            # Gunakan fungsi preprocess yang sudah ada
            tokens = preprocess_text(
                content, use_stemming=True, use_stopword_removal=True
            )
            processed_docs.append(tokens)

        # Train Word2Vec model
        self.model = Word2Vec(
            sentences=processed_docs,
            vector_size=100,  # Dimensi vektor
            window=5,  # Ukuran window konteks
            min_count=2,  # Frekuensi minimum term
            workers=4,  # Jumlah thread
            sg=1,  # Skip-gram model (lebih baik untuk kata jarang)
        )

        logger.info(
            f"Word2Vec model trained with vocabulary size: {len(self.model.wv.key_to_index)}"
        )

    async def load_pretrained_model(self, model_path: str) -> None:
        """
        Memuat model Word2Vec pretraining.

        Args:
            model_path: Path ke model pretraining.
        """
        try:
            self.model = Word2Vec.load(model_path)
            logger.info(f"Loaded pretrained model from {model_path}")
        except Exception as e:
            logger.error(f"Error loading pretrained model: {str(e)}")
            raise

    async def expand_query(self, query: str, threshold: float = 0.7) -> Dict[str, Any]:
        """
        Melakukan ekspansi query menggunakan Word2Vec.

        Args:
            query: Query original.
            threshold: Threshold untuk memilih term yang akan ditambahkan (0.0 - 1.0).

        Returns:
            Dictionary berisi query original dan expanded, serta term-term yang ditambahkan.
        """
        if not self.model:
            raise ValueError("Word2Vec model belum dilatih atau dimuat!")

        # Preprocess query
        query_terms = preprocess_text(
            query, use_stemming=True, use_stopword_removal=True
        )

        # Simpan term yang akan ditambahkan
        expansion_terms = {}

        # Untuk setiap term di query, cari term yang similar
        for term in query_terms:
            similar_terms = await self.get_similar_terms(term, threshold)
            if similar_terms:
                expansion_terms[term] = similar_terms

        # Gabungkan query asli dengan term ekspansi
        expanded_terms = query_terms.copy()
        for term_list in expansion_terms.values():
            for term_dict in term_list:
                expanded_terms.append(term_dict["term"])

        return {
            "original_query": query,
            "original_terms": query_terms,
            "expansion_terms": expansion_terms,
            "expanded_terms": list(set(expanded_terms)),  # Hapus duplikat
        }

    async def get_similar_terms(
        self, term: str, threshold: float
    ) -> List[Dict[str, Any]]:
        """
        Mendapatkan term-term yang similar dengan term yang diberikan.

        Args:
            term: Term yang ingin dicari similarnya.
            threshold: Threshold similarity untuk memilih term (0.0 - 1.0).

        Returns:
            List term-term yang similar dengan nilai similaritasnya.
        """
        if not self.model or term not in self.model.wv:
            return []

        # Dapatkan term yang similar
        similar_terms = self.model.wv.most_similar(term, topn=10)

        # Filter berdasarkan threshold dan format hasilnya
        filtered_terms = [
            {"term": t, "similarity": float(s)}
            for t, s in similar_terms
            if s >= threshold
        ]

        return filtered_terms

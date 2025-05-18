"""
Retrieval Service
----------------
Modul ini bertanggung jawab untuk:
1. Menghitung similaritas dokumen dengan query
2. Melakukan pembobotan TF-IDF
3. Membuat inverted file
4. Mengembalikan hasil retrieval dengan ranking
"""

from typing import List, Dict, Any, Optional
import logging
from app.models.query_models import (
    InteractiveQueryInput,
    BatchQueryInput,
    RetrievalResult,
)
import math

logger = logging.getLogger(__name__)


class RetrievalService:
    """
    Service untuk melakukan information retrieval.
    """

    def __init__(self):
        """
        Inisialisasi RetrievalService.
        """
        # Placeholder for implementation
        pass

    async def create_inverted_file(
        self, documents: Dict[str, Any], use_stemming: bool, use_stopword_removal: bool
    ) -> Dict[str, Any]:
        """
        Membuat inverted file dari dokumen.

        Args:
            documents: Dictionary berisi dokumen.
            use_stemming: Apakah akan menggunakan stemming.
            use_stopword_removal: Apakah akan menghilangkan stopwords.

        Returns:
            Inverted file sebagai dictionary.
        """
        # Placeholder for implementation
        pass

    async def calculate_tf_idf(
        self, term: str, doc: str, freq_file: Dict[str, Any], weighting_method: Dict[str, bool]
    ) -> Dict[str, Any]:
        """
        Menghitung bobot TF-IDF pada term di doc tertentu.

        Args:
            term: kata yang akan dihitung weight nya.
            doc: letak dokumen di mana bobot kata dihitung.
            freq_file: Frekuensi kemunculan term pada tiap dokumen.
            weighting_method: Metode pembobotan yang dipilih.

        Returns:
            Hasil perhitungan TF-IDF.
        """
        tf_raw = weighting_method.get("tf_raw", False)
        tf_log = weighting_method.get("tf_log", False)
        tf_binary = weighting_method.get("tf_binary", False)
        tf_augmented = weighting_method.get("tf_augmented", False)
        use_idf = weighting_method.get("use_idf", False)
        use_normalization = weighting_method.get("use_normalization", False)

        # Get term frequencies across docs
        term_docs = freq_file.get(term, {})
        freq_in_doc = term_docs.get(doc, 0)

        if freq_in_doc == 0:
            return {"term": term, "doc": doc, "weight": 0}
        
        tf = 1
        idf = 1
        normalization = 1

        # TF
        if tf_raw:
            tf = freq_in_doc
        elif tf_log:
            tf = 1 + math.log2(freq_in_doc)
        elif tf_binary:
            tf = 1
        elif tf_augmented:
            freqs_in_doc = [freqs.get(doc, 0) for freqs in freq_file.values()]
            max_freq = max(freqs_in_doc) if freqs_in_doc else 1
            tf = 0.5 + 0.5 * (freq_in_doc / max_freq)
        else:
            # Defaultnya adalah raw tf
            tf = freq_in_doc

        # IDF
        N = len({doc_id for docs in freq_file.values() for doc_id in docs})
        df = len(term_docs)
        idf = math.log(N / (1 + df)) if use_idf else 1.0

        # Normalization
        if use_normalization:
            doc_length = sum(freqs.get(doc, 0) for freqs in freq_file.values())
            if doc_length == 0:
                doc_length = 1  # Menghindari pembagian 0
            
            normalization = 1/doc_length

        weight = tf*idf*normalization

        return {
            "term": term,
            "doc": doc,
            "weight": weight
        }

        

    async def calculate_similarity(
        self,
        query_vector: Dict[str, float],
        document_vectors: Dict[str, Dict[str, float]],
    ) -> List[Dict[str, Any]]:
        """
        Menghitung similaritas antara query dan dokumen.

        Args:
            query_vector: Vector query.
            document_vectors: Vector dokumen.

        Returns:
            List dokumen dengan nilai similaritas, diurutkan.
        """
        # Placeholder for implementation
        pass

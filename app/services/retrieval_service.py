"""
Retrieval Service
----------------
Modul ini bertanggung jawab untuk:
1. Menghitung similaritas dokumen dengan query
2. Melakukan pembobotan TF-IDF
3. Membuat inverted file
4. Mengembalikan hasil retrieval dengan ranking
5. Mengambil bobot setiap term dalam dokumen tertentu
"""

from typing import List, Dict, Any, Optional, Tuple
import logging
from app.models.query_models import (
    InteractiveQueryInput,
    BatchQueryInput,
    RetrievalResult,
)
import math

from app.test.retrieval_test import tokenize
from app.utils.evaluation import calculate_average_precision

from ..utils.text_preprocessing import preprocess_text
from collections import Counter


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


    async def create_inverted_file (
        self, documents: Dict[str, Any], use_stemming: bool, use_stopword_removal: bool,
        document_weighting_method: Dict[str, bool]
    ) -> Dict[str, Any]:
        """
        Membuat inverted file dari dokumen.

        Args:
            documents: Dictionary berisi ID dokumen dan isi teksnya.
            use_stemming: Apakah akan menggunakan stemming.
            use_stopword_removal: Apakah akan menghilangkan stopwords.

        Returns:
            Inverted file sebagai dictionary.
        """
        freq_file = {}
        for doc_key, doc_value in documents.items():
            tokens = preprocess_text(doc_value, use_stemming, use_stopword_removal)
            tokens_freq = Counter(tokens)
            
            # Mengisi freq_file
            freq_file[doc_key] = tokens_freq

        inverted_file = {}
        # Menghitung bobot term dan menyusun inverted file
        for doc_key, doc_freq in freq_file.items():
            for token_key, _ in doc_freq.items():
                weight = await self.calculate_tf_idf (token_key, doc_key, freq_file, document_weighting_method)
                inverted_file.setdefault(weight["term"], {})[weight["doc"]] = weight["weight"]

        return inverted_file


    async def calculate_tf_idf (
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
        term_docs = freq_file.get(doc, {})
        freq_in_doc = term_docs.get(term, 0)

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
            terms = freq_file.get(doc,{})
            freqs_in_doc = [x for x in terms.values()]
            max_freq = max(freqs_in_doc) if freqs_in_doc else 1
            tf = 0.5 + 0.5 * (freq_in_doc / max_freq)
        else:
            # Defaultnya adalah raw tf
            tf = freq_in_doc

        # IDF
        N = len(freq_file)
        df = len([1 for _, terms in freq_file.items() if term in terms])
        idf = math.log2(N / df) if use_idf else 1.0

        # Normalization
        if use_normalization:
            doc_length = sum(freq_file.get(doc, {}).values())
            if doc_length == 0:
                doc_length = 1  # Menghindari pembagian 0
            
            normalization = 1/doc_length

        weight = tf*idf*normalization

        return {
            "term": term,
            "doc": doc,
            "weight": weight
        }
    
    async def calculate_query_weight (
        self, query: str, weighting_method: Dict[str,bool], inverted_file: Dict[str,Any]
    ) -> Dict[str, Any]:
        # Pembentukan vektor query
        query_terms =  tokenize(query) 
        term_freq = {}
        for term in query_terms:
            term_freq[term] = term_freq.get(term, 0) + 1
        max_tf = max(term_freq.values()) if term_freq else 1

        # Perhitungan bobot query
        N = len({doc_id for postings in inverted_file.values() for doc_id in postings})

        def get_tf_weight(tf: int) -> float:
            if weighting_method.get("tf_raw"):
                return tf
            elif weighting_method.get("tf_augmented"):
                return 0.5 + 0.5 * (tf / max_tf)
            elif weighting_method.get("tf_binary"):
                return 1.0 if tf > 0 else 0.0
            elif weighting_method.get("tf_logarithmic"):
                return 1.0 + math.log(tf) if tf > 0 else 0.0
            return tf

        query_vector = {}
        for term, tf in term_freq.items():
            tf_weight = get_tf_weight(tf)
            idf = 0.0
            if term in inverted_file:
                df = len(inverted_file[term])
                if df > 0:
                    idf = math.log(N / df)
            weight = tf_weight * idf if weighting_method.get("use_idf") else tf_weight
            if weight > 0:
                query_vector[term] = weight

        if weighting_method.get("use_normalization"):
            norm = math.sqrt(sum(w ** 2 for w in query_vector.values()))
            if norm > 0:
                for term in query_vector:
                    query_vector[term] /= norm
        return query_vector


    async def calculate_similarity (
        self,
        query_vector: Dict[str, float],
        document_vectors: Dict[str, Dict[str, float]],
    ) -> Dict[str, Any]:
        """
        Menghitung similaritas antara query dan dokumen.

        Args:
            query_vector: Vector query.
            document_vectors: Vector dokumen.

        Returns:
            List dokumen dengan nilai similaritas, diurutkan.
        """
        query_docs_similarities = {}
        
        for query_key, _ in query_vector.items():
            if (query_key in document_vectors.keys()):
                for doc, _ in document_vectors[query_key].items():
                    query_docs_similarities.setdefault(doc, 0)
                    query_docs_similarities[doc] += ( query_vector[query_key] * document_vectors[query_key][doc])
        
        if (len(query_docs_similarities) > 0):
            query_docs_similarities = dict ( sorted (
                query_docs_similarities.items(), key=lambda item: item[1], reverse=True
            ))

        return (query_docs_similarities)
    

    async def retrieve_document_single_query (
        self,
        query: str,
        inverted_file: Dict[str,Any],
        weighting_method: Dict[str, bool],
        relevant_doc: List[int],
    ) -> Tuple[Dict[str, Any], float]:
        """
        Mengambil dokumen yang relevan berdasarkan query yang dimasukkan

        Args:
            query: string query.
            inverted_file: file yang berisi bobot-bobot term pada setiap dokumen.
            weighting_method: metode pembobotan untuk query.
            relevant_doc: list id dokumen yang relevan

        Returns:
            List dokumen yang terurut berdasarkan similarity beserta average precisionnya.
        """

        query_vector = self.calculate_query_weight(query, weighting_method, inverted_file)

        # Hitung similarity
        sim = await self.calculate_similarity(query_vector, inverted_file)
            
        ranked_docs = [doc_id for doc_id in sim]

        # Hitung Average Precision (untuk batch query, yang interactive tidak ada relevance judgement)
        average_precision = 0
        if len(relevant_doc) != 0:
            relevant_doc_ids = [str(doc_id) for doc_id in relevant_doc]
            average_precision = calculate_average_precision(ranked_docs, relevant_doc_ids)

        return sim, average_precision
    

    async def retrieve_document_batch_query (
        self,
        list_query: Dict[str, str],
        inverted_file: Dict[str,Any],
        weighting_method: Dict[str, bool],
        relevant_doc: Dict[str, List[int]],
    ) -> Tuple[List[Tuple[Dict[str, float], float]], float]:
        """
        Mengambil dokumen yang relevan berdasarkan query yang dimasukkan

        Args:
            list_query: kamus ID query dengan query-nya.
            inverted_file: file yang berisi bobot-bobot term pada setiap dokumen.
            weighting_method: metode pembobotan untuk query.
            relevant_doc: list id dokumen yang relevan

        Returns:
            Tuple berisi: daftar yang berisi tuple kamus dokumen ter-retreived
            beserta masing-masing similarity-nya dengan satu query dan average
            precision-nya, dengan mean_avergae_precision secara keseluruhan.
        """

        tuple_sim_ap = []
        for query_id, query_content in list_query.items():
            if (query_id in relevant_doc):
                sim, average_precision = self.retrieve_document_single_query(
                    query_content, inverted_file, weighting_method, relevant_doc[query_id]
                )
                tuple_sim_ap.append((sim, average_precision))
        
        average_precisions = [tuple_sim_ap[i][1] for i in range(len(tuple_sim_ap))]
        mean_average_precision = sum(average_precisions) / len(average_precisions)

        retrieval_result = (tuple_sim_ap, mean_average_precision)
        return retrieval_result
    

    async def retrieve_document_by_id(
        self,
        id: str,
        documents: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        for doc in documents:
            if doc['id'] == id:
                return {
                    'author': doc['author'],
                    'title': doc['title'],
                    'content': doc['content']
                }
        return {} 


    async def retrieve_document_by_ids(
        self,
        documents: List[Dict[str, Any]],
        ids: List[str],
    ) -> List[Dict[str, Any]]:
        list_of_docs = []
        for id in ids:
            doc = await self.retrieve_document_by_id(id, documents)
            if doc:
                list_of_docs.append(doc)
        return list_of_docs


    async def get_weight_by_document_id (
            self,
            document_id: str,
            inverted_file: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Mengambil bobot setiap term dalam dokumen tertentu.

        Args:
            document_id: ID dokumen yang akan diambil bobot-bobot term-nya
            inverted_file: inverted file dalam format [term: (doc: weight)]

        Returns:
            Kamus bobot setiap kata dalam dokumen yang diinginkan. 
        """
        doc_dict = {}
        for file_key, file_value in inverted_file.items():
            if (document_id in file_value.keys()):
                doc_dict[file_key] = file_value[document_id]
        return (doc_dict)
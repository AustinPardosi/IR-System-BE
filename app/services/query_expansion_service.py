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
        self._is_trained = False

    @classmethod
    async def create(cls, document_path: str):
        """
        Async factory method to initialize and train Word2Vec model.
        """
        self = cls()
        documents = self.read_cisi_collection(file_path=document_path)
        print(f"Loaded {len(documents)} documents")

        print("Training Word2Vec model...")
        await self.train_word2vec_model(documents)

        return self

    async def ensure_model_trained(self, document_path: str = None) -> None:
        """
        Memastikan model sudah dilatih sebelum digunakan.
        Jika model belum dilatih dan document_path diberikan, model akan dilatih.

        Args:
            document_path: Path ke dokumen untuk training (opsional)
        """
        if not self._is_trained:
            if document_path:
                documents = self.read_cisi_collection(file_path=document_path)
                print(f"Loaded {len(documents)} documents")
                print("Training Word2Vec model...")
                await self.train_word2vec_model(documents)
            else:
                raise ValueError(
                    "Model belum dilatih dan tidak ada document_path yang diberikan!"
                )

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

        self._is_trained = True
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

    async def expand_query(
        self, query: str, threshold: float = 0.7, limit: int = -1
    ) -> Dict[str, Any]:
        """
        Melakukan ekspansi query menggunakan Word2Vec.

        Args:
            query: Query original.
            threshold: Threshold untuk memilih term yang akan ditambahkan (0.0 - 1.0).
            limit: Batas maksimal banyaknya kata yang ditambahkan pada query expansion (-1 jika tidak ada limit).

        Returns:
            Dictionary berisi query original dan expanded, serta term-term yang ditambahkan.
        """
        if not self._is_trained:
            raise ValueError(
                "Word2Vec model belum dilatih! Gunakan ensure_model_trained() terlebih dahulu."
            )

        isLimited = limit > -1

        # Preprocess query
        query_terms = preprocess_text(
            query, use_stemming=True, use_stopword_removal=True
        )

        # Simpan term yang akan ditambahkan
        expansion_terms = {}
        all_expansion_candidates = []

        # Untuk setiap term di query, cari term yang similar
        for term in query_terms:
            similar_terms = await self.get_similar_terms(term, threshold)
            if similar_terms:
                expansion_terms[term] = similar_terms
                if isLimited:
                    for term_dict in similar_terms:
                        term_dict["source"] = term  # keep track of original term
                        all_expansion_candidates.append(term_dict)

        if isLimited:
            # Urutkan berdasarkan similarity (descending) dan potong berdasarkan limit
            all_expansion_candidates.sort(key=lambda x: x["similarity"], reverse=True)
            all_expansion_candidates = all_expansion_candidates[:limit]

            # Rekonstruksi expansion terms (untuk kondisi limited)
            returned_expansion_terms = {}
            for item in all_expansion_candidates:
                source = item["source"]
                returned_expansion_terms.setdefault(source, []).append(
                    {"term": item["term"], "similarity": item["similarity"]}
                )

            # Final expanded terms
            expansion_terms = returned_expansion_terms
            expanded_terms = query_terms + [
                item["term"] for item in all_expansion_candidates
            ]
        else:
            # Gabungkan query asli dengan term ekspansi
            expanded_terms = query_terms.copy()
            for term_list in expansion_terms.values():
                for term_dict in term_list:
                    expanded_terms.append(term_dict["term"])

        return {
            "original_query": query,
            "original_terms": query_terms,
            "expansion_terms": expansion_terms,
            "expanded_terms": expanded_terms,  # Hapus duplikat
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

    def read_cisi_collection(self, file_path: str) -> dict:
        """
        Membaca koleksi CISI dan mengembalikan dictionary dokumen.
        """
        documents = {}
        current_id = None
        current_content = []

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith(".I "):
                    # Simpan dokumen sebelumnya jika ada
                    if current_id is not None:
                        documents[current_id] = " ".join(current_content)
                    current_id = line.strip().split(" ")[1]
                    current_content = []
                elif line.startswith(".W"):
                    continue
                else:
                    current_content.append(line.strip())

        # Simpan dokumen terakhir
        if current_id is not None:
            documents[current_id] = " ".join(current_content)

        return documents

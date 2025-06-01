# FUNCTION SAMA DENGAN DI FILE
# retrieval_service.py dan text_preprocessing.py

from typing import List, Dict, Any, Set
import math

import logging
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

logger = logging.getLogger(__name__)


STOPWORDS: Set[str] = set()
import nltk

# NLTK data sudah didownload saat startup aplikasi
# Lazy loading untuk stopwords
_stop_words_cache = None


def get_stopwords() -> Set[str]:
    """Lazy loading untuk stopwords."""
    global _stop_words_cache
    if _stop_words_cache is None:
        try:
            from nltk.corpus import stopwords

            _stop_words_cache = set(stopwords.words("english"))
        except Exception as e:
            _stop_words_cache = set()  # Fallback ke empty set
    return _stop_words_cache


from collections import Counter

stemmer = PorterStemmer()


def tokenize(text: str) -> List[str]:
    """
    Tokenisasi teks menjadi list token.

    Args:
        text: Teks yang akan ditokenisasi.

    Returns:
        List token hasil tokenisasi.
    """
    try:
        tokenized_text = word_tokenize(text)
        return tokenized_text
    except Exception as e:
        # Fallback ke simple split jika word_tokenize gagal
        return text.split()


def remove_stopwords(tokens: List[str]) -> List[str]:
    """
    Menghapus stopwords dari list token.

    Args:
        tokens: List token input.

    Returns:
        List token tanpa stopwords.
    """
    filtered_sent = []
    for w in tokens:
        if w not in get_stopwords():
            filtered_sent.append(w)

    return filtered_sent


def stem_word(word: str) -> str:
    """
    Melakukan stemming pada sebuah kata.

    Args:
        word: Kata yang akan di-stem.

    Returns:
        Kata hasil stemming.
    """
    return stemmer.stem(word)


def stem_tokens(tokens: List[str]) -> List[str]:
    """
    Melakukan stemming pada list token.

    Args:
        tokens: List token yang akan di-stem.

    Returns:
        List token hasil stemming.
    """
    return [stemmer.stem(token) for token in tokens]


def preprocess_text(
    text: str, use_stemming: bool = True, use_stopword_removal: bool = True
) -> List[str]:
    """
    Melakukan preprocessing teks lengkap.

    Args:
        text: Teks yang akan dipreprocess.
        use_stemming: Apakah akan menggunakan stemming.
        use_stopword_removal: Apakah akan menghilangkan stopwords.

    Returns:
        List token hasil preprocessing, lowercase
    """
    tokens = tokenize(text)
    if use_stemming:
        tokens = stem_tokens(tokens)
    if use_stopword_removal:
        tokens = remove_stopwords(tokens)

    return [token.lower() for token in tokens]


# -----------------------------------------------------------------------------------------------------


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

    def create_inverted_file(
        self,
        documents: Dict[str, Any],
        use_stemming: bool,
        use_stopword_removal: bool,
        document_weighting_method: Dict[str, bool],
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
        for doc_key, doc_freqs in freq_file.items():
            for token_key, _ in doc_freqs.items():
                weight = self.calculate_tf_idf(
                    token_key, doc_key, freq_file, document_weighting_method
                )
                inverted_file.setdefault(weight["term"], {})[weight["doc"]] = weight[
                    "weight"
                ]

        return inverted_file

    def calculate_tf_idf(
        self,
        term: str,
        doc: str,
        freq_file: Dict[str, Any],
        weighting_method: Dict[str, bool],
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
            freqs_in_doc = [freqs.get(term, 0) for freqs in freq_file.values()]
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

            normalization = 1 / doc_length

        weight = tf * idf * normalization

        return {"term": term, "doc": doc, "weight": weight}

    def calculate_similarity(
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
            if query_key in document_vectors:
                for doc, _ in document_vectors[query_key].items():
                    query_docs_similarities.setdefault(doc, 0)
                    query_docs_similarities[doc] += (
                        query_vector[query_key] * document_vectors[query_key][doc]
                    )

        if len(query_docs_similarities) > 0:
            query_docs_similarities = dict(
                sorted(
                    query_docs_similarities.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )
            )

        return query_docs_similarities

    def get_weight_by_document_id(
        self, document_id: str, inverted_file: Dict[str, Any]
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
            if document_id in file_value.keys():
                doc_dict[file_key] = file_value[document_id]
        return doc_dict


def print_inverted_file(inverted_file, indent=0):
    for key, value in inverted_file.items():
        print("  " * indent + str(key) + ":", end=" ")
        if isinstance(value, dict):
            print()
            print_inverted_file(value, indent + 1)
        else:
            print(str(value))


# TEST
if __name__ == "__main__":
    RS = RetrievalService()

    inverted_file = RS.create_inverted_file(
        {
            "1": "To do is to be. To be is to do.",
            "2": "To be or not to be. I am what I am.",
            "3": "I think therefore I am. Do be do be do.",
            "4": "Do do do, da da da. Let it be, let it be.",
        },
        False,
        False,
        {"tf_log": True, "use_idf": True},
    )

    print_inverted_file(inverted_file)

    query_vector = {
        "to": (1 + math.log2(1)) * math.log2(4 / 2),
        "do": (1 + math.log2(1)) * math.log2(4 / 3),
    }

    similarity = RS.calculate_similarity(query_vector, inverted_file)
    print(similarity)

    print(RS.get_weight_by_document_id("3", inverted_file))

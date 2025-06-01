"""
Text Preprocessing Utilities
-------------------------
Modul ini berisi fungsi-fungsi untuk preprocessing text:
1. Stemming (Bahasa Indonesia)
2. Stopword removal
3. Tokenization
"""

from typing import List, Set
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
            logger.warning(f"Could not load stopwords: {e}")
            _stop_words_cache = set()  # Fallback ke empty set
    return _stop_words_cache


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
        logger.warning(f"Error in tokenization, using simple split: {e}")
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

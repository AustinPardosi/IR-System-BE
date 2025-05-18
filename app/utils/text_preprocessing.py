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

logger = logging.getLogger(__name__)

# Placeholder untuk daftar stopwords
STOPWORDS: Set[str] = set()
stemmer = PorterStemmer()


def tokenize(text: str) -> List[str]:
    """
    Tokenisasi teks menjadi list token.

    Args:
        text: Teks yang akan ditokenisasi.

    Returns:
        List token hasil tokenisasi.
    """
    # Placeholder for implementation
    pass


def remove_stopwords(tokens: List[str]) -> List[str]:
    """
    Menghapus stopwords dari list token.

    Args:
        tokens: List token input.

    Returns:
        List token tanpa stopwords.
    """
    # Placeholder for implementation
    pass


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
    # Placeholder for implementation
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
        List token hasil preprocessing.
    """
    # Placeholder for implementation
    pass

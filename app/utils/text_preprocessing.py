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
nltk.download('punkt_tab')
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words("english"))

stemmer = PorterStemmer()


def tokenize(text: str) -> List[str]:
    """
    Tokenisasi teks menjadi list token.

    Args:
        text: Teks yang akan ditokenisasi.

    Returns:
        List token hasil tokenisasi.
    """
    tokenized_text = word_tokenize(text)
    return (tokenized_text)


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
        if (w not in stop_words):
            filtered_sent.append(w)
    
    return (filtered_sent)


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
    if (use_stemming):
        tokens = stem_tokens(tokens)
    if (use_stopword_removal):
        tokens = remove_stopwords(tokens)
    
    return ([token.lower() for token in tokens])
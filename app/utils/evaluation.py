"""
Evaluation Utilities
-----------------
Modul ini berisi fungsi-fungsi untuk evaluasi sistem IR:
1. Mean Average Precision (MAP)
2. Precision dan Recall
"""

from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


def calculate_precision_at_k (
    retrieved_docs: List[str], relevant_docs: List[str], k: int
) -> float:
    """
    Menghitung precision@k.

    Args:
        retrieved_docs: List dokumen yang di-retrieve.
        relevant_docs: List dokumen yang relevan.
        k: Posisi untuk menghitung precision.

    Returns:
        Nilai precision@k.
    """
    count = 0
    for index in range(0, k, 1):
        if (retrieved_docs[index] in relevant_docs):
            count = count + 1
    
    precision_at_k = count / k
    return (precision_at_k)


def calculate_average_precision (
    retrieved_docs: List[str], relevant_docs: List[str]
) -> float:
    """
    Menghitung average precision.

    Args:
        retrieved_docs: List dokumen yang di-retrieve.
        relevant_docs: List dokumen yang relevan.

    Returns:
        Nilai average precision.
    """
    sum_precisions = 0
    for idx, doc in enumerate(retrieved_docs):
        if doc in relevant_docs:
            sum_precisions = sum_precisions + calculate_precision_at_k (
                retrieved_docs, relevant_docs, idx + 1
            )
    
    average_precision = sum_precisions / len(relevant_docs)
    return (average_precision)


def calculate_map (
    all_retrieved_docs: Dict[str, List[str]], all_relevant_docs: Dict[str, List[str]]
) -> float:
    """
    Menghitung Mean Average Precision (MAP).

    Args:
        all_retrieved_docs: Dictionary mapping query ID ke list dokumen yang di-retrieve.
        all_relevant_docs: Dictionary mapping query ID ke list dokumen yang relevan.

    Returns:
        Nilai MAP.
    """
    sum_average_precisions = 0
    for retrieved_key, _ in all_retrieved_docs.items():
        if (retrieved_key not in all_relevant_docs.keys()):
            raise Exception("There is a query that doesn't have any relevant document")
        
        sum_average_precisions = sum_average_precisions + calculate_average_precision (
            all_retrieved_docs[retrieved_key], all_relevant_docs[retrieved_key]
        )
    
    mean_average_precision = sum_average_precisions / len(all_retrieved_docs.keys())
    return (mean_average_precision)






# TEST
# all_docs = ["1", "2", "3", "4", "5"]
# retrieved_docs = ["1", "3", "5"]
# relevant_docs = ["1", "4", "5"]

# print (calculate_precision_at_k(retrieved_docs, relevant_docs, 3))
# print (calculate_average_precision(retrieved_docs, relevant_docs))

# all_retrieved_docs = {
#     "A": ["1", "3", "5"],
#     "B": ["4", "2", "1", "6"]
# }

# all_relevant_docs = {
#     "A": ["1", "4", "5", "6"],
#     "B": ["2", "4", "3"]
# }

# print (calculate_map(all_retrieved_docs, all_relevant_docs))

# Output:
# 0.6666666666666666
# 0.5555555555555555
# 0.5416666666666666
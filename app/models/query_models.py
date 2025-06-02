"""
Query models untuk sistem Information Retrieval.
Berisi definisi model untuk query input dan output.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any


class QueryExpansionInput(BaseModel):
    """
    Model untuk query expansion input.
    """

    query: str = Field(..., description="Query text yang akan di-expand")
    threshold: float = Field(
        0.7, description="Threshold similarity untuk word expansion", ge=0.0, le=1.0
    )
    limit: int = Field(
        -1, description="Limit jumlah kata hasil expansion (-1 untuk unlimited)", ge=-1
    )


class InteractiveQueryInput(BaseModel):
    """
    Model untuk interactive query input.
    Digunakan untuk query yang diinput langsung oleh user.
    """

    query: str = Field(..., description="Query text yang akan dicari")
    use_stemming: bool = Field(True, description="Apakah akan menggunakan stemming")
    use_stopword_removal: bool = Field(
        True, description="Apakah akan menghilangkan stopwords"
    )
    weighting_method: Dict[str, bool] = Field(
        default_factory=lambda: {
            "tf_raw": True,
            "tf_log": False,
            "tf_binary": False,
            "tf_augmented": False,
            "use_idf": True,
            "use_normalization": True,
        },
        description="Metode pembobotan yang dipilih",
    )
    query_expansion_threshold: float = Field(
        0.7, description="Threshold untuk query expansion"
    )


class BatchQueryInput(BaseModel):
    """
    Model untuk batch query input.
    Digunakan untuk batch processing dari file query.
    """

    # Placeholder for batch query input
    pass


class RetrievalResult(BaseModel):
    """
    Model untuk hasil retrieval.
    """

    # Placeholder for retrieval result
    pass

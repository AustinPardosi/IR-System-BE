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


class DocumentRetrievalInput(BaseModel):
    """
    Model untuk document retrieval input.
    Digunakan untuk melakukan retrieval dokumen dengan inverted file yang sudah ada.
    """

    query: str = Field(..., description="Query text yang akan dicari")
    inverted_file: Dict[str, Dict[str, float]] = Field(
        ..., description="Inverted file berisi bobot term untuk setiap dokumen"
    )
    weighting_method: Dict[str, bool] = Field(
        ..., description="Metode pembobotan yang digunakan untuk query"
    )
    relevant_doc: List[int] = Field(
        default_factory=list, description="List ID dokumen yang relevan untuk evaluasi"
    )


class DocumentRetrievalInputSimple(BaseModel):
    """
    Model untuk document retrieval input yang menggunakan cached inverted file.
    Tidak perlu menyertakan inverted_file di body karena menggunakan cache.
    """

    query: str = Field(..., description="Query text yang akan dicari")
    weighting_method: Dict[str, bool] = Field(
        ..., description="Metode pembobotan yang digunakan untuk query"
    )
    relevant_doc: List[int] = Field(
        default_factory=list, description="List ID dokumen yang relevan untuk evaluasi"
    )


class BatchQueryInput(BaseModel):
    """
    Model untuk batch query input.
    Digunakan untuk batch processing dari file query.
    """

    # Placeholder for batch query input
    pass


class DocumentRetrievalResult(BaseModel):
    """
    Model untuk hasil document retrieval.
    """

    status: str = Field(..., description="Status operasi retrieval")
    ranked_documents: List[Dict[str, Any]] = Field(
        ...,
        description="List dokumen yang diurutkan berdasarkan similarity dengan format [{'id': str, 'similarity': float}]",
    )
    average_precision: float = Field(
        ...,
        description="Average precision untuk evaluasi (0 jika tidak ada relevant_doc)",
    )
    total_retrieved: int = Field(..., description="Total dokumen yang ditemukan")
    query_used: str = Field(..., description="Query yang digunakan untuk retrieval")


class RetrievalResult(BaseModel):
    """
    Model untuk hasil retrieval.
    """

    # Placeholder for retrieval result
    pass


class BatchRetrievalInput(BaseModel):
    """
    Model untuk batch retrieval input yang menggunakan cached inverted file.
    Menggunakan filepath dan relevant_doc dalam JSON body.
    """

    query_file: str = Field(
        ...,
        description="Filepath string ke file query (contoh: 'D://path/to/queries.xml')",
    )
    relevant_doc_filename: str = Field(
        ...,
        description="Filepath string ke file relevant docs (contoh: 'D://path/to/relevant_docs.json')",
    )
    weighting_method: Dict[str, bool] = Field(
        ..., description="Metode pembobotan yang digunakan untuk query"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "query_file": "D://path/to/queries.xml",
                "relevant_doc_filename": "D://path/to/relevant_docs.json",
                "weighting_method": {
                    "tf_raw": True,
                    "tf_log": False,
                    "tf_binary": False,
                    "tf_augmented": False,
                    "use_idf": True,
                    "use_normalization": True,
                },
            }
        }


class BatchRetrievalResult(BaseModel):
    """
    Model untuk hasil batch retrieval.
    """

    status: str = Field(..., description="Status operasi batch retrieval")
    total_queries: int = Field(..., description="Total query yang diproses")
    mean_average_precision: float = Field(
        ..., description="Mean Average Precision untuk semua query"
    )
    query_results: List[Dict[str, Any]] = Field(
        ..., description="Detail hasil retrieval untuk setiap query"
    )
    processing_info: Dict[str, Any] = Field(
        ..., description="Informasi proses batch retrieval"
    )

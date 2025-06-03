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
    use_stemming: bool = Field(
        True, description="Apakah menggunakan stemming dalam preprocessing query"
    )
    use_stopword_removal: bool = Field(
        True,
        description="Apakah menggunakan stopword removal dalam preprocessing query",
    )
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
    use_stemming: bool = Field(
        True, description="Apakah menggunakan stemming dalam preprocessing query"
    )
    use_stopword_removal: bool = Field(
        True,
        description="Apakah menggunakan stopword removal dalam preprocessing query",
    )
    weighting_method: Dict[str, bool] = Field(
        ..., description="Metode pembobotan yang digunakan untuk query"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "query_file": "D://path/to/queries.xml",
                "relevant_doc_filename": "D://path/to/relevant_docs.json",
                "use_stemming": True,
                "use_stopword_removal": True,
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


class QueryWeightInput(BaseModel):
    """
    Model untuk input perhitungan bobot query.
    """

    query: str = Field(..., description="Query text yang akan dihitung bobotnya")
    use_stemming: bool = Field(True, description="Apakah menggunakan stemming dalam preprocessing query")
    use_stopword_removal: bool = Field(True, description="Apakah menggunakan stopword removal dalam preprocessing query")
    weighting_method: Dict[str, bool] = Field(
        ..., description="Metode pembobotan yang digunakan untuk query"
    )


class QueryWeightResult(BaseModel):
    """
    Model untuk hasil perhitungan bobot query.
    """

    status: str = Field(..., description="Status operasi")
    query: str = Field(..., description="Query text yang diproses")
    query_vector: Dict[str, float] = Field(
        ..., description="Vector bobot query dengan format {term: weight}"
    )
    total_terms: int = Field(..., description="Total jumlah term dalam query vector")
    weighting_method: Dict[str, bool] = Field(
        ..., description="Metode pembobotan yang digunakan"
    )
    message: str = Field(..., description="Pesan informasi")


class BatchQueryExpansionInput(BaseModel):
    """
    Model untuk batch query expansion input.
    """

    query_file: str = Field(..., description="Path ke file query (format XML)")
    threshold: float = Field(
        0.7, description="Threshold similarity untuk word expansion", ge=0.0, le=1.0
    )
    limit: int = Field(
        -1, description="Limit jumlah kata hasil expansion (-1 untuk unlimited)", ge=-1
    )


class BatchQueryExpansionResult(BaseModel):
    """
    Model untuk hasil batch query expansion.
    """

    status: str = Field(..., description="Status operasi")
    total_queries: int = Field(..., description="Total query yang diproses")
    query_results: List[Dict[str, Any]] = Field(
        ..., description="Hasil expansion untuk setiap query"
    )
    parameters: Dict[str, Any] = Field(..., description="Parameter yang digunakan")
    processing_info: Dict[str, Any] = Field(..., description="Info tambahan proses")


class RetrieveDocumentsByIdsInput(BaseModel):
    """
    Model untuk input retrieve documents by IDs.
    """

    ids: List[str] = Field(..., description="List ID dokumen yang akan diambil")


class RetrieveDocumentsByIdsResult(BaseModel):
    """
    Model untuk hasil retrieve documents by IDs.
    """

    status: str = Field(..., description="Status operasi")
    total_requested: int = Field(..., description="Total ID yang diminta")
    total_found: int = Field(..., description="Total dokumen yang ditemukan")
    documents: List[Dict[str, Any]] = Field(
        ..., description="List dokumen yang ditemukan"
    )
    not_found_ids: List[str] = Field(..., description="List ID yang tidak ditemukan")
    message: str = Field(..., description="Pesan informasi")


class Word2VecRetrainingInput(BaseModel):
    """
    Model untuk input retraining Word2Vec dengan konfigurasi preprocessing custom.
    """

    use_stemming: bool = Field(
        True, description="Apakah menggunakan stemming dalam preprocessing dokumen"
    )
    use_stopword_removal: bool = Field(
        True,
        description="Apakah menggunakan stopword removal dalam preprocessing dokumen",
    )

    class Config:
        json_schema_extra = {
            "example": {"use_stemming": True, "use_stopword_removal": False}
        }


class Word2VecRetrainingResult(BaseModel):
    """
    Model untuk hasil retraining Word2Vec.
    """

    status: str = Field(..., description="Status operasi retraining")
    message: str = Field(..., description="Pesan informasi")
    training_info: Dict[str, Any] = Field(
        ..., description="Informasi detail hasil training"
    )
    previous_config: Dict[str, bool] = Field(
        ..., description="Konfigurasi preprocessing sebelumnya"
    )
    new_config: Dict[str, bool] = Field(
        ..., description="Konfigurasi preprocessing yang baru"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "message": "Word2Vec model berhasil dilatih ulang",
                "training_info": {
                    "vocabulary_size": 1250,
                    "preprocessing_config": {
                        "use_stemming": True,
                        "use_stopword_removal": False,
                    },
                    "total_documents": 1460,
                    "total_processed_sentences": 1460,
                },
                "previous_config": {"use_stemming": True, "use_stopword_removal": True},
                "new_config": {"use_stemming": True, "use_stopword_removal": False},
            }
        }

"""
Test untuk Query Expansion Service
"""

import os
import asyncio
from app.services.query_expansion_service import QueryExpansionService


def read_cisi_collection(file_path: str) -> dict:
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


def read_queries(file_path: str) -> dict:
    """
    Membaca file query dan mengembalikan dictionary query.
    """
    queries = {}
    current_id = None
    current_content = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith(".I "):
                # Simpan query sebelumnya jika ada
                if current_id is not None:
                    queries[current_id] = " ".join(current_content)
                current_id = line.strip().split(" ")[1]
                current_content = []
            elif line.startswith(".W"):
                continue
            else:
                current_content.append(line.strip())

    # Simpan query terakhir
    if current_id is not None:
        queries[current_id] = " ".join(current_content)

    return queries


async def main():
    # Inisialisasi service
    qe_service = QueryExpansionService()

    # Baca koleksi dokumen
    documents = read_cisi_collection("IRTestCollection/cisi.all")
    print(f"Loaded {len(documents)} documents")

    # Train model Word2Vec
    print("Training Word2Vec model...")
    await qe_service.train_word2vec_model(documents)

    # Baca queries
    queries = read_queries("IRTestCollection/query.text")
    print(f"Loaded {len(queries)} queries")

    # Test query expansion untuk beberapa query
    for query_id in list(queries.keys())[:5]:  # Test 5 query pertama
        query = queries[query_id]
        print(f"\nProcessing Query {query_id}:")
        print(f"Original Query: {query}")

        try:
            # Expand query
            result = await qe_service.expand_query(query, threshold=0.7)

            # Tampilkan hasil
            print("\nExpansion Results:")
            print(f"Original Terms: {result['original_terms']}")
            print("\nExpanded Terms:")
            for orig_term, similar_terms in result["expansion_terms"].items():
                print(f"\n{orig_term}:")
                for term in similar_terms:
                    print(f"  - {term['term']} (similarity: {term['similarity']:.3f})")

            print(f"\nFinal Expanded Terms: {result['expanded_terms']}")

        except Exception as e:
            print(f"Error processing query {query_id}: {str(e)}")


if __name__ == "__main__":
    asyncio.run(main())

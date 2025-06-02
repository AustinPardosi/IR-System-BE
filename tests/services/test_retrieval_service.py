import asyncio
import math
from typing import Dict, Any
import pytest

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from app.utils import text_preprocessing
from app.services.retrieval_service import RetrievalService
from collections import Counter

service = RetrievalService()
documents = {
            "1": "To do is to be. To be is to do.",
            "2": "To be or not to be. I am what I am.",
            "3": "I think therefore I am. Do be do be do.",
            "4": "Do do do, da da da. Let it be, let it be."
        }

freq_file = {}
for doc_key, doc_value in documents.items():
    tokens = text_preprocessing.preprocess_text(doc_value, False, False)
    tokens_freq = Counter(tokens)
    
    freq_file[doc_key] = tokens_freq

@pytest.mark.asyncio
async def test_inverted_file():
    inverted_file = await service.create_inverted_file (
        {
            "1": "To do is to be. To be is to do.",
            "2": "To be or not to be. I am what I am.",
            "3": "I think therefore I am. Do be do be do.",
            "4": "Do do do, da da da. Let it be, let it be."
        },
        False, False, {"tf_log": True, "use_idf": True}
    )
    assert inverted_file == {
        "to" : 
            {
                "1": 3.0,
                "2": 2.0
            },
        "do":
            {
                "1": 0.8300749985576875,
                "3": 1.072856372028895,
                "4": 1.072856372028895,
            },
        "is":
            {
                "1": 4.0,
            },
        "be":
            {
                "1": 0.0,
                "2": 0.0,
                "3": 0.0,
                "4": 0.0,
            },
        ".":
            {
                "1": 0.0,
                "2": 0.0,
                "3": 0.0,
                "4": 0.0,
            },
        "or":
            {
                "2": 2.0,
            },
        "not":
            {
                "2": 2.0,
            },
        "i":
            {
                "2": 2.0,
                "3": 2.0,
            },
        "am":
            {
                "2": 2.0,
                "3": 1.0,
            },
        "what":
            {
                "2": 2.0,
            },
        "think":
            {
                "3": 2.0,
            },
        "therefore":
            {
                "3": 2.0,
            },
        ",":
            {
                "4": 4.0
            },
        "da":
            {
                "4": 5.169925001442312,
            },
        "let":
            {
                "4": 4.0
            },
        "it":
            {
                "4": 4.0
            },
    }

# Kasus 1: Raw TF saja
@pytest.mark.asyncio
async def test_tf_idf_1():
    weight = await service.calculate_tf_idf ("to", "1", freq_file, {})
    assert weight["weight"] == 4

# Kasus 2: Log TF + IDF
@pytest.mark.asyncio
async def test_tf_idf_2():
    weight = await service.calculate_tf_idf ("am", "2", freq_file, {"tf_log": True})
    tf = 1 + math.log2(2)
    assert weight["weight"] == tf

# Kasus 3: Augmented TF
@pytest.mark.asyncio
async def test_tf_idf_3():
    weight = await service.calculate_tf_idf ("let", "4", freq_file, {"tf_augmented": True})
    tf = 0.5 + 0.5 * (2/3)
    assert weight['weight'] == tf

# Kasus 4: Binary TF
@pytest.mark.asyncio
async def test_tf_idf_4():
    weight = await service.calculate_tf_idf ("it", "4", freq_file, {"tf_binary": True})
    tf = 1
    assert weight['weight'] == tf

# Kasus 5: IDF
@pytest.mark.asyncio
async def test_tf_idf_5():
    weight = await service.calculate_tf_idf ("do", "4", freq_file, {"use_idf": True})
    idf = math.log2(4/3)
    tf = 3
    assert weight['weight'] == tf*idf

# Kasus 6: Normalization
@pytest.mark.asyncio
async def test_tf_idf_6():
    weight = await service.calculate_tf_idf ("do", "4", freq_file, {"use_normalization": True})
    normalization = 1/16
    tf = 3
    assert weight['weight'] == tf*normalization

# Kasus 7: Log TF + IDF
@pytest.mark.asyncio
async def test_tf_idf_7():
    weight = await service.calculate_tf_idf ("am", "2", freq_file, {"tf_log": True, "use_idf": True})
    tf = 1 + math.log2(2)
    idf = math.log(4/2, 2)
    assert weight["weight"] == tf*idf

# Kasus 8: Augmented TF + Normalization
@pytest.mark.asyncio
async def test_tf_idf_8():
    weight = await service.calculate_tf_idf ("therefore", "3", freq_file, {"tf_augmented": True, "use_normalization": True})
    tf = 0.5 + 0.5 * (1/3)
    normalization = 1/12
    assert weight["weight"] == tf*normalization

# Kasus 9: Binary TF + IDF + Normalization
@pytest.mark.asyncio
async def test_tf_idf_9():
    weight = await service.calculate_tf_idf ("think", "3", freq_file, {"tf_binary": True, "use_idf": True, "use_normalization": True})
    tf = 1
    idf = math.log2(4/1)
    normalization = 1/12
    assert weight["weight"] == tf*normalization*idf

# Retrieval Test
@pytest.mark.asyncio
async def test_retrieval():
    query = "information retrieval system"

    inverted_file = {
        "information": {"1": 0.3, "2": 0.5},
        "retrieval": {"1": 0.4},
        "system": {"2": 0.6, "3": 0.2},
    }

    documents = {
        "1": {"information": 2, "retrieval": 3},
        "2": {"information": 5, "system": 4},
        "3": {"system": 1},
    }

    relevant_doc = ['1', '2']

    weighting_method = {
        "tf_raw": False,
        "tf_logarithmic": True,
        "tf_binary": False,
        "tf_augmented": False,
        "use_idf": True,
        "use_normalization": True,
    }

    docs, ap = await service.retrieve_document(query,inverted_file,weighting_method,documents,relevant_doc)
    assert relevant_doc[0] == docs[0]
    assert relevant_doc[1] == docs[1]
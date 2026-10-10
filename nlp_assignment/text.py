"""Shared text preprocessing and document-vector utilities."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence

import numpy as np

TOKEN_PATTERN = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+")


def tokenize(text: object) -> list[str]:
    """Lowercase and tokenize English text without external tokenizer data."""
    if text is None:
        return []
    return TOKEN_PATTERN.findall(str(text).lower())


def mean_document_vector(
    tokens: Sequence[str], vectors: Mapping[str, np.ndarray], vector_size: int
) -> tuple[np.ndarray, int, int]:
    """Return a mean embedding and (in-vocabulary, total-token) counts."""
    found = [np.asarray(vectors[token], dtype=np.float32) for token in tokens if token in vectors]
    if not found:
        return np.zeros(vector_size, dtype=np.float32), 0, len(tokens)
    return np.mean(found, axis=0, dtype=np.float32), len(found), len(tokens)


def vectorize_documents(
    tokenized_documents: Iterable[Sequence[str]],
    vectors: Mapping[str, np.ndarray],
    vector_size: int,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Average word vectors for documents and report token/document coverage."""
    rows: list[np.ndarray] = []
    known_tokens = 0
    total_tokens = 0
    zero_documents = 0
    for tokens in tokenized_documents:
        row, known, total = mean_document_vector(tokens, vectors, vector_size)
        rows.append(row)
        known_tokens += known
        total_tokens += total
        zero_documents += int(known == 0)

    matrix = (
        np.vstack(rows).astype(np.float32, copy=False)
        if rows
        else np.empty((0, vector_size), dtype=np.float32)
    )
    coverage = known_tokens / total_tokens if total_tokens else 0.0
    return matrix, {
        "known_tokens": known_tokens,
        "total_tokens": total_tokens,
        "token_coverage": coverage,
        "zero_vector_documents": zero_documents,
    }


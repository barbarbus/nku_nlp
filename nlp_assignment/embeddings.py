"""Task 2: GloVe and Word2Vec document representations."""

from __future__ import annotations

import io
import hashlib
import zipfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .common import evaluate_classifier, train_logistic_regression
from .text import tokenize, vectorize_documents


def stable_word_hash(value: str) -> int:
    """Stable hash used by Gensim initialization across Python processes."""
    data = value.encode("utf-8") if isinstance(value, str) else bytes(value)
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "little")


def _iter_glove_lines(path: Path, vector_size: int) -> Iterable[str]:
    if path.suffix.lower() == ".zip":
        member_name = f"glove.6B.{vector_size}d.txt"
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            match = next((name for name in names if name.endswith(member_name)), None)
            if match is None:
                raise ValueError(f"{path} does not contain {member_name}")
            with archive.open(match) as binary:
                with io.TextIOWrapper(binary, encoding="utf-8") as text:
                    yield from text
    else:
        with path.open("r", encoding="utf-8") as handle:
            yield from handle


def load_glove(path: str | Path, needed_words: set[str], vector_size: int = 100):
    """Load needed vectors from TXT, glove.6B.zip, or Gensim KeyedVectors."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"GloVe file not found: {path}")
    if path.suffix.lower() == ".kv":
        from gensim.models import KeyedVectors

        keyed_vectors = KeyedVectors.load(str(path), mmap="r")
        if keyed_vectors.vector_size != vector_size:
            raise ValueError(
                f"Expected {vector_size}-dimensional GloVe vectors, got "
                f"{keyed_vectors.vector_size}."
            )
        return {
            word: np.asarray(keyed_vectors[word], dtype=np.float32)
            for word in needed_words
            if word in keyed_vectors
        }
    vectors: dict[str, np.ndarray] = {}
    for line_number, line in enumerate(_iter_glove_lines(path, vector_size), start=1):
        pieces = line.rstrip().split(" ")
        if len(pieces) != vector_size + 1:
            continue
        word = pieces[0]
        if word not in needed_words:
            continue
        try:
            vectors[word] = np.asarray(pieces[1:], dtype=np.float32)
        except ValueError as exc:
            raise ValueError(f"Invalid numeric vector on GloVe line {line_number}") from exc
    return vectors


def _fit_and_evaluate_vectors(
    method: str,
    splits,
    encoded_labels,
    keyed_vectors,
    vector_size: int,
    seed: int,
    c_value: float,
) -> dict[str, Any]:
    tokenized = {
        name: [tokenize(text) for text in frame["text"]]
        for name, frame in splits.items()
    }
    x_train, train_coverage = vectorize_documents(
        tokenized["train"], keyed_vectors, vector_size
    )
    x_test, test_coverage = vectorize_documents(tokenized["test"], keyed_vectors, vector_size)
    model = train_logistic_regression(x_train, encoded_labels["train"], seed, c_value)
    return {
        "method": method,
        "metrics": evaluate_classifier(model, x_test, encoded_labels["test"]),
        "details": {
            "vector_size": vector_size,
            "classifier": "LogisticRegression",
            "train_coverage": train_coverage,
            "test_coverage": test_coverage,
        },
    }


def run_glove_experiment(
    splits,
    encoded_labels,
    glove_path: str | Path,
    seed: int,
    vector_size: int = 100,
    c_value: float = 1.0,
) -> dict[str, Any]:
    needed_words = {
        token
        for frame in splits.values()
        for text in frame["text"]
        for token in tokenize(text)
    }
    vectors = load_glove(glove_path, needed_words, vector_size)
    return _fit_and_evaluate_vectors(
        "pretrained_glove", splits, encoded_labels, vectors, vector_size, seed, c_value
    )


def train_word2vec(
    sentences: Iterable[list[str]],
    seed: int,
    vector_size: int = 100,
    epochs: int = 10,
    min_count: int = 2,
    workers: int = 1,
):
    from gensim.models import Word2Vec

    return Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=5,
        min_count=min_count,
        workers=workers,
        sg=1,
        seed=seed,
        epochs=epochs,
        hashfxn=stable_word_hash,
    )


def run_word2vec_experiment(
    method: str,
    training_texts: Iterable[str],
    splits,
    encoded_labels,
    seed: int,
    model_output: str | Path | None = None,
    vector_size: int = 100,
    epochs: int = 10,
    min_count: int = 2,
    workers: int = 1,
    c_value: float = 1.0,
) -> dict[str, Any]:
    sentences = [tokens for text in training_texts if (tokens := tokenize(text))]
    if not sentences:
        raise ValueError("No usable tokens were found for Word2Vec training.")
    model = train_word2vec(sentences, seed, vector_size, epochs, min_count, workers)
    if model_output is not None:
        model_output = Path(model_output)
        model_output.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(model_output))
    record = _fit_and_evaluate_vectors(
        method, splits, encoded_labels, model.wv, vector_size, seed, c_value
    )
    record["details"].update(
        {
            "word2vec_epochs": epochs,
            "word2vec_min_count": min_count,
            "word2vec_vocabulary_size": len(model.wv),
        }
    )
    return record

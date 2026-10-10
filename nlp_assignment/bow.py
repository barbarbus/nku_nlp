"""Task 1: Binary Bag of Words and word-frequency classifiers."""

from __future__ import annotations

from typing import Any

import numpy as np

from .common import evaluate_classifier, train_logistic_regression
from .text import tokenize


def run_bow_experiments(
    splits,
    encoded_labels,
    seed: int,
    min_df: int = 1,
    max_features: int | None = None,
    c_value: float = 1.0,
) -> list[dict[str, Any]]:
    from sklearn.feature_extraction.text import CountVectorizer

    records = []
    for name, binary in (("binary_bow", True), ("word_frequency", False)):
        vectorizer = CountVectorizer(
            tokenizer=tokenize,
            token_pattern=None,
            lowercase=False,
            binary=binary,
            min_df=min_df,
            max_features=max_features,
            dtype=np.int32,
        )
        x_train = vectorizer.fit_transform(splits["train"]["text"])
        x_test = vectorizer.transform(splits["test"]["text"])
        model = train_logistic_regression(x_train, encoded_labels["train"], seed, c_value)
        records.append(
            {
                "method": name,
                "metrics": evaluate_classifier(model, x_test, encoded_labels["test"]),
                "details": {
                    "vocabulary_size": len(vectorizer.vocabulary_),
                    "binary": binary,
                    "classifier": "LogisticRegression",
                },
            }
        )
    return records


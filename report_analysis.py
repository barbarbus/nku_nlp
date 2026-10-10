"""Read-only diagnostics used by the extended assignment report.

Run from the project directory with ``python report_analysis.py``.  The
original experiment outputs are never modified.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from nlp_assignment.common import train_logistic_regression
from nlp_assignment.text import tokenize


ROOT = Path(__file__).resolve().parent
SPLITS = ROOT / "outputs" / "splits"
LABELS = ["business", "politics", "sports"]


def load_splits() -> dict[str, pd.DataFrame]:
    return {
        name: pd.read_csv(SPLITS / f"{name}.csv")
        for name in ("train", "validation", "test")
    }


def fit_bow(train: pd.DataFrame, test: pd.DataFrame, binary: bool) -> dict:
    vectorizer = CountVectorizer(
        tokenizer=tokenize,
        token_pattern=None,
        lowercase=False,
        binary=binary,
        dtype=np.int32,
    )
    x_train = vectorizer.fit_transform(train["text"])
    x_test = vectorizer.transform(test["text"])
    model = train_logistic_regression(x_train, train["label"], seed=42, c_value=1.0)
    predictions = model.predict(x_test)
    report = classification_report(
        test["label"], predictions, labels=LABELS, output_dict=True, zero_division=0
    )
    overlap = test["text"].isin(set(train["text"])).to_numpy()
    novel_true = test["label"].to_numpy()[~overlap]
    novel_pred = predictions[~overlap]
    result = {
        "accuracy": float(np.mean(predictions == test["label"].to_numpy())),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "per_class": {label: report[label] for label in LABELS},
        "confusion_matrix": confusion_matrix(
            test["label"], predictions, labels=LABELS
        ).tolist(),
        "test_train_overlap_correct": int(
            np.sum(predictions[overlap] == test["label"].to_numpy()[overlap])
        ),
        "novel_test_size": int((~overlap).sum()),
        "novel_test_accuracy": float(np.mean(novel_pred == novel_true)),
        "novel_test_macro_f1": float(
            f1_score(novel_true, novel_pred, labels=LABELS, average="macro")
        ),
    }
    if not binary:
        terms = vectorizer.get_feature_names_out()
        result["top_positive_terms"] = {
            label: terms[np.argsort(model.coef_[index])[-10:][::-1]].tolist()
            for index, label in enumerate(model.classes_)
        }
    return result


def main() -> None:
    splits = load_splits()
    train, validation, test = (splits[name] for name in ("train", "validation", "test"))
    all_text = pd.concat([frame["text"] for frame in splits.values()], ignore_index=True)
    overlap = {
        f"{left}_{right}": len(set(splits[left]["text"]) & set(splits[right]["text"]))
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))
    }
    majority = int(test["label"].value_counts().max())
    baseline_accuracy = majority / len(test)
    baseline_sports_f1 = 2 * baseline_accuracy / (1 + baseline_accuracy)
    result = {
        "unique_texts": int(all_text.nunique()),
        "duplicate_rows_within_splits": {
            name: int(frame["text"].duplicated().sum())
            for name, frame in splits.items()
        },
        "cross_split_unique_text_overlaps": overlap,
        "majority_sports_baseline": {
            "accuracy": baseline_accuracy,
            "macro_f1": baseline_sports_f1 / len(LABELS),
        },
        "binary_bow": fit_bow(train, test, binary=True),
        "word_frequency": fit_bow(train, test, binary=False),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

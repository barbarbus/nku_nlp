"""Read-only, prediction-level audit of the six completed homework methods.

Reconstruct traditional classifiers from the saved data split and saved word
vectors, and run inference with the saved BERT model. Every reconstructed
Accuracy and Macro-F1 must match outputs/results.json before diagnostics are
written to outputs/report_diagnostics.json. No original result is changed.
"""

from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder

from nlp_assignment.common import train_logistic_regression
from nlp_assignment.text import tokenize, vectorize_documents


ROOT = Path(__file__).resolve().parent
SPLITS = ROOT / "outputs" / "splits"
LABELS = ["business", "politics", "sports"]
ORDER = [
    "binary_bow",
    "word_frequency",
    "pretrained_glove",
    "word2vec_ag_news",
    "word2vec_nyt_train",
    "bert_base_uncased",
]


def announce(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def predict_bow(train: pd.DataFrame, test: pd.DataFrame, y_train: np.ndarray, binary: bool) -> np.ndarray:
    vectorizer = CountVectorizer(
        tokenizer=tokenize,
        token_pattern=None,
        lowercase=False,
        binary=binary,
        dtype=np.int32,
    )
    x_train = vectorizer.fit_transform(train["text"])
    x_test = vectorizer.transform(test["text"])
    model = train_logistic_regression(x_train, y_train, seed=42, c_value=1.0)
    result = model.predict(x_test)
    del vectorizer, x_train, x_test, model
    gc.collect()
    return result


def predict_vectors(
    train_tokens: list[list[str]],
    test_tokens: list[list[str]],
    y_train: np.ndarray,
    vectors,
) -> tuple[np.ndarray, dict]:
    x_train, train_coverage = vectorize_documents(train_tokens, vectors, 100)
    x_test, test_coverage = vectorize_documents(test_tokens, vectors, 100)
    model = train_logistic_regression(x_train, y_train, seed=42, c_value=1.0)
    result = model.predict(x_test)
    del x_train, x_test, model
    gc.collect()
    return result, {"train": train_coverage, "test": test_coverage}


def predict_bert(texts: list[str]) -> tuple[np.ndarray, dict]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, logging

    logging.set_verbosity_error()
    model_dir = ROOT / "outputs" / "bert_model"
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    lengths = np.asarray(
        [len(tokenizer(text, truncation=False)["input_ids"]) for text in texts],
        dtype=np.int32,
    )
    predictions: list[int] = []
    with torch.inference_mode():
        for start in range(0, len(texts), 32):
            batch = tokenizer(
                texts[start : start + 32],
                truncation=True,
                max_length=64,
                padding=True,
                return_tensors="pt",
            )
            logits = model(**{key: value.to(device) for key, value in batch.items()}).logits
            predictions.extend(logits.argmax(dim=-1).cpu().tolist())
    statistics = {
        "all_test_documents_truncated": bool(np.all(lengths > 64)),
        "untruncated_length_quantiles": {
            str(percentile): float(np.percentile(lengths, percentile))
            for percentile in (0, 25, 50, 75, 90, 100)
        },
        "median_fraction_retained": float(np.median(64 / lengths)),
    }
    return np.asarray(predictions, dtype=np.int64), statistics


def summarize_predictions(y_true: np.ndarray, y_pred: np.ndarray, novel_mask: np.ndarray) -> dict:
    report = classification_report(
        y_true, y_pred, labels=np.arange(len(LABELS)), output_dict=True, zero_division=0
    )
    per_class = {
        label: {
            "precision": float(report[str(index)]["precision"]),
            "recall": float(report[str(index)]["recall"]),
            "f1": float(report[str(index)]["f1-score"]),
            "support": int(report[str(index)]["support"]),
        }
        for index, label in enumerate(LABELS)
    }
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "correct": int(np.sum(y_true == y_pred)),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=np.arange(len(LABELS))).tolist(),
        "train_overlap_correct": int(np.sum(y_true[~novel_mask] == y_pred[~novel_mask])),
        "novel_test_size": int(np.sum(novel_mask)),
        "novel_test_accuracy": float(accuracy_score(y_true[novel_mask], y_pred[novel_mask])),
        "novel_test_macro_f1": float(f1_score(y_true[novel_mask], y_pred[novel_mask], average="macro")),
    }


def main() -> None:
    train = pd.read_csv(SPLITS / "train.csv")
    test = pd.read_csv(SPLITS / "test.csv")
    encoder = LabelEncoder().fit(train["label"])
    assert encoder.classes_.tolist() == LABELS
    y_train = encoder.transform(train["label"])
    y_test = encoder.transform(test["label"])
    novel_mask = ~test["text"].isin(set(train["text"])).to_numpy()
    predictions: dict[str, np.ndarray] = {}
    coverage: dict[str, dict] = {}

    for name, binary in (("binary_bow", True), ("word_frequency", False)):
        announce(f"Reconstructing {name}")
        predictions[name] = predict_bow(train, test, y_train, binary)

    announce("Tokenizing documents for embedding methods")
    train_tokens = [tokenize(text) for text in train["text"]]
    test_tokens = [tokenize(text) for text in test["text"]]
    from gensim.models import KeyedVectors, Word2Vec

    sources = (
        ("pretrained_glove", ROOT / ".build" / "assets" / "glove-wiki-gigaword-100.kv", True),
        ("word2vec_ag_news", ROOT / "outputs" / "models" / "word2vec_ag_news.model", False),
        ("word2vec_nyt_train", ROOT / "outputs" / "models" / "word2vec_nyt.model", False),
    )
    for name, path, is_keyed_vectors in sources:
        announce(f"Evaluating {name}")
        vectors = KeyedVectors.load(str(path), mmap="r") if is_keyed_vectors else Word2Vec.load(str(path)).wv
        predictions[name], coverage[name] = predict_vectors(train_tokens, test_tokens, y_train, vectors)
        del vectors
        gc.collect()
    del train_tokens, test_tokens
    gc.collect()

    announce("Evaluating saved BERT checkpoint")
    predictions["bert_base_uncased"], bert_lengths = predict_bert(test["text"].tolist())
    original = {entry["method"]: entry for entry in json.loads((ROOT / "outputs" / "results.json").read_text(encoding="utf-8"))}
    summaries = {}
    for name in ORDER:
        summary = summarize_predictions(y_test, predictions[name], novel_mask)
        for metric in ("accuracy", "macro_f1"):
            if abs(summary[metric] - original[name]["metrics"][metric]) > 1e-8:
                raise ValueError(f"{name} did not reproduce original {metric}: {summary[metric]}")
        summaries[name] = summary

    best_correct = predictions["word_frequency"] == y_test
    comparisons = {}
    for name in ORDER:
        if name == "word_frequency":
            continue
        other_correct = predictions[name] == y_test
        best_only = int(np.sum(best_correct & ~other_correct))
        other_only = int(np.sum(~best_correct & other_correct))
        comparisons[name] = {
            "word_frequency_only_correct": best_only,
            "other_only_correct": other_only,
            "mcnemar_exact_two_sided_p": float(binomtest(min(best_only, other_only), best_only + other_only, 0.5).pvalue),
        }
    running_adjusted = 0.0
    names_by_p = sorted(comparisons, key=lambda name: comparisons[name]["mcnemar_exact_two_sided_p"])
    for rank, name in enumerate(names_by_p):
        unadjusted = comparisons[name]["mcnemar_exact_two_sided_p"]
        running_adjusted = max(running_adjusted, min(1.0, unadjusted * (len(names_by_p) - rank)))
        comparisons[name]["holm_adjusted_p"] = running_adjusted
    error_count_per_test_document = np.sum(
        np.column_stack([predictions[name] != y_test for name in ORDER]), axis=1
    )
    payload = {
        "source_results": "outputs/results.json",
        "test_size": len(test),
        "class_order": LABELS,
        "train_test_exact_text_overlap": int(np.sum(~novel_mask)),
        "methods": summaries,
        "embedding_token_coverage": coverage,
        "bert_test_token_lengths": bert_lengths,
        "paired_vs_word_frequency": comparisons,
        "test_documents_by_number_of_method_errors": {
            str(count): int(np.sum(error_count_per_test_document == count)) for count in range(7)
        },
    }
    destination = ROOT / "outputs" / "report_diagnostics.json"
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    announce(f"Verified all six methods and wrote {destination}")


if __name__ == "__main__":
    main()

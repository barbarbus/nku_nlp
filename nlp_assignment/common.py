"""Shared classification, metric, and result helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .constants import METHOD_ORDER


def encode_labels(splits: dict[str, pd.DataFrame]):
    from sklearn.preprocessing import LabelEncoder

    encoder = LabelEncoder()
    encoder.fit(splits["train"]["label"])
    known = set(encoder.classes_)
    for split_name in ("validation", "test"):
        unseen = set(splits[split_name]["label"]) - known
        if unseen:
            raise ValueError(f"{split_name} contains labels absent from training: {sorted(unseen)}")
    encoded = {
        name: encoder.transform(frame["label"]).astype(np.int64)
        for name, frame in splits.items()
    }
    return encoder, encoded


def train_logistic_regression(x_train, y_train, seed: int, c_value: float = 1.0):
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(
        C=c_value,
        max_iter=2000,
        solver="lbfgs",
        random_state=seed,
    )
    model.fit(x_train, y_train)
    return model


def classification_metrics(y_true, y_pred) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, f1_score

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def evaluate_classifier(model, x_test, y_test) -> dict[str, float]:
    return classification_metrics(y_test, model.predict(x_test))


def save_json(payload: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def save_results(records: list[dict[str, Any]], output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "results.json"
    merged: dict[str, dict[str, Any]] = {}
    if json_path.is_file():
        try:
            previous = json.loads(json_path.read_text(encoding="utf-8"))
            merged.update({record["method"]: record for record in previous})
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
    merged.update({record["method"]: record for record in records})
    order = {method: index for index, method in enumerate(METHOD_ORDER)}
    all_records = sorted(
        merged.values(), key=lambda record: order.get(record["method"], len(order))
    )
    save_json(all_records, json_path)
    flat = []
    for record in all_records:
        row = {
            "method": record["method"],
            "accuracy": record["metrics"]["accuracy"],
            "macro_f1": record["metrics"]["macro_f1"],
        }
        flat.append(row)
    pd.DataFrame(flat).to_csv(
        output_dir / "results.csv", index=False, encoding="utf-8", float_format="%.6f"
    )

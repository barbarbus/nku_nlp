"""Validate that generated artifacts satisfy the assignment's code requirements."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from nlp_assignment.constants import (
    BERT_EPOCHS,
    BERT_MAX_LENGTH,
    BERT_MODEL_NAME,
    METHOD_ORDER,
    WORD_VECTOR_SIZE,
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def count_csv_rows(path: Path) -> int:
    require(path.is_file(), f"Missing split file: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def validate(output_dir: Path) -> list[str]:
    results_path = output_dir / "results.json"
    config_path = output_dir / "run_config.json"
    require(results_path.is_file(), f"Missing results: {results_path}")
    require(config_path.is_file(), f"Missing run configuration: {config_path}")

    results = json.loads(results_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    methods = [record["method"] for record in results]
    require(methods == list(METHOD_ORDER), f"Expected methods {METHOD_ORDER}, got {methods}")

    for record in results:
        for metric in ("accuracy", "macro_f1"):
            value = record["metrics"].get(metric)
            require(isinstance(value, (int, float)), f"{record['method']} lacks {metric}")
            require(0.0 <= value <= 1.0, f"Invalid {metric} for {record['method']}: {value}")

    constraints = config.get("assignment_constraints", {})
    expected_constraints = {
        "word_vector_size": WORD_VECTOR_SIZE,
        "bert_model": BERT_MODEL_NAME,
        "bert_max_length": BERT_MAX_LENGTH,
        "bert_epochs": BERT_EPOCHS,
    }
    require(
        constraints == expected_constraints,
        f"Assignment constraints differ: expected {expected_constraints}, got {constraints}",
    )

    sizes = config["split_sizes"]
    actual_sizes = {
        name: count_csv_rows(output_dir / "splits" / f"{name}.csv")
        for name in ("train", "validation", "test")
    }
    require(actual_sizes == sizes, f"Split sizes differ: config={sizes}, files={actual_sizes}")
    require(sum(sizes.values()) == config["total_rows"], "Split rows do not cover all NYT rows")
    require(sizes["validation"] == sizes["test"], "Validation and test must each be 10%")
    train_ratio = sizes["train"] / config["total_rows"]
    require(abs(train_ratio - 0.8) < 0.001, f"Training ratio is not 80%: {train_ratio:.6f}")

    required_artifacts = (
        output_dir / "models" / "word2vec_ag_news.model",
        output_dir / "models" / "word2vec_nyt.model",
        output_dir / "bert_model" / "config.json",
        output_dir / "bert_model" / "model.safetensors",
        output_dir / "bert_model" / "tokenizer.json",
    )
    for artifact in required_artifacts:
        require(artifact.is_file() and artifact.stat().st_size > 0, f"Missing artifact: {artifact}")

    return [
        "Six required experiments are present in assignment order",
        "Accuracy and Macro-F1 are valid for every experiment",
        "NYT split files match the recorded 80/10/10 split",
        "100d Word2Vec/GloVe and required BERT settings are locked",
        "Word2Vec and final BERT model artifacts are present",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()
    try:
        messages = validate(Path(args.output_dir))
    except (KeyError, json.JSONDecodeError, RuntimeError) as exc:
        parser.error(str(exc))
    for message in messages:
        print(f"PASS: {message}")


if __name__ == "__main__":
    main()


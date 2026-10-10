"""CLI orchestration for all assignment experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .common import encode_labels, save_json, save_results
from .constants import (
    BERT_EPOCHS,
    BERT_MAX_LENGTH,
    BERT_MODEL_NAME,
    DEFAULT_SEED,
    WORD_VECTOR_SIZE,
)
from .data import (
    file_sha256,
    load_ag_news_texts,
    load_nyt,
    save_splits,
    set_global_seed,
    split_nyt,
)


TASKS = ("bow", "glove", "w2v-ag", "w2v-nyt", "bert", "all")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the NLP assignment's BoW, embedding, and BERT experiments."
    )
    parser.add_argument("--task", choices=TASKS, default="all")
    parser.add_argument(
        "--nyt",
        default="HW-1/nyt.csv",
        help="NYT CSV/JSON/JSONL/Parquet path (default: HW-1/nyt.csv)",
    )
    parser.add_argument("--ag-news", help="Optional local AG News file; downloads it if omitted")
    parser.add_argument("--glove", help="glove.6B.zip or glove.6B.100d.txt path")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--min-df", type=int, default=1)
    parser.add_argument("--max-features", type=int)
    parser.add_argument("--logreg-c", type=float, default=1.0)
    parser.add_argument("--w2v-epochs", type=int, default=10)
    parser.add_argument("--w2v-min-count", type=int, default=2)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Word2Vec workers; 1 gives the strongest reproducibility",
    )
    parser.add_argument("--bert-batch-size", type=int, default=16)
    parser.add_argument("--bert-learning-rate", type=float, default=2e-5)
    parser.add_argument("--bert-model", default=BERT_MODEL_NAME)
    return parser


def _build_run_config(args, nyt, splits, label_encoder) -> dict:
    nyt_path = Path(args.nyt).resolve()
    identity = {
        "seed": args.seed,
        "nyt_sha256": file_sha256(nyt_path),
        "text_column": args.text_column,
        "label_column": args.label_column,
    }
    experiment_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    return {
        "experiment_id": experiment_id,
        **identity,
        "nyt_path": str(nyt_path),
        "total_rows": len(nyt),
        "split_sizes": {name: len(frame) for name, frame in splits.items()},
        "split_label_counts": {
            name: {
                str(label): int(count)
                for label, count in frame["label"].value_counts().sort_index().items()
            }
            for name, frame in splits.items()
        },
        "labels": [str(label) for label in label_encoder.classes_],
        "assignment_constraints": {
            "word_vector_size": WORD_VECTOR_SIZE,
            "bert_model": BERT_MODEL_NAME,
            "bert_max_length": BERT_MAX_LENGTH,
            "bert_epochs": BERT_EPOCHS,
        },
    }


def _validate_result_compatibility(output_dir: Path, run_config: dict) -> None:
    config_path = output_dir / "run_config.json"
    results_path = output_dir / "results.json"
    if not (config_path.is_file() and results_path.is_file()):
        return
    try:
        previous = json.loads(config_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return
    previous_id = previous.get("experiment_id")
    if previous_id and previous_id != run_config["experiment_id"]:
        raise ValueError(
            "This output directory contains results from a different dataset, seed, or "
            "column configuration. Use a new --output-dir to keep experiments comparable."
        )


def run(args: argparse.Namespace) -> list[dict]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_global_seed(args.seed)
    nyt = load_nyt(args.nyt, args.text_column, args.label_column)
    splits = split_nyt(nyt, args.seed)
    label_encoder, encoded_labels = encode_labels(splits)
    run_config = _build_run_config(args, nyt, splits, label_encoder)
    _validate_result_compatibility(output_dir, run_config)
    save_json(run_config, output_dir / "run_config.json")
    save_splits(splits, output_dir / "splits")

    selected = set(TASKS[:-1]) if args.task == "all" else {args.task}
    records: list[dict] = []
    if "bow" in selected:
        from .bow import run_bow_experiments

        records.extend(
            run_bow_experiments(
                splits,
                encoded_labels,
                args.seed,
                args.min_df,
                args.max_features,
                args.logreg_c,
            )
        )
        save_results(records, output_dir)

    if "glove" in selected:
        if not args.glove:
            raise ValueError("--glove is required for task 'glove' or 'all'.")
        from .embeddings import run_glove_experiment

        records.append(
            run_glove_experiment(
                splits,
                encoded_labels,
                args.glove,
                args.seed,
                WORD_VECTOR_SIZE,
                args.logreg_c,
            )
        )
        save_results(records, output_dir)

    if "w2v-ag" in selected:
        from .embeddings import run_word2vec_experiment

        ag_texts = load_ag_news_texts(args.ag_news)
        records.append(
            run_word2vec_experiment(
                "word2vec_ag_news",
                ag_texts,
                splits,
                encoded_labels,
                args.seed,
                output_dir / "models" / "word2vec_ag_news.model",
                WORD_VECTOR_SIZE,
                args.w2v_epochs,
                args.w2v_min_count,
                args.workers,
                args.logreg_c,
            )
        )
        save_results(records, output_dir)

    if "w2v-nyt" in selected:
        from .embeddings import run_word2vec_experiment

        records.append(
            run_word2vec_experiment(
                "word2vec_nyt_train",
                splits["train"]["text"],
                splits,
                encoded_labels,
                args.seed,
                output_dir / "models" / "word2vec_nyt.model",
                WORD_VECTOR_SIZE,
                args.w2v_epochs,
                args.w2v_min_count,
                args.workers,
                args.logreg_c,
            )
        )
        save_results(records, output_dir)

    if "bert" in selected:
        from .bert import run_bert_experiment

        records.append(
            run_bert_experiment(
                splits,
                encoded_labels,
                label_encoder,
                output_dir,
                args.seed,
                args.bert_model,
                max_length=BERT_MAX_LENGTH,
                epochs=BERT_EPOCHS,
                batch_size=args.bert_batch_size,
                learning_rate=args.bert_learning_rate,
            )
        )
        save_results(records, output_dir)
    return records


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        records = run(args)
    except (FileNotFoundError, ImportError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(records, ensure_ascii=False, indent=2))

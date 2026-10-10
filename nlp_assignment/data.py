"""Dataset loading, validation, and reproducible splitting."""

from __future__ import annotations

import json
import random
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Hash an input file so independently run tasks can prove data identity."""
    digest = sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".json", ".jsonl"}:
        try:
            return pd.read_json(path, lines=suffix == ".jsonl")
        except ValueError:
            return pd.DataFrame(json.loads(path.read_text(encoding="utf-8")))
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported dataset format: {suffix}. Use CSV, JSON, JSONL, or Parquet.")


def load_nyt(path: str | Path, text_column: str = "text", label_column: str = "label") -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"NYT dataset not found: {path}")
    frame = _read_table(path)
    missing = {text_column, label_column} - set(frame.columns)
    if missing:
        raise ValueError(f"NYT dataset is missing required column(s): {sorted(missing)}")
    frame = frame[[text_column, label_column]].rename(
        columns={text_column: "text", label_column: "label"}
    )
    frame = frame.dropna(subset=["text", "label"]).copy()
    frame["text"] = frame["text"].astype(str)
    frame = frame[frame["text"].str.strip().ne("")].reset_index(drop=True)
    if frame.empty:
        raise ValueError("NYT dataset has no usable rows after removing missing/empty values.")
    return frame


def split_nyt(frame: pd.DataFrame, seed: int = 42) -> dict[str, pd.DataFrame]:
    """Create one stratified 80/10/10 split shared by every experiment."""
    from sklearn.model_selection import train_test_split

    counts = frame["label"].value_counts()
    too_small = counts[counts < 10]
    if not too_small.empty:
        details = ", ".join(f"{label!r}: {count}" for label, count in too_small.items())
        raise ValueError(
            "Each label needs at least 10 examples for a reliable stratified 80/10/10 split. "
            f"Too-small labels: {details}"
        )

    train, remainder = train_test_split(
        frame,
        test_size=0.20,
        random_state=seed,
        shuffle=True,
        stratify=frame["label"],
    )
    validation, test = train_test_split(
        remainder,
        test_size=0.50,
        random_state=seed,
        shuffle=True,
        stratify=remainder["label"],
    )
    return {
        "train": train.reset_index(drop=True),
        "validation": validation.reset_index(drop=True),
        "test": test.reset_index(drop=True),
    }


def save_splits(splits: dict[str, pd.DataFrame], output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in splits.items():
        frame.to_csv(output_dir / f"{name}.csv", index=False, encoding="utf-8")


def load_ag_news_texts(path: str | Path | None = None) -> list[str]:
    """Load AG News text locally, or download the official Hugging Face dataset."""
    if path is not None:
        local_path = Path(path)
        if not local_path.is_file():
            raise FileNotFoundError(f"AG News dataset not found: {local_path}")
        frame = _read_table(local_path)
        if "text" in frame.columns:
            values = frame["text"]
        elif {"title", "description"}.issubset(frame.columns):
            values = frame["title"].fillna("") + " " + frame["description"].fillna("")
        else:
            raise ValueError("AG News must contain 'text', or both 'title' and 'description'.")
        return [str(value) for value in values.dropna() if str(value).strip()]

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "Install the 'datasets' package or provide --ag-news with a local file."
        ) from exc
    dataset = load_dataset("fancyzhx/ag_news")
    texts: list[str] = []
    for split in dataset.values():
        texts.extend(str(value) for value in split["text"] if str(value).strip())
    return texts

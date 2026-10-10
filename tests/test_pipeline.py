import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from nlp_assignment.bow import run_bow_experiments
from nlp_assignment.common import encode_labels
from nlp_assignment.cli import _validate_result_compatibility
from nlp_assignment.constants import (
    BERT_EPOCHS,
    BERT_MAX_LENGTH,
    BERT_MODEL_NAME,
    WORD_VECTOR_SIZE,
)
from nlp_assignment.data import split_nyt
from nlp_assignment.embeddings import (
    load_glove,
    run_word2vec_experiment,
    train_word2vec,
)


class SmallPipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rows = []
        for label, keyword in (("business", "market"), ("sports", "team"), ("science", "space")):
            for index in range(20):
                rows.append(
                    {"text": f"{keyword} {keyword} daily news report number {index}", "label": label}
                )
        cls.splits = split_nyt(pd.DataFrame(rows), seed=42)
        cls.encoder, cls.labels = encode_labels(cls.splits)

    def test_split_is_80_10_10_and_stratified(self):
        self.assertEqual({name: len(data) for name, data in self.splits.items()}, {
            "train": 48,
            "validation": 6,
            "test": 6,
        })
        for data in self.splits.values():
            self.assertEqual(set(data["label"]), {"business", "sports", "science"})

    def test_assignment_constraints_are_locked(self):
        self.assertEqual(WORD_VECTOR_SIZE, 100)
        self.assertEqual(BERT_MODEL_NAME, "google-bert/bert-base-uncased")
        self.assertEqual(BERT_MAX_LENGTH, 64)
        self.assertEqual(BERT_EPOCHS, 3)

    def test_bow_runs_end_to_end(self):
        records = run_bow_experiments(self.splits, self.labels, seed=42)
        self.assertEqual([record["method"] for record in records], [
            "binary_bow",
            "word_frequency",
        ])
        for record in records:
            self.assertGreaterEqual(record["metrics"]["accuracy"], 0.0)
            self.assertLessEqual(record["metrics"]["accuracy"], 1.0)

    def test_word2vec_runs_end_to_end(self):
        record = run_word2vec_experiment(
            "word2vec_test",
            self.splits["train"]["text"],
            self.splits,
            self.labels,
            seed=42,
            vector_size=100,
            epochs=2,
            min_count=1,
            workers=1,
        )
        self.assertEqual(record["details"]["vector_size"], 100)
        self.assertEqual(record["details"]["test_coverage"]["zero_vector_documents"], 0)

    def test_word2vec_is_reproducible(self):
        sentences = [["market", "stocks"], ["team", "match"], ["market", "report"]]
        first = train_word2vec(
            sentences, seed=42, vector_size=100, epochs=2, min_count=1, workers=1
        )
        second = train_word2vec(
            sentences, seed=42, vector_size=100, epochs=2, min_count=1, workers=1
        )
        np.testing.assert_array_equal(first.wv["market"], second.wv["market"])

    def test_glove_zip_loader_selects_needed_vectors(self):
        vector = " ".join(["0.1"] * 100)
        with tempfile.TemporaryDirectory() as directory:
            archive_path = Path(directory) / "glove.6B.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr(
                    "glove.6B.100d.txt",
                    f"market {vector}\nunused {vector}\n",
                )
            loaded = load_glove(archive_path, {"market"}, vector_size=100)
        self.assertEqual(set(loaded), {"market"})
        self.assertEqual(loaded["market"].shape, (100,))

    def test_different_experiment_cannot_mix_results(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            (output_dir / "run_config.json").write_text(
                json.dumps({"experiment_id": "old"}), encoding="utf-8"
            )
            (output_dir / "results.json").write_text("[]", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "different dataset, seed, or column"):
                _validate_result_compatibility(output_dir, {"experiment_id": "new"})


if __name__ == "__main__":
    unittest.main()

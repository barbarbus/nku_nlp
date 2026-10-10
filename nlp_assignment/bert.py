"""Task 3: BERT fine-tuning for NYT classification."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import numpy as np

from .common import classification_metrics
from .constants import BERT_EPOCHS, BERT_MAX_LENGTH, BERT_MODEL_NAME


class EncodedTextDataset:
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        item = {key: value[index] for key, value in self.encodings.items()}
        item["labels"] = int(self.labels[index])
        return item


def run_bert_experiment(
    splits,
    encoded_labels,
    label_encoder,
    output_dir: str | Path,
    seed: int,
    model_name: str = BERT_MODEL_NAME,
    max_length: int = BERT_MAX_LENGTH,
    epochs: int = BERT_EPOCHS,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
) -> dict[str, Any]:
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )

    output_dir = Path(output_dir)
    checkpoint_dir = output_dir / "bert_checkpoints"
    final_dir = output_dir / "bert_model"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    def encode(frame):
        return tokenizer(
            frame["text"].tolist(),
            truncation=True,
            max_length=max_length,
            padding=False,
            return_tensors=None,
        )

    train_dataset = EncodedTextDataset(encode(splits["train"]), encoded_labels["train"])
    validation_dataset = EncodedTextDataset(
        encode(splits["validation"]), encoded_labels["validation"]
    )
    test_dataset = EncodedTextDataset(encode(splits["test"]), encoded_labels["test"])

    id2label = {index: str(label) for index, label in enumerate(label_encoder.classes_)}
    label2id = {label: index for index, label in id2label.items()}
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id,
    )

    def compute_metrics(eval_prediction):
        predictions = np.argmax(eval_prediction.predictions, axis=-1)
        return classification_metrics(eval_prediction.label_ids, predictions)

    kwargs = {
        "output_dir": str(checkpoint_dir),
        "num_train_epochs": epochs,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": 0.01,
        "save_strategy": "epoch",
        "logging_strategy": "steps",
        "logging_steps": 50,
        "load_best_model_at_end": True,
        "metric_for_best_model": "macro_f1",
        "greater_is_better": True,
        "save_total_limit": 2,
        "seed": seed,
        "data_seed": seed,
        "report_to": [],
    }
    parameters = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" in parameters:
        kwargs["eval_strategy"] = "epoch"
    else:
        kwargs["evaluation_strategy"] = "epoch"
    trainer_kwargs = dict(
        model=model,
        args=TrainingArguments(**kwargs),
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )
    trainer_parameters = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in trainer_parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = Trainer(**trainer_kwargs)

    trainer.train()
    prediction = trainer.predict(test_dataset)
    predicted_labels = np.argmax(prediction.predictions, axis=-1)
    metrics = classification_metrics(encoded_labels["test"], predicted_labels)
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    return {
        "method": "bert_base_uncased",
        "metrics": metrics,
        "details": {
            "model": model_name,
            "max_length": max_length,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
        },
    }

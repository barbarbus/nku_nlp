"""Regenerate the two vector bar charts embedded in the LaTeX report.

The first chart reads the original experiment results.  The second refits the
two bag-of-words models using the saved split and the same project code.  It
does not overwrite any experiment results or model artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path

from reportlab.lib.colors import Color
from reportlab.pdfgen import canvas

from report_analysis import fit_bow, load_splits


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "outputs" / "results.json"
FIGURES = ROOT / "latex 模板" / "figures"

NAVY = Color(0.16, 0.35, 0.54)
ORANGE = Color(0.81, 0.43, 0.19)
GRID = Color(0.83, 0.86, 0.89)
INK = Color(0.14, 0.17, 0.20)


def text(pdf: canvas.Canvas, x: float, y: float, value: str, size: int = 9, bold: bool = False) -> None:
    pdf.setFillColor(INK)
    pdf.setFont("Helvetica-Bold" if bold else "Helvetica", size)
    pdf.drawString(x, y, value)


def legend(pdf: canvas.Canvas, x: float, y: float, first: str, second: str) -> None:
    for offset, color, label in ((0, NAVY, first), (112, ORANGE, second)):
        pdf.setFillColor(color)
        pdf.rect(x + offset, y - 2, 8, 8, stroke=0, fill=1)
        text(pdf, x + offset + 13, y - 1, label, 8)


def grid(pdf: canvas.Canvas, x0: float, x1: float, y0: float, y1: float, minimum: float, maximum: float, step: float) -> None:
    count = round((maximum - minimum) / step)
    pdf.setStrokeColor(GRID)
    pdf.setLineWidth(0.45)
    for index in range(count + 1):
        value = minimum + index * step
        x = x0 + (value - minimum) / (maximum - minimum) * (x1 - x0)
        pdf.line(x, y0, x, y1)
        pdf.setFont("Helvetica", 7.5)
        pdf.setFillColor(INK)
        pdf.drawCentredString(x, y0 - 14, f"{value:.2f}")


def bar(pdf: canvas.Canvas, x0: float, x1: float, y: float, value: float, minimum: float, maximum: float, color: Color) -> None:
    width = max(0, (value - minimum) / (maximum - minimum) * (x1 - x0))
    pdf.setFillColor(color)
    pdf.rect(x0, y, width, 9, stroke=0, fill=1)


def overall_chart(path: Path, records: list[dict]) -> None:
    labels = {
        "binary_bow": "Binary BoW",
        "word_frequency": "Word Frequency",
        "pretrained_glove": "GloVe 100d",
        "word2vec_ag_news": "AG News Word2Vec",
        "word2vec_nyt_train": "NYT Word2Vec",
        "bert_base_uncased": "BERT-base-uncased",
    }
    pdf = canvas.Canvas(str(path), pagesize=(500, 315))
    text(pdf, 8, 293, "NYT test set: six-method comparison (n = 1,152)", 11, True)
    legend(pdf, 160, 278, "Accuracy", "Macro-F1")
    x0, x1 = 160, 405
    grid(pdf, x0, x1, 47, 264, 0.90, 1.00, 0.02)
    for index, record in enumerate(records):
        y = 248 - index * 36
        metric = record["metrics"]
        text(pdf, 8, y + 1, labels[record["method"]], 8.5, record["method"] == "word_frequency")
        bar(pdf, x0, x1, y + 2, metric["accuracy"], 0.90, 1.00, NAVY)
        bar(pdf, x0, x1, y - 10, metric["macro_f1"], 0.90, 1.00, ORANGE)
        text(pdf, 414, y + 3, f'{metric["accuracy"]:.4f}', 8)
        text(pdf, 414, y - 9, f'{metric["macro_f1"]:.4f}', 8)
    text(pdf, 205, 18, "Metric value (zoomed axis: 0.90-1.00)", 8)
    pdf.showPage()
    pdf.save()


def class_chart(path: Path, binary: dict, frequency: dict) -> None:
    pdf = canvas.Canvas(str(path), pagesize=(500, 240))
    text(pdf, 8, 218, "Per-class F1: binary vs. frequency BoW", 11, True)
    legend(pdf, 160, 202, "Binary BoW", "Word Frequency")
    x0, x1 = 160, 405
    grid(pdf, x0, x1, 53, 190, 0.94, 1.00, 0.01)
    for index, label in enumerate(("business", "politics", "sports")):
        y = 169 - index * 43
        first = binary["per_class"][label]["f1-score"]
        second = frequency["per_class"][label]["f1-score"]
        text(pdf, 8, y + 1, label, 9)
        bar(pdf, x0, x1, y + 2, first, 0.94, 1.00, NAVY)
        bar(pdf, x0, x1, y - 10, second, 0.94, 1.00, ORANGE)
        text(pdf, 414, y + 3, f"{first:.4f}", 8)
        text(pdf, 414, y - 9, f"{second:.4f}", 8)
    text(pdf, 208, 23, "Per-class F1 (zoomed axis: 0.94-1.00)", 8)
    pdf.showPage()
    pdf.save()


def class_heatmap(path: Path, diagnostics: dict) -> None:
    """Show every method's test F1 for each NYT class on one scale."""
    labels = {
        "binary_bow": "Binary BoW",
        "word_frequency": "Word Frequency",
        "pretrained_glove": "GloVe 100d",
        "word2vec_ag_news": "AG News Word2Vec",
        "word2vec_nyt_train": "NYT Word2Vec",
        "bert_base_uncased": "BERT-base-uncased",
    }
    pdf = canvas.Canvas(str(path), pagesize=(500, 280))
    text(pdf, 8, 260, "Per-class F1 across all six methods (NYT test set)", 11, True)
    x0, cell_width = 168, 91
    for column, label in enumerate(("business", "politics", "sports")):
        text(pdf, x0 + column * cell_width + 17, 234, label, 8.5, True)
    for row, (name, short) in enumerate(labels.items()):
        y = 208 - row * 30
        text(pdf, 8, y + 7, short, 8.5, name == "word_frequency")
        for column, label in enumerate(("business", "politics", "sports")):
            value = diagnostics["methods"][name]["per_class"][label]["f1"]
            shade = min(1.0, max(0.0, (value - 0.90) / 0.10))
            fill = Color(0.90 - 0.74 * shade, 0.94 - 0.59 * shade, 0.97 - 0.43 * shade)
            pdf.setFillColor(fill)
            pdf.rect(x0 + column * cell_width, y, cell_width - 4, 24, stroke=0, fill=1)
            pdf.setFillColor(Color(1, 1, 1) if shade > 0.55 else INK)
            pdf.setFont("Helvetica-Bold", 9)
            pdf.drawCentredString(x0 + column * cell_width + (cell_width - 4) / 2, y + 8, f"{value:.4f}")
    text(pdf, 168, 13, "Darker cells indicate higher F1; numeric values are authoritative.", 7.5)
    pdf.showPage()
    pdf.save()


def error_type_chart(path: Path, diagnostics: dict) -> None:
    """Split test errors into minority-class confusion and sports-related errors."""
    labels = {
        "binary_bow": "Binary BoW",
        "word_frequency": "Word Frequency",
        "pretrained_glove": "GloVe 100d",
        "word2vec_ag_news": "AG News Word2Vec",
        "word2vec_nyt_train": "NYT Word2Vec",
        "bert_base_uncased": "BERT-base-uncased",
    }
    pdf = canvas.Canvas(str(path), pagesize=(500, 305))
    text(pdf, 8, 284, "Test errors by confusion type (n = 1,152)", 11, True)
    legend(pdf, 160, 267, "business vs. politics", "sports involved")
    x0, x1 = 160, 418
    pdf.setStrokeColor(GRID)
    pdf.setLineWidth(0.45)
    for value in range(0, 31, 5):
        x = x0 + value / 30 * (x1 - x0)
        pdf.line(x, 42, x, 254)
        pdf.setFillColor(INK)
        pdf.setFont("Helvetica", 7.5)
        pdf.drawCentredString(x, 27, str(value))
    for index, (name, short) in enumerate(labels.items()):
        y = 236 - index * 35
        summary = diagnostics["methods"][name]
        matrix = summary["confusion_matrix"]
        minority_confusion = matrix[0][1] + matrix[1][0]
        errors = diagnostics["test_size"] - summary["correct"]
        sports_related = errors - minority_confusion
        scale = (x1 - x0) / 30
        text(pdf, 8, y + 3, short, 8.5, name == "word_frequency")
        pdf.setFillColor(NAVY)
        pdf.rect(x0, y, minority_confusion * scale, 13, stroke=0, fill=1)
        pdf.setFillColor(ORANGE)
        pdf.rect(x0 + minority_confusion * scale, y, sports_related * scale, 13, stroke=0, fill=1)
        text(pdf, 427, y + 3, f"{minority_confusion} + {sports_related}", 8)
    text(pdf, 246, 10, "Number of misclassified documents", 8)
    pdf.showPage()
    pdf.save()


def main() -> None:
    records = json.loads(RESULTS.read_text(encoding="utf-8"))
    if len(records) != 6:
        raise ValueError("The six original experiment results are required.")
    splits = load_splits()
    binary = fit_bow(splits["train"], splits["test"], binary=True)
    frequency = fit_bow(splits["train"], splits["test"], binary=False)
    metrics = {record["method"]: record["metrics"] for record in records}
    for name, fresh in (("binary_bow", binary), ("word_frequency", frequency)):
        for metric in ("accuracy", "macro_f1"):
            if abs(fresh[metric] - metrics[name][metric]) > 1e-9:
                raise ValueError(f"Refitted {name} does not reproduce {metric}.")
    FIGURES.mkdir(parents=True, exist_ok=True)
    overall_chart(FIGURES / "overall_metrics.pdf", records)
    class_chart(FIGURES / "bow_class_f1.pdf", binary, frequency)
    diagnostics_path = ROOT / "outputs" / "report_diagnostics.json"
    if diagnostics_path.is_file():
        diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
        for record in records:
            actual = diagnostics["methods"][record["method"]]
            for metric in ("accuracy", "macro_f1"):
                if abs(actual[metric] - record["metrics"][metric]) > 1e-8:
                    raise ValueError("Prediction-level diagnostics do not match the original results.")
        class_heatmap(FIGURES / "all_class_f1.pdf", diagnostics)
        error_type_chart(FIGURES / "error_types.pdf", diagnostics)
        print("Generated four verified vector figures in", FIGURES)
    else:
        print("Generated two vector figures; run analyze_all_methods.py for two more.")


if __name__ == "__main__":
    main()

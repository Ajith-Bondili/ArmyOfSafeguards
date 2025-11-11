"""Toxicity/racism safeguard critic powered by DeBERTa-v3."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, Mapping

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Model configuration - loads from Hugging Face
MODEL_ID = "SohamNagi/tiny-toxicity-classifier"

# Load model from Hugging Face
_tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False)
_model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
_model.eval()


def predict(text: str) -> Dict[str, float]:
    """Run the toxicity critic on *text* and return the label + confidence.

    Args:
        text: Input text to evaluate for toxicity/racism

    Returns:
        Dictionary with 'label' (str) and 'confidence' (float)
        - LABEL_0: Non-toxic/safe content
        - LABEL_1: Toxic/racist content
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty string")

    inputs = _tokenizer(text, return_tensors="pt",
                        truncation=True, max_length=512)

    with torch.no_grad():
        outputs = _model(**inputs)

    logits = outputs.logits[0]
    probabilities = torch.softmax(logits, dim=-1)
    confidence, label_id = torch.max(probabilities, dim=-1)

    label = _model.config.id2label.get(label_id.item(), str(label_id.item()))
    return {"label": label, "confidence": float(confidence.item())}


def aggregate(predictions: Iterable[Mapping[str, float]]) -> Dict[str, float]:
    """Majority-vote aggregation across toxicity critics.

    Args:
        predictions: Iterable of prediction dictionaries from predict()

    Returns:
        Aggregated result with winning label, confidence, votes, and total
    """
    votes = Counter()
    confidence_totals = defaultdict(float)

    for prediction in predictions:
        label = prediction.get("label")
        if label is None:
            continue
        confidence = float(prediction.get("confidence", 0.0))
        votes[label] += 1
        confidence_totals[label] += confidence

    if not votes:
        raise ValueError(
            "predictions must be a non-empty iterable of mappings with labels")

    winning_label = min(
        votes.keys(),
        key=lambda lbl: (-votes[lbl], -confidence_totals[lbl], str(lbl)),
    )

    winning_votes = votes[winning_label]
    total_predictions = sum(votes.values())
    confidence = confidence_totals[winning_label] / max(1, winning_votes)

    return {
        "label": winning_label,
        "confidence": confidence,
        "votes": winning_votes,
        "total": total_predictions,
    }


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the toxicity safeguard critic")
    parser.add_argument("text", nargs="?",
                        help="Text snippet to evaluate for toxicity/racism")
    return parser


if __name__ == "__main__":
    cli = _build_cli()
    args = cli.parse_args()

    sample_text = args.text or input("Enter text to evaluate: ")
    result = predict(sample_text)

    print("Prediction:")
    print(f"  Label: {result['label']}")
    print(f"  Confidence: {result['confidence']:.4f}")

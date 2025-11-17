## Load huggingface model and run inference
import torch
from transformers import (
    AutoTokenizer, AutoConfig,
    AutoModelForSequenceClassification, PreTrainedModel
)
from huggingface_hub import login, hf_hub_download
from safetensors.torch import load_file
from torch import nn
from collections import Counter, defaultdict
from typing import Dict, Iterable, Mapping
import argparse

repo_id = "tommypang04/finetuned-model-jailbrak"
tok = AutoTokenizer.from_pretrained(repo_id)
model = AutoModelForSequenceClassification.from_pretrained(repo_id)
model.eval()

def predict(text: str):
    enc = tok(text, return_tensors="pt", truncation=True, max_length=384)

    with torch.no_grad():
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1).squeeze()

        pred_id = torch.argmax(probs).item()
        confidence = probs[pred_id].item()
        
        return {
            "label": bool(pred_id), # idx 0 is False, 1 is True, labels are [not_jailbreak, jailbreak]
            "confidence": confidence,
        }

def aggregate(predictions: Iterable[Mapping[str, float]]) -> Dict[str, float]:
    """Majority-vote aggregation across factuality critics."""
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
        raise ValueError("predictions must be a non-empty iterable of mappings with labels")

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
    parser = argparse.ArgumentParser(description="Run the factuality safeguard critic")
    parser.add_argument("text", nargs="?", help="Text snippet to evaluate for factuality")
    return parser


if __name__ == "__main__":
    cli = _build_cli()
    args = cli.parse_args()

    sample_text = args.text or input("Enter text to evaluate: ")
    result = predict(sample_text)

    print("Prediction:")
    print(f"  Label: {result['label']}")
    print(f"  Confidence: {result['confidence']:.4f}")
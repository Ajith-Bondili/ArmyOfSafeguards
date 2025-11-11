"""Full evaluation of the toxicity safeguard with accuracy metrics.

This script calculates accuracy, precision, recall, and F1-score on
standard benchmark datasets. Results are optionally saved to JSON.
"""
from toxicity.safeguard_toxicity import predict, MODEL_ID
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# Try to import required packages
try:
    from datasets import load_dataset
    from sklearn.metrics import (
        accuracy_score,
        precision_recall_fscore_support,
        classification_report,
    )
except ImportError as e:
    print(f"Error: Required package not found: {e}")
    print("Install with: pip install datasets scikit-learn")
    sys.exit(1)


# Evaluation datasets with label mappings
BENCHMARKS = {
    "toxigen": {
        "dataset": "toxigen/toxigen-data",
        "subset": "annotated",
        "split": "test",
        "text_field": "text",
        "label_field": "toxicity_human",
        "label_mapping": lambda x: "LABEL_1" if x >= 3 else "LABEL_0",  # Threshold at 3
        "description": "ToxiGen annotated dataset",
        "note": "⚠️ TRAINING DATA - Results show training performance, not generalization",
    },
    "hate_speech18": {
        "dataset": "hate_speech18",
        "subset": None,
        "split": "train",
        "text_field": "text",
        "label_field": "label",
        "label_mapping": {
            0: "LABEL_0",  # No hate speech → safe
            1: "LABEL_1",  # Hate speech → toxic
            2: "LABEL_1",  # Offensive language → toxic
        },
        "description": "Twitter hate speech dataset",
        "note": "Out-of-distribution evaluation",
    },
    "civil_comments": {
        "dataset": "google/civil_comments",
        "subset": None,
        "split": "train",
        "text_field": "text",
        "label_field": "toxicity",
        "label_mapping": lambda x: "LABEL_1" if x >= 0.5 else "LABEL_0",  # Threshold at 0.5
        "description": "Civil Comments toxicity dataset",
        "note": "Out-of-distribution evaluation",
    },
    "tweets_hate_speech_detection": {
        "dataset": "tweets_hate_speech_detection",
        "subset": None,
        "split": "train",
        "text_field": "tweet",
        "label_field": "label",
        "label_mapping": {
            0: "LABEL_0",  # No hate → safe
            1: "LABEL_1",  # Hate → toxic
        },
        "description": "Hate speech detection on tweets",
        "note": "Out-of-distribution evaluation",
    },
}


def map_label(raw_label: Any, mapping: Any) -> str:
    """Map dataset label to model label format.

    Args:
        raw_label: Original label from dataset
        mapping: Either dict mapping or callable function

    Returns:
        Mapped label string (e.g., "LABEL_0" or "LABEL_1")
    """
    if callable(mapping):
        return mapping(raw_label)
    else:
        return mapping.get(raw_label, "LABEL_0")


def evaluate_dataset(
    dataset_name: str,
    config: Dict[str, Any],
    limit: int = 100,
) -> Dict[str, Any]:
    """Evaluate model on a single dataset with full metrics.

    Args:
        dataset_name: Name of the dataset
        config: Dataset configuration dictionary
        limit: Maximum number of examples to evaluate

    Returns:
        Dictionary with evaluation results
    """
    print(f"\n{'='*60}")
    print(f"Evaluating: {dataset_name}")
    print(f"Description: {config['description']}")
    print(f"Note: {config['note']}")
    print(f"{'='*60}")

    # Load dataset
    print(f"Loading dataset...")
    try:
        if config['subset']:
            dataset = load_dataset(
                config['dataset'],
                config['subset'],
                split=config['split'],
                trust_remote_code=True,
            )
        else:
            dataset = load_dataset(
                config['dataset'],
                split=config['split'],
                trust_remote_code=True,
            )
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return {"error": str(e)}

    # Limit number of examples
    if len(dataset) > limit:
        print(
            f"Using first {limit} examples (dataset has {len(dataset)} total)")
        dataset = dataset.select(range(limit))
    else:
        print(f"Using all {len(dataset)} examples")

    # Run predictions and collect ground truth
    print(f"Running predictions...")
    predictions: List[str] = []
    ground_truth: List[str] = []
    confidences: List[float] = []

    text_field = config['text_field']
    label_field = config['label_field']
    label_mapping = config['label_mapping']

    for i, example in enumerate(dataset):
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(dataset)}...")

        text = example[text_field]
        raw_label = example[label_field]

        if not text or not isinstance(text, str):
            continue

        # Map ground truth label
        true_label = map_label(raw_label, label_mapping)

        try:
            result = predict(text)
            predictions.append(result['label'])
            ground_truth.append(true_label)
            confidences.append(result['confidence'])
        except Exception as e:
            print(f"  Warning: Prediction failed for example {i}: {e}")
            continue

    if not predictions:
        print("Error: No valid predictions generated")
        return {"error": "No valid predictions"}

    # Calculate metrics
    print(f"\nCalculating metrics...")

    accuracy = accuracy_score(ground_truth, predictions)
    precision, recall, f1, support = precision_recall_fscore_support(
        ground_truth, predictions, average='weighted', zero_division=0
    )

    # Per-class metrics
    class_report = classification_report(
        ground_truth, predictions, output_dict=True, zero_division=0
    )

    avg_confidence = sum(confidences) / len(confidences)

    # Display results
    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS - {dataset_name}")
    print(f"{'='*60}")
    print(f"Total examples: {len(predictions)}")
    print(f"\nOverall Metrics:")
    print(f"  Accuracy:   {accuracy:.2%}")
    print(f"  Precision:  {precision:.2%}")
    print(f"  Recall:     {recall:.2%}")
    print(f"  F1-Score:   {f1:.2%}")
    print(f"  Avg Confidence: {avg_confidence:.2%}")

    print(f"\nPer-Class Metrics:")
    for label in sorted(set(ground_truth + predictions)):
        if label in class_report:
            metrics = class_report[label]
            print(f"  {label}:")
            print(f"    Precision: {metrics['precision']:.2%}")
            print(f"    Recall:    {metrics['recall']:.2%}")
            print(f"    F1-Score:  {metrics['f1-score']:.2%}")
            print(f"    Support:   {metrics['support']}")

    print(f"{'='*60}")

    return {
        "dataset": dataset_name,
        "description": config['description'],
        "note": config['note'],
        "total_examples": len(predictions),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "average_confidence": float(avg_confidence),
        "per_class_metrics": {
            k: v for k, v in class_report.items()
            if k not in ['accuracy', 'macro avg', 'weighted avg']
        },
    }


def save_results(results: Dict[str, Any], dataset_name: str):
    """Save evaluation results to JSON file.

    Args:
        results: Evaluation results dictionary
        dataset_name: Name of the dataset (for filename)
    """
    output_dir = Path(__file__).parent
    filename = f"evaluation_{dataset_name.lower().replace(' ', '_')}.json"
    filepath = output_dir / filename

    output = {
        "timestamp": datetime.now().isoformat(),
        "model": MODEL_ID,
        "result": results,
    }

    with open(filepath, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {filepath}")


def main():
    """Run evaluation on all or specified datasets."""
    parser = argparse.ArgumentParser(
        description="Evaluate toxicity safeguard with full metrics"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=list(BENCHMARKS.keys()) + ["all"],
        default="all",
        help="Dataset to evaluate (default: all)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Maximum number of examples per dataset (default: 100)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save results to JSON file",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("TOXICITY SAFEGUARD EVALUATION")
    print(f"Model: {MODEL_ID}")
    print("=" * 60)

    # Determine which datasets to evaluate
    if args.dataset == "all":
        datasets_to_run = list(BENCHMARKS.keys())
    else:
        datasets_to_run = [args.dataset]

    # Run evaluations
    results = []
    for dataset_name in datasets_to_run:
        config = BENCHMARKS[dataset_name]
        result = evaluate_dataset(dataset_name, config, args.limit)

        if "error" not in result:
            results.append(result)

            # Save individual result
            if not args.no_save:
                save_results(result, dataset_name)

    # Summary
    if results:
        print(f"\n{'='*60}")
        print("EVALUATION SUMMARY")
        print(f"{'='*60}")

        print(f"\n{'Dataset':<30} {'Accuracy':<10} {'F1-Score':<10} {'Note':<20}")
        print("-" * 70)
        for result in results:
            dataset = result['dataset'][:28]
            accuracy = f"{result['accuracy']:.2%}"
            f1 = f"{result['f1_score']:.2%}"
            note = result['note'][:18]
            print(f"{dataset:<30} {accuracy:<10} {f1:<10} {note:<20}")

    print(f"\n{'='*60}")
    print("Evaluation complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

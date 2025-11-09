"""
Evaluation script for factuality safeguard.

This script evaluates the DeBERTa-v3 factuality model on benchmark datasets
and calculates accuracy, precision, recall, and F1-score.

Phase 2: Full evaluation with metrics
"""
import sys
from pathlib import Path

# Add parent directory to path so we can import the safeguard
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import load_dataset
from safeguard_factuality import predict
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import json
from datetime import datetime


# Benchmark datasets with label mappings
# NOTE: TruthfulQA and FEVER were used in training - use for sanity check only
# VitaminC and Climate-FEVER are OUT-OF-DISTRIBUTION datasets for true generalization testing

BENCHMARKS = {
    # === TRAINING DATASETS (for sanity check) ===
    "TruthfulQA": {
        "hf_id": "truthful_qa",
        "config": "generation",
        "split": "validation",
        "text_field": "question",
        "label_field": "category",
        "label_mapping": {
            "Correct Answers": "LABEL_0",  # Factual
            "Incorrect Answers": "LABEL_1",  # Non-factual
        },
        "default_label": "LABEL_1",
        "note": "⚠️  Used in training - sanity check only"
    },
    "FEVER": {
        "hf_id": "fever",
        "config": "v1.0",
        "split": "paper_test",
        "text_field": "claim",
        "label_field": "label",
        "label_mapping": {
            "SUPPORTS": "LABEL_0",  # Factual
            "REFUTES": "LABEL_1",   # Non-factual
            "NOT ENOUGH INFO": "LABEL_1"
        },
        "default_label": "LABEL_1",
        "note": "⚠️  Used in training - sanity check only"
    },
    
    # === OUT-OF-DISTRIBUTION DATASETS (true generalization test) ===
    "VitaminC": {
        "hf_id": "tals/vitaminc",
        "config": None,
        "split": "test",
        "text_field": "claim",
        "label_field": "label",
        "label_mapping": {
            "SUPPORTS": "LABEL_0",
            "REFUTES": "LABEL_1",
            "NOT ENOUGH INFO": "LABEL_1"
        },
        "default_label": "LABEL_1",
        "note": "✅ Out-of-distribution - true generalization test"
    },
    "Climate-FEVER": {
        "hf_id": "climate_fever",
        "config": None,
        "split": "test",
        "text_field": "claim",
        "label_field": "claim_label",
        "label_mapping": {
            "SUPPORTS": "LABEL_0",
            "REFUTES": "LABEL_1",
            "NOT ENOUGH INFO": "LABEL_1",
            "DISPUTED": "LABEL_1"
        },
        "default_label": "LABEL_1",
        "note": "✅ Out-of-distribution - climate-specific claims"
    },
    "LIAR": {
        "hf_id": "liar",
        "config": None,
        "split": "test",
        "text_field": "statement",
        "label_field": "label",
        "label_mapping": {
            0: "LABEL_1",  # pants-fire (false)
            1: "LABEL_1",  # false
            2: "LABEL_1",  # barely-true
            3: "LABEL_1",  # half-true
            4: "LABEL_0",  # mostly-true
            5: "LABEL_0",  # true
        },
        "default_label": "LABEL_1",
        "note": "✅ Out-of-distribution - political fact-checking"
    }
}


def map_label(original_label, label_mapping, default_label):
    """
    Map dataset-specific label to our binary labels (LABEL_0/LABEL_1).
    
    Args:
        original_label: Original label from dataset
        label_mapping: Dictionary mapping original labels to LABEL_0/LABEL_1
        default_label: Default label if mapping not found
        
    Returns:
        Mapped label (LABEL_0 or LABEL_1)
    """
    if isinstance(original_label, str):
        return label_mapping.get(original_label, default_label)
    return default_label


def evaluate_dataset(name: str, config: dict, limit: int = 100, verbose: bool = True):
    """
    Evaluate the factuality safeguard on a dataset with ground truth labels.
    
    Args:
        name: Name of the benchmark dataset
        config: Configuration dictionary with dataset details
        limit: Maximum number of examples to evaluate
        verbose: Whether to print progress
        
    Returns:
        Dictionary with evaluation metrics
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"Evaluating: {name}")
        if "note" in config:
            print(f"{config['note']}")
        print(f"{'='*70}")
    
    try:
        # Load dataset
        hf_id = config["hf_id"]
        ds_config = config.get("config")
        split = config.get("split", "test")
        
        if ds_config:
            ds = load_dataset(hf_id, ds_config, split=split, trust_remote_code=True)
        else:
            ds = load_dataset(hf_id, split=split, trust_remote_code=True)
        
        # Sample subset
        subset_size = min(limit, len(ds))
        subset = ds.select(range(subset_size))
        
        if verbose:
            print(f"Loaded {subset_size} examples from {name}")
            print(f"Running predictions...\n")
        
        # Run predictions and collect ground truth
        predictions = []
        ground_truth = []
        confidences = []
        
        label_mapping = config.get("label_mapping", {})
        default_label = config.get("default_label", "LABEL_1")
        
        for ex in tqdm(subset, disable=not verbose):
            # Extract text
            text_field = config.get("text_field", "claim")
            text = ex.get(text_field, str(ex))
            
            # Get prediction
            result = predict(text)
            predictions.append(result["label"])
            confidences.append(result["confidence"])
            
            # Extract and map ground truth
            label_field = config.get("label_field")
            if label_field and label_field in ex:
                original_label = ex[label_field]
                mapped_label = map_label(original_label, label_mapping, default_label)
                ground_truth.append(mapped_label)
            else:
                # Skip examples without labels
                predictions.pop()
                confidences.pop()
        
        if len(predictions) == 0 or len(ground_truth) == 0:
            return {
                "dataset": name,
                "error": "No valid examples with ground truth labels"
            }
        
        # Calculate metrics
        accuracy = accuracy_score(ground_truth, predictions)
        precision, recall, f1, support = precision_recall_fscore_support(
            ground_truth, predictions, average='binary', pos_label='LABEL_0'
        )
        
        # Calculate per-class metrics
        precision_per_class, recall_per_class, f1_per_class, support_per_class = \
            precision_recall_fscore_support(ground_truth, predictions, labels=['LABEL_0', 'LABEL_1'])
        
        # Count predictions
        pred_factual = predictions.count("LABEL_0")
        pred_nonfactual = predictions.count("LABEL_1")
        true_factual = ground_truth.count("LABEL_0")
        true_nonfactual = ground_truth.count("LABEL_1")
        
        results = {
            "dataset": name,
            "total_examples": len(predictions),
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "average_confidence": sum(confidences) / len(confidences),
            "predictions": {
                "factual": pred_factual,
                "non_factual": pred_nonfactual
            },
            "ground_truth": {
                "factual": true_factual,
                "non_factual": true_nonfactual
            },
            "per_class_metrics": {
                "LABEL_0_factual": {
                    "precision": float(precision_per_class[0]),
                    "recall": float(recall_per_class[0]),
                    "f1_score": float(f1_per_class[0]),
                    "support": int(support_per_class[0])
                },
                "LABEL_1_nonfactual": {
                    "precision": float(precision_per_class[1]),
                    "recall": float(recall_per_class[1]),
                    "f1_score": float(f1_per_class[1]),
                    "support": int(support_per_class[1])
                }
            }
        }
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"Results for {name}:")
            print(f"{'='*70}")
            print(f"  Total examples: {results['total_examples']}")
            print(f"  Accuracy: {accuracy:.2%}")
            print(f"  Precision: {precision:.2%}")
            print(f"  Recall: {recall:.2%}")
            print(f"  F1-Score: {f1:.2%}")
            print(f"  Average Confidence: {results['average_confidence']:.2%}")
            print(f"\n  Ground Truth Distribution:")
            print(f"    Factual: {true_factual} ({true_factual/len(predictions):.1%})")
            print(f"    Non-factual: {true_nonfactual} ({true_nonfactual/len(predictions):.1%})")
            print(f"\n  Prediction Distribution:")
            print(f"    Factual: {pred_factual} ({pred_factual/len(predictions):.1%})")
            print(f"    Non-factual: {pred_nonfactual} ({pred_nonfactual/len(predictions):.1%}")
        
        return results
        
    except Exception as e:
        if verbose:
            print(f"⚠️  Error evaluating {name}: {e}")
        return {"dataset": name, "error": str(e)}


def run_all_evaluations(limit: int = 100, save_results: bool = True):
    """
    Run evaluation on all benchmark datasets.
    
    Args:
        limit: Maximum number of examples per dataset
        save_results: Whether to save results to JSON file
        
    Returns:
        List of results dictionaries
    """
    print("="*70)
    print("FACTUALITY SAFEGUARD EVALUATION")
    print("="*70)
    print(f"\nEvaluating on {len(BENCHMARKS)} datasets (limit: {limit} examples each)")
    print(f"Model: ajith-bondili/deberta-v3-factuality-small")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    all_results = []
    
    for name, config in BENCHMARKS.items():
        result = evaluate_dataset(name, config, limit=limit, verbose=True)
        all_results.append(result)
    
    # Calculate overall statistics
    successful = [r for r in all_results if "error" not in r]
    failed = [r for r in all_results if "error" in r]
    
    print("\n" + "="*70)
    print("OVERALL SUMMARY")
    print("="*70)
    
    print(f"\nSuccessful evaluations: {len(successful)}/{len(BENCHMARKS)}")
    
    if successful:
        avg_accuracy = sum(r['accuracy'] for r in successful) / len(successful)
        avg_f1 = sum(r['f1_score'] for r in successful) / len(successful)
        avg_precision = sum(r['precision'] for r in successful) / len(successful)
        avg_recall = sum(r['recall'] for r in successful) / len(successful)
        
        print(f"\nAverage Metrics Across Datasets:")
        print(f"  Accuracy:  {avg_accuracy:.2%}")
        print(f"  Precision: {avg_precision:.2%}")
        print(f"  Recall:    {avg_recall:.2%}")
        print(f"  F1-Score:  {avg_f1:.2%}")
        
        print(f"\nPer-Dataset Results:")
        print(f"{'Dataset':<20} {'Accuracy':<12} {'F1-Score':<12} {'Precision':<12} {'Recall':<12}")
        print("-"*70)
        for result in successful:
            print(f"{result['dataset']:<20} "
                  f"{result['accuracy']:<12.2%} "
                  f"{result['f1_score']:<12.2%} "
                  f"{result['precision']:<12.2%} "
                  f"{result['recall']:<12.2%}")
    
    if failed:
        print(f"\nFailed evaluations: {len(failed)}")
        for result in failed:
            print(f"  {result['dataset']}: {result['error']}")
    
    # Save results
    if save_results and successful:
        output_file = Path(__file__).parent / "evaluation_results.json"
        with open(output_file, 'w') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "model": "ajith-bondili/deberta-v3-factuality-small",
                "limit_per_dataset": limit,
                "results": all_results,
                "summary": {
                    "avg_accuracy": avg_accuracy,
                    "avg_precision": avg_precision,
                    "avg_recall": avg_recall,
                    "avg_f1_score": avg_f1
                } if successful else {}
            }, f, indent=2)
        print(f"\n✅ Results saved to: {output_file}")
    
    return all_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Evaluate factuality safeguard with accuracy metrics"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Maximum examples per dataset (default: 100)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=list(BENCHMARKS.keys()),
        help="Evaluate single dataset only"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save results to JSON file"
    )
    
    args = parser.parse_args()
    
    if args.dataset:
        # Run single dataset
        config = BENCHMARKS[args.dataset]
        result = evaluate_dataset(args.dataset, config, limit=args.limit)
        
        if not args.no_save and "error" not in result:
            output_file = Path(__file__).parent / f"evaluation_{args.dataset.lower()}.json"
            with open(output_file, 'w') as f:
                json.dump({
                    "timestamp": datetime.now().isoformat(),
                    "model": "ajith-bondili/deberta-v3-factuality-small",
                    "result": result
                }, f, indent=2)
            print(f"\n✅ Results saved to: {output_file}")
    else:
        # Run all datasets
        run_all_evaluations(limit=args.limit, save_results=not args.no_save)


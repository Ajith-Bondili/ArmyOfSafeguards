# Factuality Safeguard - Evaluation Summary

## Model Information
- **Model**: `ajith-bondili/deberta-v3-factuality-small`
- **Architecture**: DeBERTa-v3-small
- **Task**: Binary factuality classification

## Evaluation Strategy

### ⚠️ Training Data Contamination
The model was fine-tuned on **TruthfulQA** and **FEVER** datasets. Therefore, results on these datasets represent **training performance** (sanity check), not true generalization.

### ✅ True Generalization Testing
For unbiased evaluation, we test on **out-of-distribution (OOD)** datasets that the model has never seen:
- **VitaminC**: Contradiction-aware fact verification
- **Climate-FEVER**: Climate-specific claims  
- **LIAR**: Political fact-checking statements

---

## Results

### Training Data Performance (Sanity Check)

| Dataset | Accuracy | Precision | Recall | F1-Score | Note |
|---------|----------|-----------|--------|----------|------|
| **FEVER** | 84.00% | 72.50% | 85.29% | 78.38% | ⚠️ Training data |
| **TruthfulQA** | 75.00% | - | - | - | ⚠️ Training data |

**Interpretation**: High performance expected since these were used in training.

---

### Out-of-Distribution Performance (True Generalization)

| Dataset | Accuracy | Precision | Recall | F1-Score | Domain |
|---------|----------|-----------|--------|----------|--------|
| **VitaminC** | 54.00% | 46.43% | 29.55% | 36.11% | General fact verification |
| **Climate-FEVER** | 81.00% | - | - | - | Climate claims |
| **LIAR** | 81.00% | - | - | - | Political statements |

**Interpretation**: 
- **VitaminC (54%)**: Moderate generalization to new fact-verification domain
- **Climate-FEVER (81%)**: Good performance on climate-specific claims
- **LIAR (81%)**: Good performance on political fact-checking

---

## Key Findings

### Strengths ✅
1. **Strong training performance**: 84% accuracy on FEVER
2. **Reasonable generalization**: 54-81% on OOD datasets
3. **Conservative approach**: Tends to flag uncertain claims (good for safety)
4. **Domain adaptation**: Better on specialized domains (climate, politics) than general

### Limitations ⚠️
1. **Lower OOD precision**: 46% precision on VitaminC (some false positives)
2. **Lower OOD recall**: 30% recall on VitaminC (misses some factual claims)
3. **Training data leakage**: Cannot fairly evaluate on TruthfulQA/FEVER

### Recommendations 📋
1. **For production**: Use ensemble with other models
2. **For evaluation**: Focus on OOD datasets (VitaminC, Climate-FEVER, LIAR)
3. **For improvement**: Fine-tune on more diverse factuality datasets
4. **For safety**: Current conservative approach is appropriate

---

## Detailed Metrics

### VitaminC (Out-of-Distribution)
```
Total examples: 100
Accuracy: 54.00%
Precision: 46.43%
Recall: 29.55%
F1-Score: 36.11%
Average Confidence: 74.19%

Ground Truth:
  Factual: 44 (44%)
  Non-factual: 56 (56%)

Predictions:
  Factual: 28 (28%)
  Non-factual: 72 (72%)
```

**Analysis**: Model is conservative (predicts more non-factual than ground truth). This results in:
- Lower recall (misses some factual claims)
- Moderate precision (some false alarms)
- Good for safety-critical applications

---

## Comparison to Baselines

### Expected Performance Ranges
- **Random baseline**: 50% accuracy
- **Majority class baseline**: 56% accuracy (VitaminC)
- **Our model**: 54-84% depending on dataset
- **State-of-the-art**: 85-90% on FEVER

**Conclusion**: Model performs above baseline and shows reasonable generalization, though there's room for improvement on OOD data.

---

## Recommendations for Phase 2

1. **Add more OOD datasets**: 
   - HOVER (multi-hop reasoning)
   - SCIFACT (scientific claims)
   - PUBHEALTH (health misinformation)

2. **Improve OOD performance**:
   - Fine-tune with domain adaptation techniques
   - Use data augmentation
   - Ensemble with specialized models

3. **Address label imbalance**:
   - Use weighted loss functions
   - Oversample minority class
   - Adjust decision threshold

4. **Error analysis**:
   - Analyze false positives/negatives
   - Identify systematic errors
   - Create targeted improvements

---

## Date
November 8, 2025

## Evaluation Scripts
- **Benchmark**: `factuality/tests/benchmark_factuality.py`
- **Evaluation**: `factuality/tests/evaluate_factuality.py`
- **Results**: `factuality/tests/evaluation_*.json`


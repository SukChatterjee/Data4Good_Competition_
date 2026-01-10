# Data4Good Factuality Classification Challenge

## Overview
This project is my solution for the **Data4Good Case Challenge**, a national competition focused on improving the factual reliability of AI-generated educational content.

The goal is to classify AI-generated answers as:

- **Factual**
- **Contradiction**
- **Irrelevant**

given a question, supporting context, and an AI-generated answer.

---

## The Challenge
AI systems can generate confident but incorrect responses.  
This challenge focuses on detecting such cases to ensure AI advances learning rather than misinformation.

---

## Data
- **Training data:** `train.json`
- **Test data:** `test.json` (contains unique `ID` values)

Predictions are submitted in **JSON format** with the predicted answer type.

---

## Approach (High Level)
The model was built iteratively:

1. Started with a TF-IDF + Logistic Regression baseline
2. Addressed class imbalance using class weights
3. Added linguistic features (overlap, negation, length)
4. Used split TF-IDF for question, context, and answer
5. Added numeric and year consistency checks
6. Finalized with a **Linear SVM** optimized for macro F1

---

## Results
- **Macro F1 Score:** **0.89**
- **Accuracy:** 95.8%

Balanced performance across all three classes, aligning with competition scoring.

---

## Key Learnings
- Accuracy alone is misleading for imbalanced tasks
- Linguistic reasoning improves factuality detection
- Feature engineering can rival complex models
- Metric alignment with judging criteria is critical

## Future Scope
- Incorporate named-entity consistency checks to better detect factual mismatches.
- Explore transformer-based entailment models for deeper semantic contradiction detection.

## Author
**Sukanya Chatterjee**  
University of Cincinnati 

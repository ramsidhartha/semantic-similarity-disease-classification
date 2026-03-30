# Project Status Report: GO-Semantic Dual-Space Lung Cancer Classification

## 1. Project Overview
This project implements a biologically interpretable machine learning pipeline that distinguishes Lung Adenocarcinoma (LUAD) from Lung Squamous Cell Carcinoma (LUSC). Unlike standard models that rely solely on gene expression (raw counts), this architecture is a **Dual-Space Classifier**. It fuses:
1. **Expression Space**: Statistical patterns from raw gene expression (PCA + Top Variance genes).
2. **Gene Ontology (GO) Space**: Functional biological signals extracted using semantic similarity and spectral clustering of the GO hierarchy.

The two spaces are evaluated independently and then fused using a Stacking Meta-Learner (Logistic Regression) to make a final prediction.

## 2. Core Architectural Components
1. **`preprocessing.py`**: Handles log2 transformation, train/val/test splits, Differential Expression (DEG) analysis, and z-score normalization.
2. **`go_processor.py`**: Parses the Gene Ontology (`go-basic.obo`) and experimental annotations (`goa_human.gaf`).
3. **`semantic_similarity.py`**: Computes Information Content (IC) and Lin similarity between genes based on their shared GO terms. Uses a "Local IC" approach (IC derived from the selected DEG gene set rather than the full global corpus).
4. **`feature_extraction.py`**:
   - **Expression Features**: PCA components + Top 30 variance genes.
   - **GO Features**: Clusters genes into functional biological modules using Spectral Clustering on the similarity network. Computes the mean expression of each module.
5. **`classifier.py`**: Trains RandomForest classifiers for the individual spaces, a GradientBoosting model for combined features, and a LogisticRegression meta-learner (with `StandardScaler`) for the stacked prediction.

## 3. Major Refactoring & Correctness Fixes Completed
We recently transformed the codebase from a "proof-of-concept" into a scientifically defensible machine learning pipeline. The following critical bugs and data leakages were resolved:

1. **Fixed Data Leakage in Preprocessing**: 
   - *Previous*: Computed DEG and Z-score normalization on the entire dataset *before* splitting.
   - *Fix*: The pipeline now splits the data first. DEG analysis and standard scaling parameters are fitted strictly on the **training set** and then applied to the validation/test sets to prevent information bleed.
2. **Fixed GO Annotation Parsing**: 
   - *Previous*: The GAF parser incorrectly included "NOT" qualifiers, mapping genes to functions they explicitly do not perform.
   - *Fix*: Added strict filtering to drop all 'NOT' qualified bindings.
3. **Fixed Gene Ordering Mismatch**: 
   - *Previous*: `feature_extraction.py` alphabetically sorted genes before interpreting the Spectral Clustering labels, misaligning the module assignments from the original `go_similarity_matrix.npy`.
   - *Fix*: It now strictly loads `similarity_genes.json` to guarantee the matrix rows/cols correctly correspond to the original gene insertion order. This change alone recovered 3 misclassifications!
4. **Fixed Variance Set Leakage**:
   - *Fix*: Top-30 variance genes are now fitted *only* on the training set and reused for the validation/test subsets.
5. **Classifier Scoring & Stacking Upgrades**:
   - *Fix*: Switched AUC-ROC score calculation to use `.predict_proba()` probabilities instead of binary labels.
   - *Fix*: The Stacking Meta-Learner was upgraded from a 2-branch input to a 3-branch input (`X_expr`, `X_go`, `X_combined`), significantly boosting CV stability.
   - *Fix*: Safely decoupled class index logic using `LabelEncoder.classes_` instead of hardcoding `1` for LUAD.

## 4. Final Validated Results
With all data leakage eliminated and modules aligning correctly, the current validated performance on the held-out validation set (170 samples) is extremely strong:

| Model | 5-Fold CV Accuracy | Validation Accuracy | AUC-ROC |
|-------|--------------------|-----------------------|---------|
| Expression-only (RF) | 92.6% | 95.3% | 0.971 |
| GO-only (RF) | 91.9% | 94.1% | 0.959 |
| Combined (concat) | 92.6% | 95.3% | 0.981 |
| **Stacked (meta-learner)** | **93.0%** | **95.3%** | **0.972** |

*Only 8 out of 170 validation samples are misclassified by the stacked model.*

## 5. Next Steps: The Interpretability Suite
The pipeline is now highly accurate and robust. The final major objective to transition to before final presentation is building an **Interpretability Suite** (`src/interpretability.py`) containing 7 planned features:

1. **Bootstrap Meta-Learner Coefficients**: Resample the training data 1000 times to generate confidence intervals for the Stacking Meta-Learner weights (proving neither branch is redundant).
2. **Per-Branch Feature Importances**: Extract Top-10 features (mapped to real gene/module names) from the Random Forest models.
3. **GO Enrichment per Module**: Automatically query the `go_processor` to name the dominant biological functions inside each Spectral Clustering module.
4. **Stacking Breakdown & Sample Sentences**: Generate human-readable narrative text explaining exactly why the meta-learner decided on a specific prediction for a specific patient.
5. **SHAP Explanations**: Use TreeSHAP on the Expression RF to generate Beeswarm and Waterfall plots for high-confidence vs misclassified samples.
6. **Calibration Plots**: Draw probability-vs-frequency curves to evaluate if the model's confidence scores are experimentally trustworthy.

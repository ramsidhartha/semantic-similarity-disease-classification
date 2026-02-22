# Dual-Space Classification of Lung Cancer Subtypes Using Gene Ontology Semantic Similarity

## 1. Problem Statement

Classifying lung cancer subtypes (LUAD vs LUSC) from gene expression data is clinically important for treatment decisions. Traditional approaches treat genes as independent numerical features, ignoring biological relationships. We propose a dual-space architecture that integrates Gene Ontology (GO) semantic similarity with expression-based features to achieve biologically interpretable classification.

---

## 2. Dataset

- **Source**: TCGA (The Cancer Genome Atlas) via UCSC Xena API
- **Samples**: 1,129 total (576 LUAD, 553 LUSC)
- **Genes**: 20,531 gene expression measurements per sample (RNA-Seq)
- **Class balance**: Near-perfect (51% / 49%)

---

## 3. Proposed Architecture

```
Input: Gene Expression Matrix (1129 samples x 20531 genes)
                          |
              Differential Expression Analysis
              (Welch's t-test, BH correction)
                          |
              Top 100 DEGs selected
              (|log2FC| > 1.0, adj p < 0.01)
                          |
            GO Annotation Mapping (GOATools)
            90/100 genes mapped (90% coverage)
                          |
        +-----------------+------------------+
        |                                    |
   GO Semantic Space                  Expression Space
   (Biology-driven)                   (Data-driven)
        |                                    |
   Lin Similarity (IC-based)            PCA reduction
   (Best-Match Average)                Top variable genes
                                            |
        |                              50 features (20 PCA + 30 top genes)
   GO-coherent gene modules                 |
   (Spectral Clustering, k=9)              |
        |                                    |
   12 GO features:                          |
    - 9 module activity scores              |
   - Centroid similarity to LUAD            |
   - Centroid similarity to LUSC            |
   - Centroid difference score              |
        |                                    |
        +-----------------+------------------+
                          |
                  Feature Fusion Layer
                  (Concatenation + Scaling)
                          |
                   62 combined features
                          |
               Dual-Space Classifier
        +------------+----------+---------------+
        |            |          |               |
    Expression   GO-only    Combined (main model)
    RF (d=10)    RF (d=8)   Gradient Boosting
    (baseline)   (baseline) (100 trees, lr=0.1)
        +------------+----------+---------------+
                          |
                  LUAD vs LUSC Prediction
```

---

## 4. Methodology

### 4.1 Data Preprocessing
- Log2 transformation and z-score normalization of expression values
- Stratified train/validation/test split (70/15/15)
  - Train: 789 samples
  - Validation: 170 samples
  - Test: 170 samples (held out)

### 4.2 Gene Selection
- Welch's t-test for differential expression between LUAD and LUSC
- Benjamini-Hochberg correction for multiple testing
- Selection criteria: |log2FC| > 1.0 and adjusted p-value < 0.01
- Result: 828 significant DEGs, top 100 selected by fold change

### 4.3 GO Annotation Mapping
- GO ontology: go-basic.obo (42,666 terms)
- Human gene annotations: goa_human.gaf
- Filtered to Biological Process (BP) namespace
- Coverage: 90/100 genes have GO:BP annotations (90%)
- Average 8.8 GO terms per annotated gene, 556 unique GO terms

### 4.4 Semantic Similarity Computation
- Method: Lin similarity (Information Content-based)
  - Measures term similarity via the Most Informative Common Ancestor (MICA)
  - Selected over hybrid (Lin+Wang) after ablation study (corr=0.81, no significant CV improvement)
- IC computed from annotation frequency in our gene set
- Gene-level similarity: Best-Match Average (bidirectional)
- Output: 90x90 pairwise gene similarity matrix (all annotated genes, no sampling)

### 4.5 GO Feature Engineering
- Spectral clustering on GO similarity matrix to form 9 gene modules
- k=9 selected via eigenvalue spectrum analysis (second-largest eigengap at k=9)
- Module activity: mean expression of module member genes per sample
- Centroid similarity: correlation-based distance to LUAD/LUSC expression centroids (computed on training data only)

### 4.6 Expression Feature Engineering
- PCA: top 20 principal components
- Top 30 high-variance genes from selected DEGs

### 4.7 Classification
- Expression-only baseline: Random Forest (100 trees, max depth 10)
- GO-only baseline: Random Forest (100 trees, max depth 8)
- Combined (main model): Gradient Boosting (100 trees, learning rate 0.1, max depth 5)
- We experimented with stacking (LogReg meta-learner over base model predictions), but it did not improve cross-validated performance. Therefore, we retained the simpler combined model.
- Evaluation: 5-fold stratified cross-validation

---

## 5. Results

### 5.1 Validation Set Performance

| Model             | Accuracy | F1 Score | AUC-ROC |
|-------------------|----------|----------|---------|
| Expression-only   | 93.5%    | 0.939    | 0.935   |
| GO-only           | 94.7%    | 0.950    | 0.946   |
| **Combined**      | **91.2%**| **0.917**| **0.911**|

### 5.2 5-Fold Cross-Validation

| Model           | Mean Accuracy | Std  |
|-----------------|---------------|------|
| Expression-only | 92.6%         | 0.8% |
| GO-only         | 91.7%         | 0.7% |
| **Combined**    | **93.1%**     | 0.8% |

---

## 6. Key Observations

1. Both spaces achieve comparable validation accuracy (~94%) when viewing the same 100 genes, confirming true dual-space design.
2. Lin similarity was selected over hybrid (Lin+Wang) after ablation study showed no significant CV improvement (corr=0.81).
3. k=9 modules selected via eigenvalue spectrum analysis; k=25 produced 9 singletons and higher variance.
4. 5-fold CV shows Combined features (**93.1%**) outperform individual spaces (Expression: 92.6%, GO: 91.7%), confirming complementary information.
5. Stacking ensemble did not improve over the simpler combined model, so we retained Gradient Boosting as the main classifier.
6. Case study analysis identifies 15/170 misclassified samples on validation, with GO and expression often disagreeing on boundary cases.
7. 90% GO annotation coverage (90/100 genes) ensures both spaces operate on a consistent, well-annotated gene set.

---

## 7. Current Project File Structure

```
semantic-similarity-disease-classification/
    data/
        raw/
            luad_expression.csv
            lusc_expression.csv
            lung_cancer_combined.csv
        go/
            go-basic.obo
            goa_human.gaf
        processed/
            train.csv, val.csv, test.csv
            deg_analysis.csv, deg_significant.csv
            gene_go_mapping.json
            go_similarity_matrix.npy
            selected_genes.txt
            *_features_*.npy
    src/
        data_acquisition.py
        preprocessing.py
        go_processor.py
        semantic_similarity.py
        feature_extraction.py
        classifier.py
        visualization.py
        run_pipeline.py
    results/
        model_comparison.json
        figures/
            volcano_plot.png
            go_heatmap.png
            go_network.png
            model_comparison.png
    notebooks/
        01_exploratory_data_analysis.ipynb
```

---

## 8. Remaining Work

- SHAP-based feature importance analysis
- Biological interpretation of top GO modules
- Comparison with known LUAD/LUSC markers from literature
- Final report with novel findings

---

## 9. Tools and Libraries

| Component        | Tool/Library     |
|------------------|------------------|
| Data source      | TCGA / UCSC Xena |
| GO ontology      | GOATools          |
| ML models        | scikit-learn      |
| Visualization    | matplotlib, seaborn, networkx |
| Language         | Python 3.13       |

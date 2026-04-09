# Dual-Space Classification of Lung Cancer Subtypes Using Gene Ontology Semantic Similarity

## 1. Problem Statement

Classifying lung cancer subtypes (LUAD vs LUSC) from gene expression data is clinically important for treatment decisions. Traditional approaches treat genes as independent numerical features, ignoring biological relationships. We propose a dual-space architecture that integrates Gene Ontology (GO) semantic similarity with expression-based features to achieve biologically interpretable classification.

---

## 2. Dataset

- **Source**: TCGA (The Cancer Genome Atlas) via UCSC Xena API
- **Samples**: 1,129 total (576 LUAD, 553 LUSC)
- **Genes**: 20,531 gene expression measurements per sample (RNA-Seq)
- **Split**: Train 789 / Validation 170 / Test 170 (stratified 70/15/15)
- **Class balance**: Near-perfect (51% / 49%)

---

## 3. Architecture

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
   Best-Match Average (BMA)            Top variable genes
        |                                    |
   Spectral Clustering (k=9)          50 features
   9 gene modules                     (20 PCA + 30 top-var)
        |                                    |
   12 GO features:                           |
    - 9 module activity scores               |
    - Centroid sim to LUAD                   |
    - Centroid sim to LUSC                   |
    - Centroid difference score              |
        |                                    |
        +-----------------+------------------+
                          |
                  62 combined features
                          |
               Dual-Space Classifier
        +------------+----------+---------------+
        |            |          |               |
    Expression   GO-only    Combined        Stacked
    RF (d=10)    RF (d=8)   GBM (main)    Meta-LR
    (baseline)  (baseline) (100 trees)   (on OOF probs)
        +------------+----------+---------------+
                          |
                  LUAD vs LUSC Prediction
```

---

## 4. Methodology

### 4.1 Data Preprocessing
- Log2 transformation and z-score normalization (fitted on train only)
- Stratified split before any analysis to prevent leakage

### 4.2 Gene Selection
- Welch's t-test for differential expression between LUAD and LUSC
- Benjamini-Hochberg correction for multiple testing
- Criteria: |log2FC| > 1.0 and adjusted p-value < 0.01
- Result: 828 significant DEGs, top 100 selected by fold change

### 4.3 GO Annotation Mapping
- GO ontology: go-basic.obo (42,666 terms)
- Human gene annotations: goa_human.gaf
- Filtered to Biological Process (BP) namespace only
- NOT-qualified annotations excluded
- Coverage: 90/100 genes annotated, average 8.8 GO terms per gene, 556 unique GO terms

### 4.4 Semantic Similarity Computation
- **Method**: Lin (1998) similarity via Most Informative Common Ancestor (MICA)
- **IC computation**: Global corpus (goa_human.gaf, ~20k genes) with full DAG propagation
  — each annotation propagated up to all ancestors, ensuring IC(ancestor) ≤ IC(child)
  — this bounds Lin similarity strictly to [0, 1]
- **Gene-level similarity**: Best-Match Average (BMA) — bidirectional, symmetric
- **Output**: 84×84 pairwise gene similarity matrix

### 4.5 GO Feature Engineering
- Spectral clustering on GO similarity matrix (k=9, affinity=precomputed)
- k=9 selected via eigenvalue spectrum analysis (second-largest eigengap at k=9, gap=0.112)
- Module activity: mean expression of module members per sample
- Centroid similarity: correlation-based distance to LUAD/LUSC centroids (train-only)

### 4.6 Expression Feature Engineering
- PCA: top 20 principal components (fitted on train, applied to val/test)
- Top 30 high-variance genes (variance computed on train only)

### 4.7 Classification
- Expression baseline: Random Forest (100 trees, max depth 10)
- GO baseline: Random Forest (100 trees, max depth 8)
- Combined: Gradient Boosting (100 trees, learning rate 0.1, max depth 5)
- **Stacked Meta-Learner**: Logistic Regression with StandardScaler trained on out-of-fold probabilities from all three branches (5-fold OOF)

---

## 5. Experiments

### 5.1 IC Computation Fix — Local vs Global Corpus

**Experiment:**
The original implementation computed Information Content using only the 84 DEG genes as the annotation corpus. This was mathematically invalid: with a small local corpus, MICA IC can exceed the IC of its child terms, causing Lin similarity values > 1 (observed mean=1.65, max=2.87).

**Finding:**
Switching to a global corpus (all ~20k human genes in goa_human.gaf) without propagation still produced values > 1 (mean=1.193, max=2.796). Root cause: GAF files list only direct annotations; without propagating annotations up the GO DAG, ancestor terms accumulate fewer counts than their descendants.

**Fix Applied to Pipeline:**
Full DAG propagation on the global corpus. For every gene's direct annotation, all ancestor terms are credited. This guarantees IC(ancestor) ≤ IC(child), bounding Lin similarity to [0, 1].

**ML/DS Impact:**
- Before fix: degenerate spectral clustering [1,1,1,1,1,1,1,35,42] — one giant cluster, all others singletons
- After fix: balanced 9-module structure [4,4,5,5,8,9,10,10,29]
- GO-only model accuracy: 92.4% → 94.1%
- Combined model accuracy: 84.1% → 94.1%
- GO similarity matrix: Lin-Wang Pearson correlation 0.456 (broken) → 0.924 (fixed)

---

### 5.2 Similarity Measure Ablation — Lin vs Wang

**Experiment:**
Compared Lin (IC-based) and Wang (graph-based S-value) similarity measures. Computed both 84×84 matrices and ran 5-fold CV on the combined model using each.

**Finding:**
Pearson correlation between Lin and Wang matrices: 0.924 (post-fix) — high redundancy. All three approaches (Lin-only, Wang-only, Hybrid) yield comparable CV performance within each other's standard deviations.

| Measure | Val Acc | 5-fold CV |
|---------|---------|-----------|
| Lin-only | 94.1% | 93.6% ± 1.0% |
| Wang-only | 93.5% | 93.2% ± 0.7% |
| Hybrid (0.5+0.5) | 95.3% | 93.0% ± 1.5% |

**Fix Applied to Pipeline:**
Lin-only adopted. Most established measure in the literature (Lin, 1998), best CV accuracy, avoids Wang's additional graph traversal cost.

---

### 5.3 Cluster Count Ablation — k Selection

**Experiment:**
Eigenvalue spectrum analysis of the GO similarity matrix Laplacian. Tested k=5, 9, 15, 25 for singleton rate, GO CV, and combined CV.

**Finding:**

| k | Singletons | GO CV | Combined CV | Std |
|---|-----------|-------|-------------|-----|
| 5 | 0/5 | 92.1% | 93.0% | 0.7% |
| **9** | **0/9** | **93.1%** | **93.3%** | **1.3%** |
| 15 | 1/15 | 93.0% | 93.4% | 1.0% |
| 25 | 9/25 | 93.9% | 92.8% | 1.6% |

k=25 produced 9 singleton modules (1-gene clusters) and highest variance. k=9 aligns with the eigenspectrum (eigengap=0.112), zero singletons, stable performance.

**Fix Applied to Pipeline:**
k=9 adopted.

---

### 5.4 Core vs Periphery Signal Experiment (Sandbox)

**Experiment:**
Tested whether the largest GO spectral cluster (29 genes — squamous differentiation TFs and keratins) carries most of the classification signal, or whether signal is distributed across all 84 genes. RF classifier (100 trees, max depth 10, 5-fold CV) run separately on core genes (n=29), periphery genes (n=55), and full DEG set (n=100).

**Finding:**

| Gene Set | Genes | CV Accuracy | Val Accuracy |
|----------|-------|-------------|--------------|
| Full DEG set | 100 | 92.7% | 93.5% |
| Core only (largest cluster) | 29 | 92.6% | 92.4% |
| Periphery only (rest) | 55 | 92.5% | 92.9% |

All three subsets achieve ~92.5% CV accuracy — within noise of each other.

**Implication:**
The LUAD/LUSC signal is genome-wide and not concentrated in any single functional cluster. Any ~30-gene subset of strongly differentially expressed genes achieves near-ceiling accuracy. This is the multi-gene signal paradox — it explains why GO adds interpretability rather than raw accuracy improvement.

**No pipeline change** — confirms existing design is appropriate.

---

### 5.5 Eigengene Correlation Experiment (Sandbox)

**Experiment:**
Tested whether the 9 GO modules are independently regulated (parallel LUSC markers) or co-regulated (unified differentiation program). Computed eigengenes (first PC) of each cluster and ran Pearson/Spearman correlations: (a) naive on all 1,129 samples, (b) within-LUSC only (n=553), (c) within-LUAD only (n=576). Within-cancer analysis removes the binary label as a confounder.

**Finding — Within LUSC (n=553 samples):**

| Cluster 1 vs | Pearson r | p-value | Interpretation |
|-------------|-----------|---------|----------------|
| Calcium/S100 | 0.899 | <10⁻¹⁰⁰ | STRONG co-movement |
| Ion channels | 0.881 | <10⁻¹⁰⁰ | STRONG co-movement |
| Cornification | 0.875 | <10⁻¹⁰⁰ | STRONG co-movement |
| Serine proteases | 0.847 | <10⁻¹⁰⁰ | STRONG co-movement |
| Desmosomes | 0.823 | <10⁻¹⁰⁰ | STRONG co-movement |
| Type II keratins | 0.810 | <10⁻¹⁰⁰ | STRONG co-movement |
| Mixed effectors | 0.800 | <10⁻¹⁰⁰ | STRONG co-movement |
| TMPRSS proteases | 0.767 | <10⁻¹⁰⁰ | STRONG co-movement |

**Effector-to-effector correlations within LUSC:**
Cornification vs Desmosomes: r=0.818 | Cornification vs Type II keratins: r=0.791 | Desmosomes vs Type II keratins: r=0.665

**Implication:**
All 9 clusters co-activate together within LUSC samples. Cluster 1 (lineage identity — TFs and core keratins) is the most coherent module and co-moves strongly with all terminal effector modules. The GO modules do not represent independent pathways; they reflect a single unified squamous differentiation program that is collectively activated in LUSC and collectively suppressed in LUAD.

**No pipeline change** — this is a biological validation of the GO feature design.

---

## 6. Biological Results

### 6.1 The 9 GO Functional Modules (k=9 Spectral Clustering)

The 84 GO-annotated DEGs cluster into 9 biologically coherent modules:

| Cluster | n | Biological Identity | Representative Genes |
|---------|---|--------------------|--------------------|
| 1 | 29 | Lineage Identity — squamous TFs + core keratins | TP63, KRT5, KRT14, KRT6A, KRT6B, DSP |
| 2 | 10 | Calcium-binding / S100 proteins | S100A2, S100A7, S100A8, S100A9, S100A14 |
| 3 | 10 | Ion channels and transporters | KCNK5, SLC6A14, CLCA2, ANO1 |
| 4 | 9 | Cornification — SPRR/LCE barrier proteins | SPRR1A, SPRR1B, SPRR2A, LCE3D |
| 5 | 8 | Serine proteases and MAGE antigens | KLK5, KLK6, KLK7, MAGEA3 |
| 6 | 5 | Desmosomes — cell-cell adhesion | DSC2, DSC3, DSG2, DSG3 |
| 7 | 5 | Type II keratins | KRT1, KRT2, KRT10, KRT76 |
| 8 | 4 | Mixed effectors | FAT2, PERP, TRIM29 |
| 9 | 4 | TMPRSS serine proteases | TMPRSS4, TMPRSS11D, TMPRSS11E |

### 6.2 Biological Interpretation: Squamous Differentiation Program

The 9 modules collectively represent the **squamous cell differentiation program**:

- **LUSC** activates this entire program: squamous cells undergo terminal differentiation, expressing keratins (structural), S100 proteins (calcium signalling), desmosomes (cell adhesion), cornification proteins (barrier formation), and serine proteases (ECM remodelling).
- **LUAD** suppresses this program: adenocarcinoma derives from glandular/alveolar cells with no squamous identity, so all 9 modules are collectively downregulated.

This is not 9 independent signals — the eigengene correlation experiment confirms it is one co-regulated program switching on/off as a unit.

### 6.3 LUSC-Enriched vs LUAD-Enriched GO Terms

| Cancer | Enriched Biological Processes |
|--------|-------------------------------|
| LUSC | Keratinization, Keratinocyte differentiation, Cornification, Epidermis development |
| LUAD | Cell adhesion, Extracellular matrix organization, Epithelium development |

These align precisely with established LUAD (glandular origin) and LUSC (squamous/keratinization origin) biology in published literature.

### 6.4 What GO Adds Over Expression

GO features do not improve raw accuracy (both spaces achieve ~94%). The value of GO is:

1. **Named biological modules**: each of the 9 clusters maps to a known biological process, allowing a clinician or biologist to say "LUSC shows high cornification and desmosome activity" rather than "feature 23 is high."
2. **Biological validation**: the fact that GO-clustered genes show extreme co-movement within a cancer type (r=0.767–0.899) confirms the classifier is using real biology, not statistical artefacts.
3. **Robustness across feature views**: GO-only (94.1%) matches expression-only (95.3%) without using raw expression magnitudes — it captures functional pathway activity independently.

---

## 7. ML/DS Results

### 7.1 Validation Set Performance (Final — Post IC Fix)

| Model | Accuracy | F1 Score | AUC-ROC |
|-------|----------|----------|---------|
| Expression-only | 95.3% | 0.955 | 0.973 |
| GO-only | 94.1% | 0.943 | 0.958 |
| Combined | 94.1% | 0.943 | 0.979 |
| **Stacked (Meta-Learner)** | **94.7%** | **0.949** | **0.973** |

### 7.2 5-Fold Cross-Validation (Final — Post IC Fix)

| Model | Mean Accuracy | Std Dev |
|-------|---------------|---------|
| Expression-only | 92.6% | 0.7% |
| GO-only | 92.5% | 1.1% |
| Combined | 92.7% | 1.0% |
| **Stacked (Meta-Learner)** | **95.3%** | **0.6%** |

The Stacked Meta-Learner is the best model: 95.3% CV with 0.6% standard deviation — the tightest spread of all models.

### 7.3 Pipeline Changes Made During the Project

| Change | Reason | Impact |
|--------|--------|--------|
| Split-first, DEG on train only | Prevent preprocessing leakage | Scientific validity |
| Z-score with train params applied to val/test | Prevent normalization leakage | Scientific validity |
| NOT qualifier filtering in GO parser | Was incorrectly including negated annotations | GO annotation correctness |
| Gene ordering fix (similarity_genes.json alignment) | Alphabetical sort caused row/col mismatch | Spectral clustering correctness |
| Top-variance gene selection on train only | Prevent feature selection leakage | Scientific validity |
| IC computation: global corpus + DAG propagation | Local corpus breaks Lin similarity bounds | Core fix — Lin bounded [0,1] |
| k=9 (from k=25) | Eigenspectrum-aligned, zero singletons | Module quality |
| Lin-only (from hybrid Lin+Wang) | High redundancy (r=0.924), Lin wins CV | Simpler, interpretable |
| Stacking meta-learner added | Best CV performance, stable | Final model |

### 7.4 Case Study — Misclassified Samples

Only 9 of 170 validation samples misclassified. Representative cases:

| Sample ID | True | Predicted | GO Confidence | Expr Confidence |
|-----------|------|-----------|---------------|-----------------|
| TCGA-66-2756 | LUSC | LUAD | 0.96 | 0.95 |
| TCGA-56-7731 | LUSC | LUAD | 0.76 | 0.75 |
| TCGA-22-1017 | LUSC | LUAD | 1.00 | 0.88 |

Both GO and expression models agree on the wrong prediction — suggests genuine biological boundary cases (mixed adeno-squamous pathology) rather than model error.

---

## 8. Tools and Libraries

| Component | Tool/Library |
|-----------|-------------|
| Data source | TCGA / UCSC Xena API |
| GO ontology | GOATools, go-basic.obo, goa_human.gaf |
| ML models | scikit-learn |
| Visualization | matplotlib, seaborn |
| Language | Python 3.13 |

---

## 9. File Structure

```
semantic-similarity-disease-classification/
    data/
        raw/         — TCGA expression matrices
        go/          — go-basic.obo, goa_human.gaf
        processed/   — train/val/test CSVs, DEG analysis,
                       gene_go_mapping.json, go_similarity_matrix.npy
    src/
        data_acquisition.py
        preprocessing.py
        go_processor.py
        semantic_similarity.py   ← IC fix here
        feature_extraction.py
        classifier.py
        visualization.py
        run_pipeline.py
    experiments/
        core_periphery_experiment.py   ← sandbox, read-only
        eigengene_correlation.py       ← sandbox, read-only
    results/
        model_comparison.json
        figures/
            volcano_plot.png
            go_heatmap.png
            roc_curves.png
            confusion_matrices.png
```

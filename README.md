# GO-Semantic Dual-Space Lung Cancer Classification

Interpretable classifier distinguishing **Lung Adenocarcinoma (LUAD)** from **Lung Squamous Cell Carcinoma (LUSC)** by fusing gene expression data with Gene Ontology (GO) biological knowledge.

## Key Results

| Model | 5-Fold CV Accuracy | Validation Accuracy | AUC-ROC |
|-------|--------------------|-----------------------|---------|
| Expression-only (RF) | 92.6% | 95.3% | 0.971 |
| GO-only (RF) | 91.9% | 94.1% | 0.959 |
| Combined (concat) | 92.6% | 95.3% | 0.981 |
| **Stacked (meta-learner)** | **93.0%** | **95.3%** | **0.972** |

All results use leak-free preprocessing (split-first, train-only DEG + z-score), correctly ordered GO module features, and probability-based AUC-ROC. Only 8/170 validation samples misclassified.

## Architecture

```
Gene Expression (1129 samples × 20531 genes)
            ↓
    Top 100 DEGs (Welch's t-test + BH correction)
            ↓
    GO Annotation Mapping (84/100 genes annotated)
            ↓
    ┌───────────────┬───────────────┐
    │  GO Space     │  Expr Space   │
    │               │               │
    │  Lin Sim →    │  PCA (20) +   │
    │  Spectral     │  Top 30 genes │
    │  Clustering   │               │
    │  (k=9)        │               │
    │               │               │
    │  12 features  │  50 features  │
    └───────┬───────┴───────┬───────┘
            │               │
            └───── 62 combined features
                        ↓
              Stacking Meta-Learner (LR)
                        ↓
               LUAD vs LUSC Prediction
```

## Design Choices (Empirically Validated)

Each design decision was tested via ablation study:

| Component | Alternatives Tested | Chosen | Reason |
|-----------|-------------------|--------|--------|
| Similarity | Lin, Wang, Hybrid | **Lin-only** | Best CV accuracy; Lin & Wang corr=0.81 (redundant) |
| Cluster count | k=5, 9, 15, 25 | **k=9** | Eigengap-aligned; k=25 produced 9 singletons |
| Ensemble | Concat, Stacking | **Stacking** | Meta-learner learns optimal branch weighting (Expr: 4.5, GO: 3.0) |

## Setup

```bash
# Clone and setup
git clone https://github.com/YOUR_USERNAME/semantic-similarity-disease-classification.git
cd semantic-similarity-disease-classification
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Download data (TCGA + GO files)
python src/data_acquisition.py

# Run full pipeline
python src/run_pipeline.py
```

## Project Structure

```
src/
├── data_acquisition.py    # Downloads TCGA + GO data
├── preprocessing.py       # Normalization, DEG selection, splits
├── go_processor.py        # GO annotation mapping
├── semantic_similarity.py # Lin similarity computation
├── feature_extraction.py  # Spectral clustering, feature engineering
├── classifier.py          # Training, evaluation, cross-validation
├── visualization.py       # Figures generation
└── run_pipeline.py        # End-to-end pipeline runner

docs/
├── GUIDE_DOCUMENT.md      # Comprehensive project report
└── PROJECT_DOCUMENTATION.md # Technical documentation

results/
├── figures/               # Volcano plot, heatmaps, network, comparison
├── similarity_matrix_sparse.png
└── model_comparison.json
```

## Data

Data files are excluded from the repo due to size (~480MB). Run `python src/data_acquisition.py` to download:
- **TCGA RNA-seq**: 576 LUAD + 553 LUSC samples (GDC Portal)
- **GO Ontology**: go-basic.obo (42,666 terms)
- **GO Annotations**: goa_human.gaf (Human gene-GO mappings)

## Documentation

- [Guide Document](docs/GUIDE_DOCUMENT.md) — Full project report with methodology, results, and design choices
- [Project Documentation](docs/PROJECT_DOCUMENTATION.md) — Technical reference with architecture and methodology details

## License

MIT License

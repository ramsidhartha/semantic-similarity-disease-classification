#!/usr/bin/env python3
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from preprocessing import main as preprocess_main
from go_processor import main as go_processor_main
from semantic_similarity import main as similarity_main
from feature_extraction import main as feature_main
from classifier import main as classifier_main
from visualization import generate_all_visualizations


def run_pipeline():
    print("="*60)
    print("DUAL-SPACE LUNG CANCER CLASSIFICATION PIPELINE")
    print("="*60)
    
    print("\n[STEP 1/6] Data Preprocessing")
    print("-"*40)
    preprocess_main()
    
    print("\n[STEP 2/6] GO Annotation Processing")
    print("-"*40)
    go_processor_main()
    
    print("\n[STEP 3/6] Semantic Similarity Computation")
    print("-"*40)
    similarity_main()
    
    print("\n[STEP 4/6] Feature Extraction")
    print("-"*40)
    feature_main()
    
    print("\n[STEP 5/6] Model Training and Evaluation")
    print("-"*40)
    classifier_main()
    
    print("\n[STEP 6/6] Generating Visualizations")
    print("-"*40)
    generate_all_visualizations()
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)
    print("\nOutputs:")
    print("  - data/processed/     : Processed data and features")
    print("  - results/            : Model performance metrics")
    print("  - results/figures/    : Visualizations for presentation")


if __name__ == "__main__":
    run_pipeline()

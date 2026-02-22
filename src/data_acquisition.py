"""
Data Acquisition Script for TCGA Lung Cancer Data

Downloads gene expression data for:
- LUAD (Lung Adenocarcinoma)  
- LUSC (Lung Squamous Cell Carcinoma)

Using UCSC Xena Python API
"""

import os
import pandas as pd
import xenaPython as xena

# Configuration
TCGA_HUB = "https://tcga.xenahubs.net"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")

# TCGA dataset names for gene expression (RNA-Seq)
DATASETS = {
    "LUAD": "TCGA.LUAD.sampleMap/HiSeqV2",
    "LUSC": "TCGA.LUSC.sampleMap/HiSeqV2"
}


def get_samples(cancer_type: str) -> list:
    """Get all sample IDs for a given cancer type."""
    dataset = DATASETS[cancer_type]
    samples = xena.dataset_samples(TCGA_HUB, dataset, None)
    print(f"  Found {len(samples)} samples for {cancer_type}")
    return samples


def get_genes(cancer_type: str) -> list:
    """Get all gene names (probes) for a dataset."""
    dataset = DATASETS[cancer_type]
    genes = xena.dataset_field(TCGA_HUB, dataset)
    print(f"  Found {len(genes)} genes for {cancer_type}")
    return genes


def download_expression_data(cancer_type: str) -> pd.DataFrame:
    """
    Download gene expression matrix for a cancer type.
    
    Returns:
        DataFrame with samples as rows and genes as columns
    """
    print(f"\nDownloading {cancer_type} data...")
    
    dataset = DATASETS[cancer_type]
    
    # Get samples and genes
    samples = get_samples(cancer_type)
    genes = get_genes(cancer_type)
    
    # Download expression data
    # Note: xena returns genes as rows, samples as columns
    print(f"  Fetching expression matrix (this may take a few minutes)...")
    expression_data = xena.dataset_fetch(TCGA_HUB, dataset, samples, genes)
    
    # Convert to DataFrame
    # expression_data is a list of lists: each inner list is one gene's values across samples
    df = pd.DataFrame(expression_data, index=genes, columns=samples)
    
    # Transpose so samples are rows (standard ML format)
    df = df.T
    
    # Add cancer type label
    df.insert(0, 'cancer_type', cancer_type)
    
    print(f"  Final shape: {df.shape}")
    return df


def main():
    """Main function to download all data."""
    print("=" * 60)
    print("TCGA Lung Cancer Data Acquisition")
    print("=" * 60)
    
    # Create output directory if needed
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    all_data = []
    
    for cancer_type in DATASETS.keys():
        try:
            df = download_expression_data(cancer_type)
            
            # Save individual file
            output_path = os.path.join(OUTPUT_DIR, f"{cancer_type.lower()}_expression.csv")
            df.to_csv(output_path, index_label='sample_id')
            print(f"  Saved to: {output_path}")
            
            all_data.append(df)
            
        except Exception as e:
            print(f"  ERROR downloading {cancer_type}: {e}")
            raise
    
    # Combine all data
    print("\n" + "=" * 60)
    print("Combining datasets...")
    combined_df = pd.concat(all_data, axis=0)
    
    # Save combined file
    combined_path = os.path.join(OUTPUT_DIR, "lung_cancer_combined.csv")
    combined_df.to_csv(combined_path, index_label='sample_id')
    print(f"Combined data saved to: {combined_path}")
    print(f"Total samples: {len(combined_df)}")
    print(f"  - LUAD: {len(combined_df[combined_df['cancer_type'] == 'LUAD'])}")
    print(f"  - LUSC: {len(combined_df[combined_df['cancer_type'] == 'LUSC'])}")
    
    print("\n" + "=" * 60)
    print("Data acquisition complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

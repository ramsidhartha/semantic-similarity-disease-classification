import os
import json
from collections import defaultdict
from goatools.obo_parser import GODag

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
GO_DIR = os.path.join(BASE_DIR, "data", "go")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")


def load_go_dag():
    obo_path = os.path.join(GO_DIR, "go-basic.obo")
    godag = GODag(obo_path)
    return godag


def load_gene_annotations():
    """
    Parse human GO annotations from GAF file.
    Methodology: gene symbol → Biological Process (aspect='P') GO IDs only.
    'NOT' qualifiers are excluded (gene explicitly does NOT have that function).
    Note: gene symbol matching may miss aliases; 84% coverage on our DEG set is acceptable.
    """
    gaf_path = os.path.join(GO_DIR, "goa_human.gaf")

    gene_to_go = defaultdict(set)
    go_to_genes = defaultdict(set)

    with open(gaf_path, 'r') as f:
        for line in f:
            if line.startswith('!'):
                continue
            parts = line.strip().split('\t')
            if len(parts) < 15:
                continue

            qualifier = parts[3]
            gene_symbol = parts[2]
            go_id = parts[4]
            aspect = parts[8]

            # Skip "NOT" qualifiers — gene explicitly lacks this function
            if 'NOT' in qualifier:
                continue

            if aspect == 'P':
                gene_to_go[gene_symbol].add(go_id)
                go_to_genes[go_id].add(gene_symbol)

    return dict(gene_to_go), dict(go_to_genes)


def filter_genes_with_go(selected_genes, gene_to_go, min_terms=1):
    filtered = {}
    for gene in selected_genes:
        if gene in gene_to_go and len(gene_to_go[gene]) >= min_terms:
            filtered[gene] = list(gene_to_go[gene])
    return filtered


def main():
    print("Loading GO DAG...")
    godag = load_go_dag()
    print(f"Loaded {len(godag)} GO terms")

    print("\nLoading gene annotations...")
    gene_to_go, go_to_genes = load_gene_annotations()
    print(f"Loaded annotations for {len(gene_to_go)} genes")

    genes_path = os.path.join(PROCESSED_DIR, "selected_genes.txt")
    with open(genes_path, 'r') as f:
        selected_genes = [line.strip() for line in f]

    print(f"\nFiltering {len(selected_genes)} selected genes...")
    filtered_mapping = filter_genes_with_go(selected_genes, gene_to_go, min_terms=1)
    print(f"Genes with GO annotations: {len(filtered_mapping)}/{len(selected_genes)}")

    mapping_path = os.path.join(PROCESSED_DIR, "gene_go_mapping.json")
    with open(mapping_path, 'w') as f:
        json.dump(filtered_mapping, f)

    stats = {
        'total_genes': len(selected_genes),
        'genes_with_go': len(filtered_mapping),
        'coverage': len(filtered_mapping) / len(selected_genes),
        'avg_terms_per_gene': sum(len(v) for v in filtered_mapping.values()) / len(filtered_mapping) if filtered_mapping else 0,
        'unique_go_terms': len(set(t for terms in filtered_mapping.values() for t in terms))
    }

    stats_path = os.path.join(PROCESSED_DIR, "go_stats.json")
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\nGO Statistics:")
    for k, v in stats.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")

    return godag, filtered_mapping


if __name__ == "__main__":
    main()

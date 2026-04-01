import os
import json
import numpy as np
from goatools.obo_parser import GODag

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
GO_DIR = os.path.join(BASE_DIR, "data", "go")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")


class SemanticSimilarity:
    def __init__(self):
        self.godag = GODag(os.path.join(GO_DIR, "go-basic.obo"))

        with open(os.path.join(PROCESSED_DIR, "gene_go_mapping.json"), 'r') as f:
            self.gene_to_go = json.load(f)

        # Gene order is determined by JSON insertion order (Python 3.7+ dicts preserve order).
        # similarity_genes.json is saved alongside the matrix so downstream steps
        # can always recover the correct row/column mapping.
        self.genes = list(self.gene_to_go.keys())
        self.ic = self._compute_ic()

    def _compute_ic(self):
        """
        Compute Information Content using propagated annotation counts from the
        full GOA human corpus (NOT excluded, Biological Process only).

        For each GO term t: frequency(t) = genes annotated to t OR any descendant
        of t, divided by total unique annotated genes.

        Propagation guarantees that ancestor terms always have frequency >= their
        descendants, so IC(ancestor) <= IC(child). This keeps the MICA IC below
        both child-term ICs, ensuring Lin similarity stays in [0, 1].
        """
        gaf_path = os.path.join(GO_DIR, "goa_human.gaf")
        gene_direct_terms = {}
        all_genes = set()

        with open(gaf_path, 'r') as f:
            for line in f:
                if line.startswith('!'):
                    continue
                parts = line.strip().split('\t')
                if len(parts) < 15:
                    continue
                if 'NOT' in parts[3]:
                    continue
                if parts[8] != 'P':
                    continue
                gene = parts[2]
                go_id = parts[4]
                all_genes.add(gene)
                gene_direct_terms.setdefault(gene, set()).add(go_id)

        # Propagate each annotation up to all ancestors (memoized).
        _anc_cache = {}
        def get_ancestors_cached(tid):
            if tid in _anc_cache:
                return _anc_cache[tid]
            result = {tid}
            if tid in self.godag:
                for parent in self.godag[tid].parents:
                    result.update(get_ancestors_cached(parent.id))
            _anc_cache[tid] = result
            return result

        term_gene_count = {}
        for gene, terms in gene_direct_terms.items():
            covered = set()
            for t in terms:
                covered.update(get_ancestors_cached(t))
            for t in covered:
                term_gene_count[t] = term_gene_count.get(t, 0) + 1

        total = len(all_genes)
        ic = {}
        for go_id, count in term_gene_count.items():
            ic[go_id] = -np.log(count / total)

        return ic

    def term_similarity_lin(self, term1, term2):
        """Lin (1998) term-term similarity using MICA (most informative common ancestor)."""
        if term1 not in self.godag or term2 not in self.godag:
            return 0.0

        if term1 == term2:
            return 1.0

        ancestors1 = self._get_ancestors(term1)
        ancestors2 = self._get_ancestors(term2)
        common = ancestors1 & ancestors2

        if not common:
            return 0.0

        mica = max(common, key=lambda t: self.ic.get(t, 0))
        ic_mica = self.ic.get(mica, 0)
        ic1 = self.ic.get(term1, 0)
        ic2 = self.ic.get(term2, 0)

        if ic1 + ic2 == 0:
            return 0.0

        return (2 * ic_mica) / (ic1 + ic2)

    def _get_ancestors(self, term_id):
        ancestors = set()
        if term_id not in self.godag:
            return ancestors

        term = self.godag[term_id]
        ancestors.add(term_id)

        for parent in term.parents:
            ancestors.add(parent.id)
            ancestors.update(self._get_ancestors(parent.id))

        return ancestors

    def gene_similarity(self, gene1, gene2):
        """
        Symmetric Best-Match Average (BMA):
        For each term in gene1, find max Lin similarity to any term in gene2, then average.
        Repeat in reverse. Return mean of both directions.
        """
        if gene1 not in self.gene_to_go or gene2 not in self.gene_to_go:
            return 0.0

        terms1 = self.gene_to_go[gene1]
        terms2 = self.gene_to_go[gene2]

        scores_1to2 = [
            max([self.term_similarity_lin(t1, t2) for t2 in terms2], default=0)
            for t1 in terms1
        ]
        scores_2to1 = [
            max([self.term_similarity_lin(t1, t2) for t1 in terms1], default=0)
            for t2 in terms2
        ]

        if not scores_1to2 or not scores_2to1:
            return 0.0

        return (np.mean(scores_1to2) + np.mean(scores_2to1)) / 2

    def compute_similarity_matrix(self, genes=None, sample_size=None):
        if genes is None:
            genes = self.genes

        if sample_size and len(genes) > sample_size:
            genes = genes[:sample_size]

        n = len(genes)
        matrix = np.zeros((n, n))

        total_pairs = n * (n - 1) // 2
        computed = 0

        for i in range(n):
            matrix[i, i] = 1.0
            for j in range(i + 1, n):
                sim = self.gene_similarity(genes[i], genes[j])
                matrix[i, j] = sim
                matrix[j, i] = sim

                computed += 1
                if computed % 1000 == 0:
                    print(f"  Progress: {computed}/{total_pairs} pairs ({100*computed/total_pairs:.1f}%)")

        return matrix, genes


def main():
    print("Initializing semantic similarity engine...")
    ss = SemanticSimilarity()
    print(f"Loaded {len(ss.genes)} genes with GO annotations")

    print("\nComputing similarity matrix on all annotated genes...")
    matrix, genes = ss.compute_similarity_matrix()

    np.save(os.path.join(PROCESSED_DIR, "go_similarity_matrix.npy"), matrix)
    with open(os.path.join(PROCESSED_DIR, "similarity_genes.json"), 'w') as f:
        json.dump(genes, f)

    print(f"\nSimilarity matrix shape: {matrix.shape}")
    print(f"Mean similarity: {matrix.mean():.3f}")
    print(f"Max similarity (non-diagonal): {matrix[~np.eye(len(matrix), dtype=bool)].max():.3f}")

    return matrix, genes


if __name__ == "__main__":
    main()

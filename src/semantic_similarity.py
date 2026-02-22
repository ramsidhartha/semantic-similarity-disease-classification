import os
import json
import numpy as np
from goatools.obo_parser import GODag
from goatools.semantic import semantic_similarity, lin_sim, resnik_sim

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
GO_DIR = os.path.join(BASE_DIR, "data", "go")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")


class SemanticSimilarity:
    def __init__(self):
        self.godag = GODag(os.path.join(GO_DIR, "go-basic.obo"))
        
        with open(os.path.join(PROCESSED_DIR, "gene_go_mapping.json"), 'r') as f:
            self.gene_to_go = json.load(f)
        
        self.genes = list(self.gene_to_go.keys())
        self.ic = self._compute_ic()
    
    def _compute_ic(self):
        term_counts = {}
        total = len(self.gene_to_go)
        
        for terms in self.gene_to_go.values():
            for term in terms:
                term_counts[term] = term_counts.get(term, 0) + 1
        
        ic = {}
        for term, count in term_counts.items():
            freq = count / total
            ic[term] = -np.log(freq)
        
        return ic
    
    def term_similarity_lin(self, term1, term2):
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
    
    def term_similarity_wang(self, term1, term2):
        if term1 not in self.godag or term2 not in self.godag:
            return 0.0
        
        if term1 == term2:
            return 1.0
        
        sv1 = self._s_values(term1)
        sv2 = self._s_values(term2)
        
        common = set(sv1.keys()) & set(sv2.keys())
        if not common:
            return 0.0
        
        numerator = sum(sv1[t] + sv2[t] for t in common)
        denominator = sum(sv1.values()) + sum(sv2.values())
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    
    def _s_values(self, term_id, weight_is_a=0.8, weight_part_of=0.6):
        s_values = {term_id: 1.0}
        queue = [(term_id, 1.0)]
        
        while queue:
            current, current_s = queue.pop(0)
            if current not in self.godag:
                continue
            term = self.godag[current]
            for parent in term.parents:
                w = weight_is_a
                new_s = current_s * w
                if parent.id not in s_values or s_values[parent.id] < new_s:
                    s_values[parent.id] = new_s
                    queue.append((parent.id, new_s))
        
        return s_values
    
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
        if gene1 not in self.gene_to_go or gene2 not in self.gene_to_go:
            return 0.0
        
        terms1 = self.gene_to_go[gene1]
        terms2 = self.gene_to_go[gene2]
        
        scores_1to2 = []
        for t1 in terms1:
            best = max([self.term_similarity_lin(t1, t2) for t2 in terms2], default=0)
            scores_1to2.append(best)
        
        scores_2to1 = []
        for t2 in terms2:
            best = max([self.term_similarity_lin(t1, t2) for t1 in terms1], default=0)
            scores_2to1.append(best)
        
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

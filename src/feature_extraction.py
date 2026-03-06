import os
import json
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import SpectralClustering

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")


class FeatureExtractor:
    def __init__(self, n_pca_components=20, n_modules=9):
        self.n_pca_components = n_pca_components
        self.n_modules = n_modules
        self.pca = None
        self.scaler_expr = StandardScaler()
        self.scaler_go = StandardScaler()
        self.gene_modules = None
        self.luad_centroid = None
        self.lusc_centroid = None
        
        with open(os.path.join(PROCESSED_DIR, "gene_go_mapping.json"), 'r') as f:
            self.gene_to_go = json.load(f)
        
        self.go_matrix = np.load(os.path.join(PROCESSED_DIR, "go_similarity_matrix.npy"))
        self.go_genes = sorted(self.gene_to_go.keys())
    
    def _cluster_genes_by_go(self):
        if len(self.go_genes) < self.n_modules:
            return {i: [g] for i, g in enumerate(self.go_genes)}
        
        similarity_for_clustering = np.clip(self.go_matrix, 0, 1)
        np.fill_diagonal(similarity_for_clustering, 1)
        
        clustering = SpectralClustering(
            n_clusters=self.n_modules,
            affinity='precomputed',
            random_state=42,
            assign_labels='kmeans'
        )
        labels = clustering.fit_predict(similarity_for_clustering)
        
        modules = {}
        for i, label in enumerate(labels):
            if label not in modules:
                modules[label] = []
            modules[label].append(self.go_genes[i])
        
        return modules
    
    def fit(self, X_train, y_train):
        self.pca = PCA(n_components=min(self.n_pca_components, X_train.shape[1]))
        self.pca.fit(X_train)
        
        self.gene_modules = self._cluster_genes_by_go()
        
        luad_mask = y_train == 'LUAD'
        lusc_mask = y_train == 'LUSC'
        
        go_genes_in_data = [g for g in self.go_genes if g in X_train.columns]
        
        if go_genes_in_data:
            self.luad_centroid = X_train.loc[luad_mask, go_genes_in_data].mean()
            self.lusc_centroid = X_train.loc[lusc_mask, go_genes_in_data].mean()
        
        return self
    
    def extract_expression_features(self, X):
        pca_features = self.pca.transform(X)
        top_var_genes = X.var().nlargest(30).index.tolist()
        top_expr = X[top_var_genes].values
        
        expr_features = np.hstack([pca_features, top_expr])
        return expr_features
    
    def extract_go_features(self, X):
        features = []
        
        for idx in range(len(X)):
            sample = X.iloc[idx]
            sample_features = []
            
            for module_id, genes in self.gene_modules.items():
                genes_in_data = [g for g in genes if g in X.columns]
                if genes_in_data:
                    activity = sample[genes_in_data].mean()
                else:
                    activity = 0
                sample_features.append(activity)
            
            go_genes_in_data = [g for g in self.go_genes if g in X.columns]
            if go_genes_in_data and self.luad_centroid is not None:
                sample_go = sample[go_genes_in_data]
                sim_luad = np.corrcoef(sample_go, self.luad_centroid[go_genes_in_data])[0, 1]
                sim_lusc = np.corrcoef(sample_go, self.lusc_centroid[go_genes_in_data])[0, 1]
                sample_features.extend([sim_luad, sim_lusc, sim_luad - sim_lusc])
            else:
                sample_features.extend([0, 0, 0])
            
            features.append(sample_features)
        
        return np.array(features)
    
    def extract_all_features(self, X, y=None):
        expr_features = self.extract_expression_features(X)
        go_features = self.extract_go_features(X)
        
        combined = np.hstack([expr_features, go_features])
        
        return {
            'expression': expr_features,
            'go': go_features,
            'combined': combined
        }


def main():
    print("Loading data splits...")
    train = pd.read_csv(os.path.join(PROCESSED_DIR, "train.csv"), index_col=0)
    val = pd.read_csv(os.path.join(PROCESSED_DIR, "val.csv"), index_col=0)
    
    y_train = train['cancer_type']
    X_train = train.drop(columns=['cancer_type'])
    y_val = val['cancer_type']
    X_val = val.drop(columns=['cancer_type'])
    
    print("Fitting feature extractor...")
    extractor = FeatureExtractor()
    extractor.fit(X_train, y_train)
    
    print("Extracting features...")
    train_features = extractor.extract_all_features(X_train)
    val_features = extractor.extract_all_features(X_val)
    
    print(f"\nFeature dimensions:")
    print(f"  Expression: {train_features['expression'].shape[1]}")
    print(f"  GO: {train_features['go'].shape[1]}")
    print(f"  Combined: {train_features['combined'].shape[1]}")
    
    np.save(os.path.join(PROCESSED_DIR, "train_features_expr.npy"), train_features['expression'])
    np.save(os.path.join(PROCESSED_DIR, "train_features_go.npy"), train_features['go'])
    np.save(os.path.join(PROCESSED_DIR, "train_features_combined.npy"), train_features['combined'])
    np.save(os.path.join(PROCESSED_DIR, "val_features_expr.npy"), val_features['expression'])
    np.save(os.path.join(PROCESSED_DIR, "val_features_go.npy"), val_features['go'])
    np.save(os.path.join(PROCESSED_DIR, "val_features_combined.npy"), val_features['combined'])
    
    return extractor


if __name__ == "__main__":
    main()

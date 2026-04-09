import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import networkx as nx

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")


def plot_deg_volcano(deg_df, output_path):
    fig, ax = plt.subplots(figsize=(10, 8))
    
    deg_df['neg_log_pval'] = -np.log10(deg_df['pvalue_adj'] + 1e-300)
    
    significant = (deg_df['abs_log2fc'] >= 1.0) & (deg_df['pvalue_adj'] < 0.01)
    
    ax.scatter(
        deg_df.loc[~significant, 'log2fc'],
        deg_df.loc[~significant, 'neg_log_pval'],
        c='gray', alpha=0.5, s=10, label='Not significant'
    )
    
    up_luad = significant & (deg_df['log2fc'] > 0)
    ax.scatter(
        deg_df.loc[up_luad, 'log2fc'],
        deg_df.loc[up_luad, 'neg_log_pval'],
        c='red', alpha=0.7, s=20, label='Upregulated in LUAD'
    )
    
    up_lusc = significant & (deg_df['log2fc'] < 0)
    ax.scatter(
        deg_df.loc[up_lusc, 'log2fc'],
        deg_df.loc[up_lusc, 'neg_log_pval'],
        c='blue', alpha=0.7, s=20, label='Upregulated in LUSC'
    )
    
    ax.axhline(y=-np.log10(0.01), linestyle='--', color='black', linewidth=0.5)
    ax.axvline(x=1.0, linestyle='--', color='black', linewidth=0.5)
    ax.axvline(x=-1.0, linestyle='--', color='black', linewidth=0.5)
    
    ax.set_xlabel('Log2 Fold Change (LUAD vs LUSC)', fontsize=12)
    ax.set_ylabel('-Log10 Adjusted P-value', fontsize=12)
    ax.set_title('Differential Expression Analysis', fontsize=14)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_go_similarity_heatmap(matrix, genes, output_path, n_genes=50):
    if len(genes) > n_genes:
        matrix = matrix[:n_genes, :n_genes]
        genes = genes[:n_genes]
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    sns.heatmap(
        matrix, 
        cmap='RdYlBu_r',
        xticklabels=genes,
        yticklabels=genes,
        ax=ax
    )
    
    ax.set_title('GO Semantic Similarity Matrix', fontsize=14)
    plt.xticks(rotation=90, fontsize=6)
    plt.yticks(rotation=0, fontsize=6)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_adjacency_matrix(matrix, genes, output_path, threshold=0.5):
    """Binary adjacency matrix: black = similarity > threshold, white = no edge."""
    n = len(genes)
    adj = (matrix > threshold).astype(float)
    np.fill_diagonal(adj, 0)  # hide diagonal (self-similarity)

    fig, ax = plt.subplots(figsize=(14, 12))
    ax.imshow(1 - adj, cmap='gray', vmin=0, vmax=1, interpolation='nearest')

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(genes, rotation=90, fontsize=5)
    ax.set_yticklabels(genes, fontsize=5)

    ax.set_title(
        f'GO Functional Similarity Adjacency Matrix\n'
        f'(black = Lin similarity > {threshold}, {n} genes)',
        fontsize=12
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_go_network(matrix, genes, output_path, threshold=0.5, n_genes=30):
    if len(genes) > n_genes:
        matrix = matrix[:n_genes, :n_genes]
        genes = genes[:n_genes]
    
    G = nx.Graph()
    
    for i, gene in enumerate(genes):
        G.add_node(gene)
    
    for i in range(len(genes)):
        for j in range(i + 1, len(genes)):
            if matrix[i, j] > threshold:
                G.add_edge(genes[i], genes[j], weight=matrix[i, j])
    
    fig, ax = plt.subplots(figsize=(14, 12))
    
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    degrees = dict(G.degree())
    node_sizes = [100 + degrees[node] * 50 for node in G.nodes()]
    
    edge_weights = [G[u][v]['weight'] * 2 for u, v in G.edges()]
    
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='lightblue', ax=ax)
    nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.5, edge_color='gray', ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)
    
    ax.set_title(f'GO Similarity Network (threshold > {threshold})', fontsize=14)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_model_comparison(results, output_path):
    models = list(results.keys())
    accuracies = [results[m]['accuracy'] for m in models]
    f1_scores = [results[m]['f1'] for m in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars1 = ax.bar(x - width/2, accuracies, width, label='Accuracy', color='steelblue')
    bars2 = ax.bar(x + width/2, f1_scores, width, label='F1 Score', color='coral')
    
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Comparison', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_', ' ').title() for m in models], fontsize=10)
    ax.legend()
    ax.set_ylim(0, 1.1)
    
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)
    
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_feature_importance(importance, feature_names, output_path, top_n=15):
    indices = np.argsort(importance)[::-1][:top_n]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.barh(range(top_n), importance[indices][::-1], color='steelblue')
    ax.set_yticks(range(top_n))
    
    if feature_names is not None and len(feature_names) > max(indices):
        labels = [feature_names[i] for i in indices[::-1]]
    else:
        labels = [f'Feature {i}' for i in indices[::-1]]
    
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_title('Top Feature Importance', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_roc_curves(results, output_path):
    fig, ax = plt.subplots(figsize=(8, 8))
    
    colors = {
        'expression_only': 'steelblue',
        'go_only': 'coral',
        'combined': 'mediumseagreen',
        'stacked': 'mediumpurple'
    }
    
    # model_comparison.json has a "validation" root key we need to unnest
    plot_data = results.get('validation', results)
    
    for model_name, data in plot_data.items():
        if 'roc_curve' in data:
            fpr = data['roc_curve']['fpr']
            tpr = data['roc_curve']['tpr']
            auc = data.get('auc_roc', 0)
            
            label = f"{model_name.replace('_', ' ').title()} (AUC = {auc:.3f})"
            color = colors.get(model_name, 'gray')
            
            ax.plot(fpr, tpr, label=label, color=color, linewidth=2)
            
    # Plot diagonal (random chance)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Chance')
    
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves by Model', fontsize=14)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_confusion_matrices(results, output_path):
    import seaborn as sns
    
    plot_data = results.get('validation', results)
    
    # Filter only models that have a confusion matrix
    models = [m for m, d in plot_data.items() if 'confusion_matrix' in d]
    
    if not models:
        print("No confusion matrix data found.")
        return
        
    n_models = len(models)
    cols = 2
    rows = int(np.ceil(n_models / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(12, 5 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
        
    labels = ['LUAD', 'LUSC']
    
    for i, model_name in enumerate(models):
        ax = axes[i]
        cm = np.array(plot_data[model_name]['confusion_matrix'])
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                    xticklabels=labels, yticklabels=labels)
        
        ax.set_title(model_name.replace('_', ' ').title(), fontsize=14)
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_xlabel('Predicted Label', fontsize=12)
        
    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
        
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def generate_all_visualizations():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    
    print("Generating visualizations...")
    
    deg_path = os.path.join(PROCESSED_DIR, "deg_analysis.csv")
    if os.path.exists(deg_path):
        deg_df = pd.read_csv(deg_path)
        plot_deg_volcano(deg_df, os.path.join(FIGURES_DIR, "volcano_plot.png"))
        print("  Created volcano_plot.png")
    
    matrix_path = os.path.join(PROCESSED_DIR, "go_similarity_matrix.npy")
    genes_path = os.path.join(PROCESSED_DIR, "similarity_genes.json")
    if os.path.exists(matrix_path) and os.path.exists(genes_path):
        matrix = np.load(matrix_path)
        with open(genes_path, 'r') as f:
            genes = json.load(f)
        
        plot_go_similarity_heatmap(matrix, genes, os.path.join(FIGURES_DIR, "go_heatmap.png"))
        print("  Created go_heatmap.png")
        
    
    full_results_path = os.path.join(RESULTS_DIR, "model_comparison.json")
    if os.path.exists(full_results_path):
        with open(full_results_path, 'r') as f:
            full_results = json.load(f)
        plot_roc_curves(full_results, os.path.join(FIGURES_DIR, "roc_curves.png"))
        print("  Created roc_curves.png")
        
        plot_confusion_matrices(full_results, os.path.join(FIGURES_DIR, "confusion_matrices.png"))
        print("  Created confusion_matrices.png")
    
    print("\nAll visualizations complete.")


if __name__ == "__main__":
    generate_all_visualizations()

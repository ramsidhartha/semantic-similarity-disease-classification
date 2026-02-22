import os
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
RESULTS_DIR = os.path.join(BASE_DIR, "results")


class DualSpaceClassifier:
    def __init__(self):
        self.expr_clf = RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42
        )
        self.go_clf = RandomForestClassifier(
            n_estimators=100, max_depth=8, random_state=42
        )
        self.combined_clf = GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42
        )
        self.le = LabelEncoder()
    
    def fit(self, X_expr, X_go, X_combined, y):
        y_encoded = self.le.fit_transform(y)
        self.expr_clf.fit(X_expr, y_encoded)
        self.go_clf.fit(X_go, y_encoded)
        self.combined_clf.fit(X_combined, y_encoded)
        return self
    
    def predict(self, X_combined):
        return self.le.inverse_transform(self.combined_clf.predict(X_combined))
    
    def evaluate(self, X_expr, X_go, X_combined, y):
        y_pred_expr = self.le.inverse_transform(self.expr_clf.predict(X_expr))
        y_pred_go = self.le.inverse_transform(self.go_clf.predict(X_go))
        y_pred_combined = self.le.inverse_transform(self.combined_clf.predict(X_combined))
        
        results = {}
        for name, y_pred in [
            ('expression_only', y_pred_expr),
            ('go_only', y_pred_go),
            ('combined', y_pred_combined)
        ]:
            results[name] = {
                'accuracy': float(accuracy_score(y, y_pred)),
                'f1': float(f1_score(y, y_pred, pos_label='LUAD')),
                'auc_roc': float(roc_auc_score(
                    (np.array(y) == 'LUAD').astype(int),
                    (np.array(y_pred) == 'LUAD').astype(int)
                )),
                'confusion_matrix': confusion_matrix(y, y_pred).tolist()
            }
        
        return results
    
    def get_feature_importance(self):
        return {
            'expression': self.expr_clf.feature_importances_,
            'go': self.go_clf.feature_importances_,
            'combined': self.combined_clf.feature_importances_
        }


def cross_validate(X_expr, X_go, X_combined, y, n_folds=5):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    fold_results = {
        'expression_only': [], 'go_only': [], 'combined': []
    }
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_expr, y)):
        clf = DualSpaceClassifier()
        clf.fit(X_expr[train_idx], X_go[train_idx], X_combined[train_idx], y[train_idx])
        results = clf.evaluate(
            X_expr[val_idx], X_go[val_idx], X_combined[val_idx], y[val_idx]
        )
        
        for model in fold_results:
            fold_results[model].append(results[model]['accuracy'])
        
        print(f"  Fold {fold+1}: Expr={results['expression_only']['accuracy']:.3f}  "
              f"GO={results['go_only']['accuracy']:.3f}  "
              f"Combined={results['combined']['accuracy']:.3f}")
    
    cv_summary = {}
    for model, scores in fold_results.items():
        cv_summary[model] = {
            'mean_accuracy': float(np.mean(scores)),
            'std_accuracy': float(np.std(scores)),
            'fold_scores': [float(s) for s in scores]
        }
    
    return cv_summary


def case_study_analysis(classifier, X_expr, X_go, X_combined, y, sample_ids):
    y_pred = classifier.predict(X_combined)
    
    misclassified = np.where(np.array(y) != np.array(y_pred))[0]
    correct = np.where(np.array(y) == np.array(y_pred))[0]
    
    cases = []
    
    for idx in misclassified[:5]:
        go_prob = classifier.go_clf.predict_proba(X_go[idx:idx+1])[0]
        expr_prob = classifier.expr_clf.predict_proba(X_expr[idx:idx+1])[0]
        cases.append({
            'sample_id': str(sample_ids[idx]) if sample_ids is not None else str(idx),
            'true_label': str(y[idx]),
            'predicted': str(y_pred[idx]),
            'type': 'misclassified',
            'go_confidence': float(max(go_prob)),
            'expr_confidence': float(max(expr_prob)),
            'go_prediction': classifier.le.inverse_transform([np.argmax(go_prob)])[0],
            'expr_prediction': classifier.le.inverse_transform([np.argmax(expr_prob)])[0]
        })
    
    high_conf_idx = correct[:3]
    for idx in high_conf_idx:
        go_prob = classifier.go_clf.predict_proba(X_go[idx:idx+1])[0]
        expr_prob = classifier.expr_clf.predict_proba(X_expr[idx:idx+1])[0]
        cases.append({
            'sample_id': str(sample_ids[idx]) if sample_ids is not None else str(idx),
            'true_label': str(y[idx]),
            'predicted': str(y_pred[idx]),
            'type': 'correct_high_confidence',
            'go_confidence': float(max(go_prob)),
            'expr_confidence': float(max(expr_prob)),
            'go_prediction': classifier.le.inverse_transform([np.argmax(go_prob)])[0],
            'expr_prediction': classifier.le.inverse_transform([np.argmax(expr_prob)])[0]
        })
    
    return {
        'total_samples': len(y),
        'misclassified_count': len(misclassified),
        'cases': cases
    }


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    print("Loading features...")
    train_expr = np.load(os.path.join(PROCESSED_DIR, "train_features_expr.npy"))
    train_go = np.load(os.path.join(PROCESSED_DIR, "train_features_go.npy"))
    train_combined = np.load(os.path.join(PROCESSED_DIR, "train_features_combined.npy"))
    
    val_expr = np.load(os.path.join(PROCESSED_DIR, "val_features_expr.npy"))
    val_go = np.load(os.path.join(PROCESSED_DIR, "val_features_go.npy"))
    val_combined = np.load(os.path.join(PROCESSED_DIR, "val_features_combined.npy"))
    
    train_df = pd.read_csv(os.path.join(PROCESSED_DIR, "train.csv"), index_col=0)
    val_df = pd.read_csv(os.path.join(PROCESSED_DIR, "val.csv"), index_col=0)
    
    y_train = train_df['cancer_type'].values
    y_val = val_df['cancer_type'].values
    
    print("Training dual-space classifier...")
    classifier = DualSpaceClassifier()
    classifier.fit(train_expr, train_go, train_combined, y_train)
    
    print("\nEvaluating on validation set...")
    results = classifier.evaluate(val_expr, val_go, val_combined, y_val)
    
    print("\n" + "="*50)
    print("RESULTS COMPARISON")
    print("="*50)
    for model, metrics in results.items():
        print(f"\n{model.upper()}")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  F1 Score:  {metrics['f1']:.4f}")
        print(f"  AUC-ROC:   {metrics['auc_roc']:.4f}")
        cm = metrics['confusion_matrix']
        print(f"  Confusion: [[{cm[0][0]}, {cm[0][1]}], [{cm[1][0]}, {cm[1][1]}]]")
    
    print("\n5-Fold Cross-Validation...")
    all_expr = np.vstack([train_expr, val_expr])
    all_go = np.vstack([train_go, val_go])
    all_combined = np.vstack([train_combined, val_combined])
    all_y = np.concatenate([y_train, y_val])
    
    cv_results = cross_validate(all_expr, all_go, all_combined, all_y)
    
    print(f"\nCV Results (mean +/- std):")
    for model, cv in cv_results.items():
        print(f"  {model}: {cv['mean_accuracy']:.3f} +/- {cv['std_accuracy']:.3f}")
    
    print("\nCase Study Analysis...")
    case_results = case_study_analysis(
        classifier, val_expr, val_go, val_combined, y_val,
        sample_ids=val_df.index.tolist()
    )
    print(f"  Misclassified: {case_results['misclassified_count']}/{case_results['total_samples']}")
    for case in case_results['cases']:
        print(f"  [{case['type']}] {case['sample_id']}: "
              f"true={case['true_label']} pred={case['predicted']} "
              f"GO_conf={case['go_confidence']:.2f} Expr_conf={case['expr_confidence']:.2f}")
    
    all_results = {
        'validation': results,
        'cross_validation': cv_results,
        'case_studies': case_results
    }
    
    with open(os.path.join(RESULTS_DIR, "model_comparison.json"), 'w') as f:
        json.dump(all_results, f, indent=2)
    
    simple_results = {k: {'accuracy': v['accuracy'], 'f1': v['f1']} for k, v in results.items()}
    with open(os.path.join(RESULTS_DIR, "model_comparison_simple.json"), 'w') as f:
        json.dump(simple_results, f, indent=2)
    
    return classifier, all_results


if __name__ == "__main__":
    main()

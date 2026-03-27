import os
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

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
        # Meta-learner: LR wrapped in a pipeline with StandardScaler
        # (probabilities are 0-1 but scaling is good practice for LR)
        self.meta_clf = Pipeline([
            ('scaler', StandardScaler()),
            ('lr', LogisticRegression(random_state=42))
        ])
        self.le = LabelEncoder()
        self._stacking_fitted = False
    
    def fit(self, X_expr, X_go, X_combined, y):
        y_encoded = self.le.fit_transform(y)
        self.expr_clf.fit(X_expr, y_encoded)
        self.go_clf.fit(X_go, y_encoded)
        self.combined_clf.fit(X_combined, y_encoded)
        
        # Stacking: generate out-of-fold predictions to train meta-learner
        self._fit_stacking(X_expr, X_go, X_combined, y_encoded)
        return self
    
    def _fit_stacking(self, X_expr, X_go, X_combined, y_encoded):
        """Train meta-learner using out-of-fold predictions to avoid data leakage.
        Uses all three base models: expression, GO, and combined."""
        n_samples = len(y_encoded)
        meta_features = np.zeros((n_samples, 3))  # 3 branches now
        
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # Safe class index: resolve once from LabelEncoder
        luad_idx = list(self.le.classes_).index('LUAD')
        
        for train_idx, val_idx in skf.split(X_expr, y_encoded):
            expr_fold = RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=42
            )
            go_fold = RandomForestClassifier(
                n_estimators=100, max_depth=8, random_state=42
            )
            combined_fold = GradientBoostingClassifier(
                n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42
            )
            
            expr_fold.fit(X_expr[train_idx], y_encoded[train_idx])
            go_fold.fit(X_go[train_idx], y_encoded[train_idx])
            combined_fold.fit(X_combined[train_idx], y_encoded[train_idx])
            
            meta_features[val_idx, 0] = expr_fold.predict_proba(X_expr[val_idx])[:, luad_idx]
            meta_features[val_idx, 1] = go_fold.predict_proba(X_go[val_idx])[:, luad_idx]
            meta_features[val_idx, 2] = combined_fold.predict_proba(X_combined[val_idx])[:, luad_idx]
        
        self.meta_clf.fit(meta_features, y_encoded)
        self._stacking_fitted = True
    
    def _predict_stacked(self, X_expr, X_go, X_combined):
        """Generate stacked prediction using all three base model probabilities."""
        luad_idx = list(self.le.classes_).index('LUAD')
        meta_features = np.column_stack([
            self.expr_clf.predict_proba(X_expr)[:, luad_idx],
            self.go_clf.predict_proba(X_go)[:, luad_idx],
            self.combined_clf.predict_proba(X_combined)[:, luad_idx]
        ])
        return self.le.inverse_transform(self.meta_clf.predict(meta_features))
    
    def _predict_stacked_proba(self, X_expr, X_go, X_combined):
        """Generate stacked prediction probabilities from all three branches."""
        luad_idx = list(self.le.classes_).index('LUAD')
        meta_features = np.column_stack([
            self.expr_clf.predict_proba(X_expr)[:, luad_idx],
            self.go_clf.predict_proba(X_go)[:, luad_idx],
            self.combined_clf.predict_proba(X_combined)[:, luad_idx]
        ])
        return self.meta_clf.predict_proba(meta_features)
    
    def predict(self, X_combined):
        return self.le.inverse_transform(self.combined_clf.predict(X_combined))
    
    def evaluate(self, X_expr, X_go, X_combined, y):
        luad_idx = list(self.le.classes_).index('LUAD')
        y_true_binary = (np.array(y) == 'LUAD').astype(int)
        
        y_pred_expr = self.le.inverse_transform(self.expr_clf.predict(X_expr))
        y_pred_go = self.le.inverse_transform(self.go_clf.predict(X_go))
        y_pred_combined = self.le.inverse_transform(self.combined_clf.predict(X_combined))
        y_pred_stacked = self._predict_stacked(X_expr, X_go, X_combined)
        
        # Probabilities for proper AUC-ROC (not hard labels)
        y_prob_expr = self.expr_clf.predict_proba(X_expr)[:, luad_idx]
        y_prob_go = self.go_clf.predict_proba(X_go)[:, luad_idx]
        y_prob_combined = self.combined_clf.predict_proba(X_combined)[:, luad_idx]
        y_prob_stacked = self._predict_stacked_proba(X_expr, X_go, X_combined)[:, luad_idx]
        
        results = {}
        for name, y_pred, y_prob in [
            ('expression_only', y_pred_expr, y_prob_expr),
            ('go_only', y_pred_go, y_prob_go),
            ('combined', y_pred_combined, y_prob_combined),
            ('stacked', y_pred_stacked, y_prob_stacked)
        ]:
            results[name] = {
                'accuracy': float(accuracy_score(y, y_pred)),
                'f1': float(f1_score(y, y_pred, pos_label='LUAD')),
                'auc_roc': float(roc_auc_score(y_true_binary, y_prob)),
                'confusion_matrix': confusion_matrix(y, y_pred).tolist()
            }
        
        # Show stacking weights (access LR inside pipeline)
        if self._stacking_fitted:
            lr = self.meta_clf.named_steps['lr']
            coefs = lr.coef_[0]
            results['stacking_weights'] = {
                'expression_weight': float(coefs[0]),
                'go_weight': float(coefs[1]),
                'combined_weight': float(coefs[2]),
                'intercept': float(lr.intercept_[0])
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
        'expression_only': [], 'go_only': [], 'combined': [], 'stacked': []
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
              f"Combined={results['combined']['accuracy']:.3f}  "
              f"Stacked={results['stacked']['accuracy']:.3f}")
    
    cv_summary = {}
    for model, scores in fold_results.items():
        cv_summary[model] = {
            'mean_accuracy': float(np.mean(scores)),
            'std_accuracy': float(np.std(scores)),
            'fold_scores': [float(s) for s in scores]
        }
    
    return cv_summary


def case_study_analysis(classifier, X_expr, X_go, X_combined, y, sample_ids):
    # Use stacked prediction for case studies (all 3 branches)
    y_pred = classifier._predict_stacked(X_expr, X_go, X_combined)
    
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
    
    print("Training dual-space classifier (with stacking)...")
    classifier = DualSpaceClassifier()
    classifier.fit(train_expr, train_go, train_combined, y_train)
    
    print("\nEvaluating on validation set...")
    results = classifier.evaluate(val_expr, val_go, val_combined, y_val)
    
    print("\n" + "="*60)
    print("RESULTS COMPARISON")
    print("="*60)
    for model, metrics in results.items():
        if model == 'stacking_weights':
            continue
        print(f"\n{model.upper()}")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  F1 Score:  {metrics['f1']:.4f}")
        print(f"  AUC-ROC:   {metrics['auc_roc']:.4f}")
        cm = metrics['confusion_matrix']
        print(f"  Confusion: [[{cm[0][0]}, {cm[0][1]}], [{cm[1][0]}, {cm[1][1]}]]")
    
    if 'stacking_weights' in results:
        w = results['stacking_weights']
        print(f"\nSTACKING WEIGHTS:")
        print(f"  Expression branch: {w['expression_weight']:.3f}")
        print(f"  GO branch:         {w['go_weight']:.3f}")
        print(f"  Intercept:         {w['intercept']:.3f}")
    
    print("\n5-Fold Cross-Validation...")
    all_expr = np.vstack([train_expr, val_expr])
    all_go = np.vstack([train_go, val_go])
    all_combined = np.vstack([train_combined, val_combined])
    all_y = np.concatenate([y_train, y_val])
    
    cv_results = cross_validate(all_expr, all_go, all_combined, all_y)
    
    print(f"\nCV Results (mean +/- std):")
    for model, cv in cv_results.items():
        print(f"  {model}: {cv['mean_accuracy']:.3f} +/- {cv['std_accuracy']:.3f}")
    
    print("\nCase Study Analysis (using stacked model)...")
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
    
    simple_results = {}
    for k, v in results.items():
        if k != 'stacking_weights':
            simple_results[k] = {'accuracy': v['accuracy'], 'f1': v['f1']}
    with open(os.path.join(RESULTS_DIR, "model_comparison_simple.json"), 'w') as f:
        json.dump(simple_results, f, indent=2)
    
    return classifier, all_results


if __name__ == "__main__":
    main()

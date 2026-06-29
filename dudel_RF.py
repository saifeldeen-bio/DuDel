
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="joblib")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix, PrecisionRecallDisplay,
                             precision_recall_fscore_support, average_precision_score)
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold, learning_curve
from imblearn.over_sampling import SMOTE
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import BaseEstimator, TransformerMixin

# --- FIXED GAUSSIAN NOISE TRANSFORMER ---
class GaussianNoise(BaseEstimator, TransformerMixin):
    def __init__(self, sigma=0.05, random_state=None):
        self.sigma = sigma
        self.random_state = random_state
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        # Crucial for scikit-learn validation: Do NOT add noise during validation/testing
        return X
        
    def fit_transform(self, X, y=None):
        rng = np.random.default_rng(self.random_state)
        noise = rng.normal(0, self.sigma, X.shape)
        # Convert to numpy array safely if it's a pandas DataFrame
        X_val = X.values if isinstance(X, pd.DataFrame) else X
        return X_val + noise

# --- 1. DATA LOADING & CLEANING ---
nmdData = pd.read_csv("/cbio/projects/003/saifeldeen/samples/newTrial/training_set/new/allNmds_F.csv")

# Extract features, targets, and group partitions
X = nmdData.drop(columns=['SV_call', "Gene", "Group", "X7_pct_gc", "X14_seq_len"])
y = nmdData["SV_call"].values
groups = nmdData["Group"].values
labels = ["DEL", "Normal", "DUP"]

majority_count = pd.Series(y).value_counts().max()
target_val = int(majority_count * 0.1)
pipe_sampling_strategy = {0: target_val, 2: target_val}

logo = LeaveOneGroupOut()
all_y_true, all_y_pred, all_y_probs = [], [], []
sigma = 0.05

print(f"Starting LOGO Cross-Validation on {nmdData['Group'].nunique()} groups...")

# --- 2. LEAVE-ONE-GROUP-OUT CROSS VALIDATION ---
for train_idx, test_idx in logo.split(X, y, groups=groups):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    fold_majority = pd.Series(y_train).value_counts().max()
    fold_target = int(fold_majority * 0.1)
    
    # Robust isolated pipeline per validation block
    fold_pipeline = ImbPipeline([
        ('smote', SMOTE(random_state=2, sampling_strategy={0: fold_target, 2: fold_target})),
        ('noise', GaussianNoise(sigma=sigma, random_state=9)), # Noise only runs on fit_transform
        ('classifier', BalancedRandomForestClassifier(
            n_estimators=500, criterion='gini', max_depth=None,
            min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
            max_features='sqrt', max_leaf_nodes=None, min_impurity_decrease=0.0,
            bootstrap=True, oob_score=False, n_jobs=-1,
            random_state=9, verbose=0, warm_start=False,
            class_weight={0: 5, 1: 1, 2: 5}
        ))
    ])
    
    # Train safely (Pipeline applies SMOTE and GaussianNoise sequentially to train data)
    fold_pipeline.fit(X_train, y_train)
    
    # Evaluate safely (Pipeline skips noise adjustments automatically during inference)
    y_pred = fold_pipeline.predict(X_test)
    y_probs = fold_pipeline.predict_proba(X_test)

    print(f'\n--- PERFORMANCE METRICS (LOGO) FOR FOLD ---')
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred)
    for i, label in enumerate(labels):
        print(f"{label} -> Precision: {prec[i]:.3f}, Recall: {rec[i]:.3f}, F1: {f1[i]:.3f}")
    
    all_y_true.extend(y_test)
    all_y_pred.extend(y_pred)
    all_y_probs.extend(y_probs)

all_y_true = np.array(all_y_true)
all_y_pred = np.array(all_y_pred)
all_y_probs = np.array(all_y_probs)

# --- 3. FINAL PRODUCTION MODEL ---
print("\nTraining Final Production Model...")
final_pipeline = ImbPipeline([
    ('smote', SMOTE(random_state=2, sampling_strategy=pipe_sampling_strategy)),
    ('noise', GaussianNoise(sigma=sigma, random_state=9)),
    ('classifier', BalancedRandomForestClassifier(
        n_estimators=500, criterion='gini', max_depth=None,
        min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
        max_features='sqrt', max_leaf_nodes=None, min_impurity_decrease=0.0,
        bootstrap=True, oob_score=False, n_jobs=-1,
        random_state=9, verbose=0, warm_start=False,
        class_weight={0: 5, 1: 1, 2: 5}
    ))
])
final_pipeline.fit(X, y)

# --- 4. METRICS & VISUALIZATIONS ---
print('\n--- HONEST PERFORMANCE METRICS (LOGO) ---')
prec, rec, f1, _ = precision_recall_fscore_support(all_y_true, all_y_pred, labels=[0, 2])
for i, label in enumerate(["DEL", "DUP"]):
    print(f"{label} -> Precision: {prec[i]:.3f}, Recall: {rec[i]:.3f}, F1: {f1[i]:.3f}")

# Filter out the 'Normal (1)' background class to evaluate actual CNV variants honestly
mask = np.isin(all_y_true, [0, 2])
y_true_filtered = all_y_true[mask]
y_pred_filtered = all_y_pred[mask]
y_probs_filtered = all_y_probs[mask]

# Confusion Matrix Plots
cm = confusion_matrix(y_true_filtered, y_pred_filtered, labels=[0, 1, 2], normalize='true')
cm_subset = cm[[0, 2], :]  # Rows: DEL, DUP / Columns: DEL, Normal, DUP

plt.figure(figsize=(7, 5))
sns.heatmap(cm_subset, annot=True, fmt=".2%", cmap="Blues",
            xticklabels=["DEL", "Normal", "DUP"],
            yticklabels=["DEL", "DUP"])
plt.title("Confusion Matrix: True DEL/DUP samples")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig("/cbio/projects/003/saifeldeen/samples/newTrial/models/FinlaFinal/confusion_matrix_final_DEL_DUP.png", dpi=300)
plt.close()

# Feature Importance extraction from Pipeline steps
final_classifier = final_pipeline.named_steps['classifier']
importances = final_classifier.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(12, 6))
plt.title("Feature Importance for CNV Detection")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), [X.columns[i] for i in indices], rotation=45)
plt.tight_layout()
plt.savefig("/cbio/projects/003/saifeldeen/samples/newTrial/models/FinlaFinal/feature_importance_final.png", dpi=300)
plt.close()

# --- 5. LEARNING CURVE ---
print("Generating Learning Curve...")
from sklearn.metrics import make_scorer, f1_score

def f1_del_dup(y_true, y_pred):
    return f1_score(y_true, y_pred, labels=[0, 2], average='macro')

custom_scorer = make_scorer(f1_del_dup)
cv_strat = StratifiedKFold(n_splits=5, shuffle=True, random_state=2)
sizes = np.linspace(0.1, 1.0, 5)

def save_shaded_plot(train_scores, test_scores, x_axis, title, xlabel, filename, color_train, color_test):
    train_mean, train_std = np.mean(train_scores, axis=1), np.std(train_scores, axis=1)
    test_mean, test_std = np.mean(test_scores, axis=1), np.std(test_scores, axis=1)
    plt.figure(figsize=(8, 6))
    plt.plot(x_axis, train_mean, 'o-', color=color_train, label="Training score", lw=2)
    plt.plot(x_axis, test_mean, 'o-', color=color_test, label="Cross-validation score", lw=2)
    plt.fill_between(x_axis, train_mean - train_std, train_mean + train_std, alpha=0.15, color=color_train)
    plt.fill_between(x_axis, test_mean - test_std, test_mean + test_std, alpha=0.15, color=color_test)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Macro F1 Score')
    plt.legend(loc="best")
    plt.grid(alpha=0.3)
    plt.ylim([0, 1.05])
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

l_sizes, l_train, l_test = learning_curve(
    final_pipeline, X, y, cv=cv_strat, n_jobs=1,
    train_sizes=sizes, scoring=custom_scorer
)
save_shaded_plot(l_train, l_test, l_sizes, "Learning Curve", "Number of Training Sets",
                 "/cbio/projects/003/saifeldeen/samples/newTrial/models/FinlaFinal/learning_curve_final.png",
                 "navy", "darkorange")

joblib.dump(fold_pipeline, "/cbio/projects/003/saifeldeen/samples/newTrial/models/FinlaFinal/nmdscan_fold_production_model_lzma_32.joblib", compress=("lzma", 9))
joblib.dump(final_pipeline, "/cbio/projects/003/saifeldeen/samples/newTrial/models/FinlaFinal/nmdscan_final_production_model_lzma_32.joblib", compress=("lzma", 9))

import pandas as pd
import numpy as np
import os
import joblib
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix, roc_curve, auc, f1_score, 
                             precision_recall_curve, precision_recall_fscore_support)
from sklearn.model_selection import LeaveOneGroupOut
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import label_binarize
from imblearn.ensemble import BalancedRandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, PrecisionRecallDisplay, average_precision_score
from sklearn import 

# --- DATA LOADING ---
# nmdDataDir = "/cbio/projects/003/saifeldeen/samples/newTrial/training_set/new"
# all_dfs = []
# for file in os.listdir(nmdDataDir):
#     df = pd.read_csv(os.path.join(nmdDataDir, file), sep=",", header=0)
#     all_dfs.append(df)
# nmdData = pd.concat(all_dfs, axis=0, ignore_index=True)
# mapping = {'DEL': 0, 'Normal': 1, 'DUP': 2}
# Calculate how much the 10 references vary among themselves
# nmdData['ref_std'] = nmdData[['Ref1', 'Ref2', 'Ref3', 'Ref4', 'Ref5', 
#                             'Ref6', 'Ref7', 'Ref8', 'Ref9', 'Ref10']].std(axis=1)
# nmdData['log2_ratio'] = np.log2((nmdData['counts'] + 0.1) / (nmdData['meanRef'] + 0.1))
# Z-score: How many standard deviations is our count away from the mean?
# nmdData['z_score'] = (nmdData['counts'] - nmdData['meanRef']) / (nmdData['ref_std'] + 0.1)
# nmdData["SV_call"] = nmdData["SV_call"].map(mapping)
# nmdData = nmdData.dropna()
nmdData = pd.read_csv("/home/user/M.Sc/Samples/nmdmodel/newTrial/training_set/IC_SGS_00013.csv")

X = nmdData.drop(columns=['SV_call', "Gene", "Group", "meanRef"])
y = nmdData["SV_call"]


groups = nmdData["Group"]


logo = LeaveOneGroupOut()
all_y_true = []
all_y_pred = []
all_y_probs = []

print(f"Starting LOGO Cross-Validation on {nmdData['Group'].nunique()} samples...")

for train_idx, test_idx in logo.split(X, y, groups=groups):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]


    smote = SMOTE(random_state=2, sampling_strategy={0:int(max(y_train.value_counts()) * 0.1), 2:int(max(y_train.value_counts()) * 0.1)})
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    BalancedRandomForestClassifier = BalancedRandomForestClassifier(n_estimators=500, criterion='gini', max_depth=None,
                                min_samples_split=2, min_samples_leaf=1,min_weight_fraction_leaf=0.0,
                                max_features='sqrt',max_leaf_nodes=None,min_impurity_decrease=0.0,
                                bootstrap=True,oob_score=False, n_jobs=-1,
                                random_state=9, verbose=1,warm_start=False, class_weight={0:5, 1:1, 2:5})
    BalancedRandomForestClassifier.fit(X_train_res, y_train_res)
    
    all_y_true.extend(y_test)
    all_y_pred.extend(BalancedRandomForestClassifier.predict(X_test))
    all_y_probs.extend(BalancedRandomForestClassifier.predict_proba(X_test))

all_y_true = np.array(all_y_true)
all_y_pred = np.array(all_y_pred)
all_y_probs = np.array(all_y_probs)

print('\n--- HONEST PERFORMANCE METRICS ---')
prec, rec, f1, _ = precision_recall_fscore_support(all_y_true, all_y_pred)
labels = ["DEL", "Normal", "DUP"]

for i, label in enumerate(labels):
    print(f"{label} -> Precision: {prec[i]:.3f}, Recall: {rec[i]:.3f}, F1: {f1[i]:.3f}")

joblib.dump(BalancedRandomForestClassifier, '/cbio/projects/003/saifeldeen/samples/newTrial/models/dudel_smote_brf.pkl')

# Compute CM
cm = confusion_matrix(all_y_true, all_y_pred)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(8, 6))
sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap="Blues", 
            xticklabels=labels, yticklabels=labels)
plt.title("Normalized Confusion Matrix")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.savefig("/cbio/projects/003/saifeldeen/samples/newTrial/models/confusion_matrix.png")

# Binarize the output for multi-class plotting
y_true_bin = label_binarize(all_y_true, classes=[0, 1, 2])
n_classes = y_true_bin.shape[1]

fig, ax = plt.subplots(figsize=(8, 6))

colors = ['purple', 'gray', 'orange']
for i, color in enumerate(colors):
    display = PrecisionRecallDisplay.from_predictions(
        y_true_bin[:, i], 
        all_y_probs[:, i], 
        name=f"Class {labels[i]}", 
        color=color,
        ax=ax
    )

plt.title("Multi-class Precision-Recall Curve")
plt.savefig("/cbio/projects/003/saifeldeen/samples/newTrial/models/pr_curve.png")


importances = BalancedRandomForestClassifier.feature_importances_
feature_names = X.columns
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.title("Feature Importance for CNV Detection")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), [feature_names[i] for i in indices], rotation=45)
plt.tight_layout()
plt.savefig("/cbio/projects/003/saifeldeen/samples/newTrial/models/feature_importance.png")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from xgboost import XGBClassifier

# ==========================================
# LOAD DATASET
# ==========================================

data = pd.read_csv("data/UNSW_NB15_training-set.csv")

print("Dataset Loaded Successfully")
print("Dataset Shape:", data.shape)

# ==========================================
# ENCODE TARGET LABELS
# ==========================================

label_encoder = LabelEncoder()

data['label'] = label_encoder.fit_transform(data['attack_cat'])

# ==========================================
# FEATURE AND TARGET SELECTION
# ==========================================

X = data.drop(columns=['label', 'attack_cat'])

y = data['label']

# ==========================================
# HANDLE CATEGORICAL FEATURES
# ==========================================

X = pd.get_dummies(X)

# ==========================================
# FEATURE SCALING
# ==========================================

scaler = StandardScaler()

X = scaler.fit_transform(X)

# ==========================================
# TRAIN-TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

# ==========================================
# CLASSIFIER DEFINITIONS
# ==========================================

classifiers = {

    "Naive Bayes": GaussianNB(),

    "Bagging": BaggingClassifier(
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        random_state=42
    ),

    "MLP Neural Network": MLPClassifier(
        max_iter=300,
        random_state=42
    ),

    "XGBoost": XGBClassifier(
        eval_metric='mlogloss',
        random_state=42
    )
}

# ==========================================
# TRAINING AND EVALUATION
# ==========================================

results = []

best_model = None
best_model_name = ""
best_f1 = 0

for name, model in classifiers.items():

    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Metrics

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        average='weighted',
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        average='weighted',
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average='weighted',
        zero_division=0
    )

    results.append({
        "Classifier": name,
        "Accuracy": accuracy * 100,
        "Precision": precision * 100,
        "Recall": recall * 100,
        "F1-Score": f1 * 100
    })

    print(
        f"{name} --> "
        f"Accuracy: {accuracy:.4f}, "
        f"Precision: {precision:.4f}, "
        f"Recall: {recall:.4f}, "
        f"F1-Score: {f1:.4f}"
    )

    # Save Best Model

    if f1 > best_f1:
        best_f1 = f1
        best_model = model
        best_model_name = name

# ==========================================
# RESULTS DATAFRAME
# ==========================================

results_df = pd.DataFrame(results)

print("\n==============================")
print("PERFORMANCE SUMMARY")
print("==============================")

print(results_df)

# ==========================================
# SAVE RESULTS
# ==========================================

results_df.to_csv(
    "classifier_results_summary.csv",
    index=False
)

print("\nResults saved successfully.")

# ==========================================
# PERFORMANCE CHART
# ==========================================

plt.figure(figsize=(14, 7))

bar_width = 0.18

x = np.arange(len(results_df['Classifier']))

plt.bar(
    x,
    results_df['Accuracy'],
    width=bar_width,
    label='Accuracy'
)

plt.bar(
    x + bar_width,
    results_df['Precision'],
    width=bar_width,
    label='Precision'
)

plt.bar(
    x + (2 * bar_width),
    results_df['Recall'],
    width=bar_width,
    label='Recall'
)

plt.bar(
    x + (3 * bar_width),
    results_df['F1-Score'],
    width=bar_width,
    label='F1-Score'
)

# ==========================================
# CHART FORMATTING
# ==========================================

plt.xlabel("Machine Learning Classifiers")

plt.ylabel("Performance (%)")

plt.title(
    "Comparative Performance of Machine Learning Classifiers"
)

plt.xticks(
    x + bar_width,
    results_df['Classifier'],
    rotation=10
)

plt.legend()

plt.grid(
    axis='y',
    linestyle='--',
    alpha=0.7
)

plt.tight_layout()

# ==========================================
# SAVE PERFORMANCE CHART
# ==========================================

plt.savefig(
    "classifier_performance_chart.png",
    dpi=300
)

print("Performance chart saved successfully.")

plt.close()

# ==========================================
# CONFUSION MATRIX
# ==========================================

print(f"\nGenerating Confusion Matrix for {best_model_name}...")

y_pred_best = best_model.predict(X_test)

cm = confusion_matrix(y_test, y_pred_best)

plt.figure(figsize=(8, 6))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm
)

disp.plot(cmap='Blues')

plt.title(
    f"Confusion Matrix for {best_model_name}"
)

plt.tight_layout()

# ==========================================
# SAVE CONFUSION MATRIX
# ==========================================

plt.savefig(
    "confusion_matrix.png",
    dpi=300
)

print("Confusion matrix saved successfully.")

plt.close()

# ==========================================
# CLASSIFICATION REPORT
# ==========================================

print("\n==============================")
print("CLASSIFICATION REPORT")
print("==============================")

print(
    classification_report(
        y_test,
        y_pred_best
    )
)

# ==========================================
# BEST MODEL
# ==========================================

print("\n==============================")
print("BEST PERFORMING MODEL")
print("==============================")

print("Best Model:", best_model_name)

print("Best F1-Score:", round(best_f1 * 100, 2), "%")

# ==========================================
# ADVANCED EVALUATION FOR XGBOOST
# ==========================================

from sklearn.metrics import (
    roc_curve,
    auc,
    RocCurveDisplay
)

from sklearn.preprocessing import label_binarize

# ==========================================
# XGBOOST PREDICTIONS
# ==========================================

print("\nGenerating Advanced Evaluation Metrics...")

y_pred_xgb = best_model.predict(X_test)

# ==========================================
# CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(y_test, y_pred_xgb)

plt.figure(figsize=(8, 6))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm
)

disp.plot(cmap='Blues')

plt.title(
    f'Confusion Matrix for {best_model_name}'
)

plt.tight_layout()

plt.savefig(
    'xgboost_confusion_matrix.png',
    dpi=300
)

print("Confusion Matrix Saved Successfully.")

plt.close()

# ==========================================
# ROC CURVE
# ==========================================

# Binarize labels for ROC computation

classes = np.unique(y_test)

y_test_bin = label_binarize(
    y_test,
    classes=classes
)

# Predict probabilities

y_score = best_model.predict_proba(X_test)

# Compute ROC curve and ROC area

fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(len(classes)):

    fpr[i], tpr[i], _ = roc_curve(
        y_test_bin[:, i],
        y_score[:, i]
    )

    roc_auc[i] = auc(
        fpr[i],
        tpr[i]
    )

# ==========================================
# PLOT ROC CURVES
# ==========================================

plt.figure(figsize=(10, 8))

for i in range(len(classes)):

    plt.plot(
        fpr[i],
        tpr[i],
        lw=2,
        label=f'Class {i} (AUC = {roc_auc[i]:0.2f})'
    )

# Reference Line

plt.plot(
    [0, 1],
    [0, 1],
    linestyle='--'
)

# Labels and Title

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title(
    f'ROC Curve for {best_model_name}'
)

plt.legend(loc='lower right')

plt.grid(alpha=0.3)

plt.tight_layout()

# Save ROC Curve

plt.savefig(
    'xgboost_roc_curve.png',
    dpi=300
)

print("ROC Curve Saved Successfully.")

plt.close()

# ==========================================
# FEATURE IMPORTANCE
# ==========================================

if best_model_name == "XGBoost":

    importance = best_model.feature_importances_

    feature_names = pd.get_dummies(
        data.drop(columns=['label', 'attack_cat'])
    ).columns

    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    })

    # Top 10 Features

    top_features = feature_importance_df.sort_values(
        by='Importance',
        ascending=False
    ).head(10)

    # Plot Feature Importance

    plt.figure(figsize=(10, 6))

    plt.barh(
        top_features['Feature'],
        top_features['Importance']
    )

    plt.xlabel('Importance Score')

    plt.ylabel('Features')

    plt.title(
        'Top 10 Important Features in XGBoost'
    )

    plt.gca().invert_yaxis()

    plt.tight_layout()

    # Save Feature Importance Figure

    plt.savefig(
        'xgboost_feature_importance.png',
        dpi=300
    )

    print("Feature Importance Chart Saved Successfully.")

    plt.close()
"""
Machine Learning module for news classification.

Implements two classifiers for comparison:
  1. Logistic Regression + TF-IDF
  2. Linear SVM (LinearSVC) + TF-IDF

Both use the same TF-IDF vectorizer so features are consistent.
Models are trained once, evaluated, and saved using joblib.
"""
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from sklearn.preprocessing import LabelEncoder

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")


def train_and_evaluate(df: pd.DataFrame, text_column: str = "processed_text",
                       label_column: str = "category", test_size: float = 0.2,
                       random_state: int = 42) -> dict:
    """
    Train Logistic Regression and Linear SVM classifiers on the dataset.

    Pipeline:
      1. Encode labels (category → integer)
      2. Train/test split (80/20, stratified)
      3. Fit TF-IDF vectorizer on training data only
      4. Train both models
      5. Evaluate on test data
      6. Save all artifacts (models, vectorizer, encoder, metrics)

    Args:
        df: DataFrame with preprocessed text and category columns
        text_column: Name of the preprocessed text column
        label_column: Name of the category label column
        test_size: Fraction of data for testing
        random_state: Random seed for reproducibility

    Returns:
        Dictionary containing evaluation metrics for both models
    """
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Drop rows with missing text or labels
    df_clean = df[[text_column, label_column]].dropna()
    print(f"Training on {len(df_clean)} articles across {df_clean[label_column].nunique()} categories")

    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df_clean[label_column])
    X_text = df_clean[text_column].values

    # Stratified train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_text, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # TF-IDF Vectorization
    # Parameters chosen for a ~2000 article dataset:
    #   max_features=5000: keep top 5000 terms (sufficient for BBC dataset)
    #   ngram_range=(1,2): unigrams + bigrams for better context
    #   min_df=2: ignore terms appearing in fewer than 2 documents
    #   max_df=0.95: ignore terms appearing in >95% of documents
    tfidf = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True  # Apply log normalization to term frequencies
    )
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    print(f"TF-IDF matrix shape: {X_train_tfidf.shape}")

    results = {"models": [], "categories": list(label_encoder.classes_)}

    # --- Model 1: Logistic Regression ---
    lr_model = LogisticRegression(
        max_iter=1000,
        C=1.0,
        solver='lbfgs',
        random_state=random_state
    )
    lr_model.fit(X_train_tfidf, y_train)

    lr_pred = lr_model.predict(X_test_tfidf)

    lr_metrics = _compute_metrics("Logistic Regression", y_test, lr_pred, label_encoder)
    results["models"].append(lr_metrics)
    print(f"Logistic Regression accuracy: {lr_metrics['accuracy']:.4f}")

    # --- Model 2: Linear SVM ---
    svm_model = LinearSVC(
        C=1.0,
        max_iter=2000,
        random_state=random_state
    )
    svm_model.fit(X_train_tfidf, y_train)
    svm_pred = svm_model.predict(X_test_tfidf)

    svm_metrics = _compute_metrics("Linear SVM", y_test, svm_pred, label_encoder)
    results["models"].append(svm_metrics)
    print(f"Linear SVM accuracy: {svm_metrics['accuracy']:.4f}")

    # Save all artifacts
    joblib.dump(tfidf, os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib"))
    joblib.dump(label_encoder, os.path.join(MODELS_DIR, "label_encoder.joblib"))
    joblib.dump(lr_model, os.path.join(MODELS_DIR, "logistic_regression.joblib"))
    joblib.dump(svm_model, os.path.join(MODELS_DIR, "linear_svm.joblib"))
    joblib.dump(results, os.path.join(MODELS_DIR, "evaluation_results.joblib"))

    # Save the best model indicator (based on F1 macro)
    best_idx = 0 if results["models"][0]["f1_macro"] >= results["models"][1]["f1_macro"] else 1
    best_model_name = results["models"][best_idx]["name"]
    joblib.dump(best_model_name, os.path.join(MODELS_DIR, "best_model_name.joblib"))
    print(f"Best model (by macro F1): {best_model_name}")

    print(f"All artifacts saved to {MODELS_DIR}")
    return results


def _compute_metrics(model_name: str, y_true, y_pred, label_encoder) -> dict:
    """
    Compute classification metrics for a model.

    Returns accuracy, precision, recall, F1 (macro-averaged),
    per-class metrics, and the confusion matrix.
    """
    categories = list(label_encoder.classes_)

    # Overall metrics (macro-averaged for multi-class fairness)
    accuracy = accuracy_score(y_true, y_pred)
    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)

    # Per-class metrics
    precision_per = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per = f1_score(y_true, y_pred, average=None, zero_division=0)

    per_class = []
    for i, cat in enumerate(categories):
        per_class.append({
            "category": cat,
            "precision": round(float(precision_per[i]), 4),
            "recall": round(float(recall_per[i]), 4),
            "f1": round(float(f1_per[i]), 4)
        })

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    return {
        "name": model_name,
        "accuracy": round(float(accuracy), 4),
        "precision_macro": round(float(precision_macro), 4),
        "recall_macro": round(float(recall_macro), 4),
        "f1_macro": round(float(f1_macro), 4),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "categories": categories
    }


def get_top_tfidf_terms(tfidf_vectorizer, n: int = 30) -> list[dict]:
    """
    Extract the top TF-IDF terms by their IDF score.

    Higher IDF = term is more discriminative (appears in fewer documents).
    """
    feature_names = tfidf_vectorizer.get_feature_names_out()
    idf_scores = tfidf_vectorizer.idf_

    # Sort by IDF (descending = more discriminative)
    sorted_indices = np.argsort(idf_scores)[::-1][:n]

    return [
        {"term": feature_names[i], "score": round(float(idf_scores[i]), 4)}
        for i in sorted_indices
    ]


def get_top_tfidf_per_category(tfidf_vectorizer, model, label_encoder, n: int = 10) -> dict:
    """
    For each category, find the terms with the highest model coefficients.

    This works with Logistic Regression which has a coef_ matrix
    of shape (n_classes, n_features).
    """
    feature_names = tfidf_vectorizer.get_feature_names_out()

    if not hasattr(model, 'coef_'):
        return {}

    result = {}
    for i, category in enumerate(label_encoder.classes_):
        coefficients = model.coef_[i]
        top_indices = np.argsort(coefficients)[::-1][:n]
        result[category] = [
            {"term": feature_names[idx], "score": round(float(coefficients[idx]), 4)}
            for idx in top_indices
        ]

    return result

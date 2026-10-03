"""
Baatmi — News Analytics & NLP Backend

FastAPI application that serves real NLP/ML results from the BBC News dataset.

Startup sequence:
  1. Download dataset (if not present)
  2. Load and preprocess dataset
  3. Train models (if not already trained) / load saved artifacts
  4. Precompute analytics (sentiment, word frequencies)
  5. Serve API endpoints

All analytics come from actual computed results — nothing is hardcoded.
"""
import os
import sys
import traceback

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib

# Add backend dir to path so imports work when running from backend/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.data_service import DataService
from preprocessing.text_preprocessor import preprocess_text
from nlp.sentiment import analyze_sentiment
from ml.classifier import (
    train_and_evaluate,
    get_top_tfidf_terms,
    get_top_tfidf_per_category
)

# --- Global state ---
data_service = DataService()
ml_artifacts = {
    "tfidf": None,
    "label_encoder": None,
    "lr_model": None,
    "svm_model": None,
    "evaluation": None,
    "best_model_name": None,
    "tfidf_top_terms": None,
    "tfidf_per_category": None,
}

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def _load_or_train_models():
    """Load saved model artifacts, or train from scratch if not found."""
    model_path = os.path.join(MODELS_DIR, "logistic_regression.joblib")

    if os.path.exists(model_path):
        print("Loading pre-trained model artifacts...")
        ml_artifacts["tfidf"] = joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib"))
        ml_artifacts["label_encoder"] = joblib.load(os.path.join(MODELS_DIR, "label_encoder.joblib"))
        ml_artifacts["lr_model"] = joblib.load(os.path.join(MODELS_DIR, "logistic_regression.joblib"))
        ml_artifacts["svm_model"] = joblib.load(os.path.join(MODELS_DIR, "linear_svm.joblib"))
        ml_artifacts["evaluation"] = joblib.load(os.path.join(MODELS_DIR, "evaluation_results.joblib"))
        ml_artifacts["best_model_name"] = joblib.load(os.path.join(MODELS_DIR, "best_model_name.joblib"))
        print("All model artifacts loaded successfully.")
    else:
        print("No pre-trained models found. Training from scratch...")
        results = train_and_evaluate(data_service.df)
        # Reload the just-saved artifacts
        ml_artifacts["tfidf"] = joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib"))
        ml_artifacts["label_encoder"] = joblib.load(os.path.join(MODELS_DIR, "label_encoder.joblib"))
        ml_artifacts["lr_model"] = joblib.load(os.path.join(MODELS_DIR, "logistic_regression.joblib"))
        ml_artifacts["svm_model"] = joblib.load(os.path.join(MODELS_DIR, "linear_svm.joblib"))
        ml_artifacts["evaluation"] = results
        ml_artifacts["best_model_name"] = joblib.load(os.path.join(MODELS_DIR, "best_model_name.joblib"))
        print("Training complete and artifacts saved.")

    # Precompute TF-IDF analytics
    ml_artifacts["tfidf_top_terms"] = get_top_tfidf_terms(ml_artifacts["tfidf"], n=50)
    ml_artifacts["tfidf_per_category"] = get_top_tfidf_per_category(
        ml_artifacts["tfidf"], ml_artifacts["lr_model"], ml_artifacts["label_encoder"]
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application startup: load dataset, train/load models, precompute analytics.
    This runs once before the app starts serving requests.
    """
    try:
        # Step 1: Download dataset if needed
        dataset_path = os.path.join(DATA_DIR, "bbc_news.csv")
        if not os.path.exists(dataset_path):
            print("Dataset not found, downloading...")
            from data.download_dataset import download_dataset
            download_dataset()

        # Step 2: Load and preprocess dataset
        data_service.load_dataset(dataset_path)

        # Step 3: Train or load ML models
        _load_or_train_models()

        # Step 4: Precompute expensive analytics in background
        # (sentiment is computed lazily on first request)

        print("\n[OK] Backend ready. All systems operational.\n")

    except Exception as e:
        print(f"STARTUP ERROR: {e}")
        traceback.print_exc()

    yield  # App serves requests during this period

    print("Shutting down...")


# --- FastAPI App ---
app = FastAPI(
    title="Baatmi NLP Backend",
    description="News Analytics & NLP Machine Learning API — BBC News Dataset",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow the Next.js frontend to communicate
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Pydantic Models ---
class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    predicted_category: str
    confidence: float
    sentiment: dict
    top_terms: list


# ==================== API ROUTES ====================


@app.get("/api/health")
def health_check():
    """Health check endpoint reflecting actual backend state."""
    return {
        "status": "ok",
        "dataset_loaded": data_service.is_loaded,
        "models_loaded": ml_artifacts["lr_model"] is not None,
        "total_articles": len(data_service.df) if data_service.is_loaded else 0,
    }


# --- News Articles ---

@app.get("/api/news")
def get_news(
    search: str = Query("", description="Search query"),
    category: str = Query("", description="Category filter"),
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(20, ge=1, le=100, description="Articles per page"),
):
    """Search and paginate through news articles from the real dataset."""
    if not data_service.is_loaded:
        raise HTTPException(status_code=503, detail="Dataset not loaded")
    return data_service.search_articles(query=search, category=category, page=page, per_page=per_page)


@app.get("/api/news/{article_id}")
def get_article(article_id: int):
    """Get a single article with NLP analysis."""
    if not data_service.is_loaded:
        raise HTTPException(status_code=503, detail="Dataset not loaded")

    article = data_service.get_article(article_id)
    if article is None:
        raise HTTPException(status_code=404, detail="Article not found")

    # Add TF-IDF top terms for this article
    if ml_artifacts["tfidf"] is not None:
        tfidf_vec = ml_artifacts["tfidf"]
        processed = article["processed_text"]
        if processed:
            import numpy as np
            vec = tfidf_vec.transform([processed])
            feature_names = tfidf_vec.get_feature_names_out()
            scores = vec.toarray()[0]
            top_indices = np.argsort(scores)[::-1][:15]
            article["top_terms"] = [
                {"term": feature_names[i], "score": round(float(scores[i]), 4)}
                for i in top_indices if scores[i] > 0
            ]
        else:
            article["top_terms"] = []

    # Remove processed_text from response (not needed by frontend)
    article.pop("processed_text", None)

    return article


@app.get("/api/categories")
def get_categories():
    """Get list of all news categories."""
    if not data_service.is_loaded:
        raise HTTPException(status_code=503, detail="Dataset not loaded")
    return {"categories": data_service.get_categories()}


# --- Prediction ---

@app.post("/api/predict")
def predict(req: PredictRequest):
    """
    Predict the category of submitted text using the trained model.

    Pipeline:
      1. Preprocess text (same pipeline as training)
      2. TF-IDF transform
      3. Model prediction + confidence
      4. Sentiment analysis
      5. Top TF-IDF terms
    """
    if ml_artifacts["tfidf"] is None or ml_artifacts["lr_model"] is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    import numpy as np

    # Step 1: Preprocess
    processed = preprocess_text(req.text)

    # Step 2: TF-IDF
    tfidf_vec = ml_artifacts["tfidf"]
    X = tfidf_vec.transform([processed])

    # Step 3: Predict using the best model
    best_name = ml_artifacts["best_model_name"]
    if best_name == "Logistic Regression":
        model = ml_artifacts["lr_model"]
    else:
        model = ml_artifacts["svm_model"]

    prediction = model.predict(X)[0]
    label_encoder = ml_artifacts["label_encoder"]
    predicted_category = label_encoder.inverse_transform([prediction])[0]

    # Confidence (probability for LR, decision function distance for SVM)
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X)[0]
        confidence = float(np.max(proba))
        all_proba = {
            label_encoder.inverse_transform([i])[0]: round(float(p), 4)
            for i, p in enumerate(proba)
        }
    else:
        decision = model.decision_function(X)[0]
        # Convert decision function to pseudo-probability using softmax
        exp_scores = np.exp(decision - np.max(decision))
        proba = exp_scores / exp_scores.sum()
        confidence = float(np.max(proba))
        all_proba = {
            label_encoder.inverse_transform([i])[0]: round(float(p), 4)
            for i, p in enumerate(proba)
        }

    # Step 4: Sentiment
    sentiment = analyze_sentiment(req.text)

    # Step 5: Top terms
    feature_names = tfidf_vec.get_feature_names_out()
    scores = X.toarray()[0]
    top_indices = np.argsort(scores)[::-1][:10]
    top_terms = [
        {"term": feature_names[i], "score": round(float(scores[i]), 4)}
        for i in top_indices if scores[i] > 0
    ]

    return {
        "predicted_category": predicted_category,
        "confidence": round(confidence, 4),
        "probabilities": all_proba,
        "sentiment": sentiment,
        "top_terms": top_terms,
        "model_used": best_name,
    }


# --- Analytics ---

@app.get("/api/analytics")
def get_analytics():
    """Get complete dataset statistics and analytics overview."""
    if not data_service.is_loaded:
        raise HTTPException(status_code=503, detail="Dataset not loaded")
    stats = data_service.get_dataset_stats()
    return stats


@app.get("/api/analytics/sentiment")
def get_sentiment_analytics():
    """Get sentiment distribution across the entire dataset."""
    if not data_service.is_loaded:
        raise HTTPException(status_code=503, detail="Dataset not loaded")
    return data_service.compute_all_sentiments()


@app.get("/api/analytics/categories")
def get_category_analytics():
    """Get category distribution and per-category statistics."""
    if not data_service.is_loaded:
        raise HTTPException(status_code=503, detail="Dataset not loaded")

    stats = data_service.get_dataset_stats()
    categories = stats.get("categories", {})

    result = []
    for cat, count in categories.items():
        cat_df = data_service.df[data_service.df['category'] == cat]
        result.append({
            "category": cat,
            "count": count,
            "avg_word_count": round(float(cat_df['word_count'].mean()), 1),
            "percentage": round(count / stats["total_articles"] * 100, 1),
        })

    return {"categories": result, "total": stats["total_articles"]}


@app.get("/api/analytics/top-words")
def get_top_words(n: int = Query(30, ge=1, le=100)):
    """Get most frequent words across the preprocessed dataset."""
    if not data_service.is_loaded:
        raise HTTPException(status_code=503, detail="Dataset not loaded")
    return {"terms": data_service.get_word_frequencies_cached(top_n=n)}


@app.get("/api/analytics/tfidf")
def get_tfidf_terms(n: int = Query(30, ge=1, le=100)):
    """Get top TF-IDF terms (most discriminative words)."""
    if ml_artifacts["tfidf_top_terms"] is None:
        raise HTTPException(status_code=503, detail="TF-IDF not computed")
    return {"terms": ml_artifacts["tfidf_top_terms"][:n]}


@app.get("/api/analytics/tfidf/categories")
def get_tfidf_by_category():
    """Get top TF-IDF terms per category (based on model coefficients)."""
    if ml_artifacts["tfidf_per_category"] is None:
        raise HTTPException(status_code=503, detail="TF-IDF not computed")
    return {"categories": ml_artifacts["tfidf_per_category"]}


# --- Model Comparison ---

@app.get("/api/models")
def get_model_comparison():
    """
    Get real evaluation metrics for all trained models.
    These come from actual train/test evaluation, not hardcoded values.
    """
    if ml_artifacts["evaluation"] is None:
        raise HTTPException(status_code=503, detail="Models not evaluated")
    return ml_artifacts["evaluation"]


@app.get("/api/methodology")
def get_methodology():
    """
    Return the NLP/ML methodology used in this project.
    Useful for the academic transparency section.
    """
    return {
        "dataset": {
            "name": "BBC News Dataset",
            "source": "http://mlg.ucd.ie/datasets/bbc.html",
            "size": len(data_service.df) if data_service.is_loaded else 0,
            "categories": data_service.get_categories(),
            "description": "2,225 news articles from the BBC, categorized into 5 topics."
        },
        "preprocessing": {
            "steps": [
                "Text cleaning (remove HTML, URLs, special characters)",
                "Lowercase conversion",
                "Tokenization (NLTK word_tokenize)",
                "Stopword removal (NLTK English stopwords + domain-specific)",
                "Lemmatization (NLTK WordNetLemmatizer)"
            ],
            "library": "NLTK 3.x"
        },
        "feature_extraction": {
            "method": "TF-IDF (Term Frequency — Inverse Document Frequency)",
            "parameters": {
                "max_features": 5000,
                "ngram_range": "(1, 2)",
                "min_df": 2,
                "max_df": 0.95,
                "sublinear_tf": True
            },
            "library": "scikit-learn TfidfVectorizer"
        },
        "sentiment_analysis": {
            "method": "TextBlob (lexicon-based polarity analysis)",
            "range": "-1.0 (negative) to +1.0 (positive)",
            "thresholds": {"positive": "> 0.05", "negative": "< -0.05", "neutral": "-0.05 to 0.05"}
        },
        "classification": {
            "models": ["Logistic Regression (multinomial)", "Linear SVM (LinearSVC)"],
            "train_test_split": "80/20 (stratified)",
            "evaluation_metrics": ["Accuracy", "Precision (macro)", "Recall (macro)", "F1-score (macro)", "Confusion matrix"],
            "library": "scikit-learn"
        },
        "pipeline_diagram": [
            "Raw Text",
            "↓ Text Cleaning",
            "↓ Tokenization",
            "↓ Stopword Removal",
            "↓ Lemmatization",
            "↓ TF-IDF Vectorization",
            "↓ ML Classification",
            "↓ Evaluation",
            "Results"
        ]
    }


# --- Run ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

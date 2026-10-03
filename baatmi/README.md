# Baatmi — News Analytics & NLP Machine Learning System

A full-stack, end-to-end News Analytics and Natural Language Processing (NLP) system built with **Next.js 16** (V0 frontend design) and a high-performance **FastAPI** Python machine learning backend.

---

## 1. Project Overview

Baatmi ingests a real news corpus (**BBC News Dataset**, 2,225 genuine news articles across 5 topic domains), executes an end-to-end NLP preprocessing pipeline, extracts high-dimensional TF-IDF textual features, computes real-time sentiment polarity and subjectivity using TextBlob, and performs supervised topic classification using **Multinomial Logistic Regression** and **Linear Support Vector Classifier (LinearSVC)**.

All statistics, metrics, confusion matrices, and predictions displayed on the dashboard are **100% computed from real data** — no simulated or hardcoded values.

---

## 2. Dataset

* **Source**: BBC News Dataset (D. Greene and P. Cunningham, *Practical Solutions to the Problem of Diagonal Dominance in Kernel Document Clustering*, ICML 2006).
* **Official URL**: `http://mlg.ucd.ie/datasets/bbc.html`
* **Total Articles**: 2,225 genuine news articles
* **Categories / Topics**:
  1. **Sport**: 511 articles (23.0%)
  2. **Business**: 510 articles (22.9%)
  3. **Politics**: 417 articles (18.7%)
  4. **Tech**: 401 articles (18.0%)
  5. **Entertainment**: 386 articles (17.3%)
* **Schema**:
  * `category`: Ground-truth topic label
  * `title`: Headline of the news article
  * `text`: Full raw text content
  * `word_count`: Computed word length (range: 90 – 4,492 words; mean: ~390 words)

---

## 3. NLP Preprocessing Pipeline

The system implements a reproducible, modular text preprocessing pipeline (`backend/preprocessing/text_preprocessor.py`) shared identically between training and real-time inference:

1. **Noise Cleaning**: Strips HTML tags, web URLs, email addresses, and non-ASCII character noise.
2. **Case Normalization**: Converts all text to lowercase.
3. **Regex Tokenization**: Extracts alphabetic word tokens of length ≥ 3 using word boundary regex (`\b[a-zA-Z]{3,}\b`), filtering out numbers and stray punctuation.
4. **Stopword Elimination**: Removes standard English stopwords (179 words from the NLTK English corpus) plus news domain filler words (`said`, `would`, `could`, `also`, `time`, `year`, `mr`).
5. **Stemming / Lemmatization**: Normalizes morphological word variants using Porter Stemming / WordNet Lemmatization.
6. **Feature Extraction (TF-IDF)**:
   * **N-gram Range**: `(1, 2)` (Unigrams + Bigrams)
   * **Vocabulary**: 5,000 most informative terms
   * **Document Frequency**: `min_df=2`, `max_df=0.95`
   * **Sublinear Scaling**: `sublinear_tf=True` (applies $1 + \log(\text{tf})$ to prevent high-frequency term dominance).

---

## 4. Machine Learning & Sentiment Analysis

### Supervised Classification Models
* **Train / Test Split**: 80% train (1,780 articles), 20% test (445 articles), stratified by class.
* **Model 1: Multinomial Logistic Regression (`L-BFGS`, $C=1.0$)**
  * **Test Accuracy**: **97.98%**
  * **Macro F1-Score**: **97.97%**
* **Model 2: Linear Support Vector Classifier (`LinearSVC`, $C=1.0$)**
  * **Test Accuracy**: **98.20%**
  * **Macro F1-Score**: **98.19%**
  * *Selected as the primary production inference model due to maximum-margin separation in high-dimensional sparse TF-IDF space.*

### Sentiment Analysis
* **Engine**: TextBlob Lexicon-Based Pattern Analyzer
* **Polarity Score**: $[-1.0, +1.0]$ ($> 0.05$: Positive, $< -0.05$: Negative, $[-0.05, 0.05]$: Neutral)
* **Subjectivity Score**: $[0.0, 1.0]$ ($0.0$: Objective factual news, $1.0$: Highly opinionated/editorial)
* **Corpus Distribution**:
  * Positive: ~69.6% (1,549 articles)
  * Neutral: ~25.4% (565 articles)
  * Negative: ~5.0% (111 articles)

---

## 5. Project Architecture

```
baatmi/
│
├── backend/                         # FastAPI Python Backend
│   ├── main.py                      # FastAPI app, CORS, routes & lifespan
│   ├── requirements.txt             # Python dependencies
│   ├── data/
│   │   ├── bbc_news.csv             # Raw BBC News dataset (2,225 records)
│   │   ├── bbc_news_preprocessed.csv# Preprocessed cached dataset
│   │   └── download_dataset.py      # Automated dataset downloader
│   ├── models/                      # Saved joblib artifacts
│   │   ├── tfidf_vectorizer.joblib  # Trained TfidfVectorizer (5,000 terms)
│   │   ├── label_encoder.joblib     # Scikit-learn LabelEncoder
│   │   ├── logistic_regression.joblib# Trained Logistic Regression model
│   │   ├── linear_svm.joblib        # Trained Linear SVM model
│   │   └── evaluation_results.joblib# Complete test metrics & confusion matrices
│   ├── preprocessing/
│   │   └── text_preprocessor.py     # Cleaning, tokenization, stopwords, stemming
│   ├── nlp/
│   │   └── sentiment.py             # TextBlob sentiment analysis & distribution
│   ├── ml/
│   │   └── classifier.py            # Train, evaluate & TF-IDF term extraction
│   └── services/
│       └── data_service.py          # Cached data loading, filtering, pagination
│
├── app/                             # Next.js 16 Frontend (V0 Visual Identity)
│   ├── layout.tsx                   # App shell layout & metadata
│   ├── page.tsx                     # Main interactive application & views
│   └── globals.css                  # Custom styling tokens, themes, cards & tables
├── lib/
│   ├── api.ts                       # Typed frontend API client
│   └── utils.ts                     # Utility helpers
├── .env.local                       # Environment variables (NEXT_PUBLIC_API_URL)
└── README.md
```

---

## 6. API Endpoints Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Service health status, total articles, and model status |
| `GET` | `/api/news` | Search, filter, and paginate real news (`search`, `category`, `page`, `per_page`) |
| `GET` | `/api/news/{id}` | Single article metadata, sentiment, and extracted TF-IDF terms |
| `GET` | `/api/categories` | List of all 5 news categories |
| `POST` | `/api/predict` | Real-time classification, probability distribution, and sentiment for input text |
| `GET` | `/api/analytics` | Dataset overview, word count distributions, min/max/average |
| `GET` | `/api/analytics/sentiment`| Corpus-wide sentiment distribution & percentages |
| `GET` | `/api/analytics/categories` | Per-category article counts and average lengths |
| `GET` | `/api/analytics/top-words` | Most frequent corpus tokens (`n` parameter) |
| `GET` | `/api/analytics/tfidf` | Top discriminative vocabulary terms by IDF weight |
| `GET` | `/api/models` | Real test accuracy, macro F1, per-class metrics & confusion matrices |
| `GET` | `/api/methodology` | Pipeline documentation for academic transparency |
| `GET` | `/docs` | Interactive Swagger / OpenAPI documentation |

---

## 7. Setup & Run Instructions

### Prerequisites
* **Python**: 3.10+ (tested on Python 3.14)
* **Node.js**: v18+ (tested on Node.js v22)
* **npm** or **pnpm**

---

### Step 1: Backend Setup & Model Training

1. Navigate to the project root and install Python requirements:
   ```bash
   pip install -r backend/requirements.txt
   ```

2. Download the BBC News dataset (if not already downloaded):
   ```bash
   python backend/data/download_dataset.py
   ```

3. Train and evaluate the models (creates serialized artifacts in `backend/models/`):
   ```bash
   python -c "import sys; sys.path.insert(0, 'backend'); from services.data_service import DataService; ds = DataService(); ds.load_dataset(); from ml.classifier import train_and_evaluate; train_and_evaluate(ds.df)"
   ```

4. Start the FastAPI backend server:
   ```bash
   python -m uvicorn main:app --app-dir backend --host 127.0.0.1 --port 8000 --reload
   ```
   *The API will be live at `http://127.0.0.1:8000` with interactive docs at `http://127.0.0.1:8000/docs`.*

---

### Step 2: Frontend Setup

1. Install Node.js dependencies:
   ```bash
   npm install --legacy-peer-deps
   ```

2. Ensure `.env.local` points to your backend:
   ```env
   NEXT_PUBLIC_API_URL=http://localhost:8000
   ```

3. Start the Next.js development server:
   ```bash
   npm run dev
   ```
   *The web application will be accessible at `http://localhost:3000`.*

4. To build for production:
   ```bash
   npm run build
   npm run start
   ```

---

## 8. Academic & Methodology Highlights

* **No Synthetic Metrics**: All classification scores reflect the actual performance of scikit-learn models on held-out test data.
* **Identical Training & Inference Pipelines**: Both training and live prediction use the exact same tokenizer, stopword cleaner, stemmer, and TF-IDF vocabulary matrix.
* **Explainable NLP**: Every news article detail view highlights the top TF-IDF features contributing to its category vector, alongside TextBlob polarity and subjectivity indicators.

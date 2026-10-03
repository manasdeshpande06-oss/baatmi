"""
Data service — loads, preprocesses, and caches the BBC News dataset.

This is loaded ONCE at application startup and cached in memory.
All API endpoints read from this cached data rather than reloading.
"""
import os
import pandas as pd
import numpy as np
from preprocessing.text_preprocessor import preprocess_text, get_word_frequencies, ensure_nltk_data
from nlp.sentiment import analyze_sentiment, sentiment_distribution


DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "bbc_news.csv")


class DataService:
    """
    Singleton-like service that holds the loaded and preprocessed dataset.
    Initialized once during FastAPI startup.
    """

    def __init__(self):
        self.df: pd.DataFrame | None = None
        self.is_loaded: bool = False
        self._sentiment_cache: dict | None = None
        self._stats_cache: dict | None = None
        self._word_freq_cache: list | None = None

    def load_dataset(self, path: str = DATA_PATH):
        """
        Load the BBC News dataset CSV and preprocess all articles.
        Caches preprocessed text to disk for instantaneous subsequent startups.
        """
        preprocessed_path = os.path.join(os.path.dirname(path), "bbc_news_preprocessed.csv")
        
        if os.path.exists(preprocessed_path):
            print(f"Loading cached preprocessed dataset from {preprocessed_path}...")
            self.df = pd.read_csv(preprocessed_path)
            self.is_loaded = True
            print(f"Dataset loaded: {len(self.df)} articles across {self.df['category'].nunique()} categories")
            return

        print(f"Loading dataset from {path}...")
        self.df = pd.read_csv(path)
        print(f"Raw dataset shape: {self.df.shape}")
        print(f"Columns: {list(self.df.columns)}")

        # Add an ID column if missing
        if 'id' not in self.df.columns:
            self.df['id'] = range(len(self.df))

        # Use title if present, otherwise extract from first line
        if 'title' not in self.df.columns or self.df['title'].isnull().all():
            self.df['title'] = self.df['text'].apply(self._extract_title)

        # Preprocess all article text
        print("Preprocessing text (cleaning + tokenization + lemmatization)...")
        ensure_nltk_data()
        self.df['processed_text'] = self.df['text'].apply(preprocess_text)

        # Compute word counts
        self.df['word_count'] = self.df['text'].apply(lambda x: len(str(x).split()))

        # Compute article lengths (characters)
        self.df['char_count'] = self.df['text'].apply(lambda x: len(str(x)))

        # Save preprocessed cache
        try:
            self.df.to_csv(preprocessed_path, index=False, encoding='utf-8')
            print(f"Saved preprocessed dataset cache to {preprocessed_path}")
        except Exception as e:
            print(f"Warning: could not save preprocessed cache: {e}")

        self.is_loaded = True
        print(f"Dataset loaded: {len(self.df)} articles across {self.df['category'].nunique()} categories")
        print(f"Categories: {self.df['category'].value_counts().to_dict()}")


    def _extract_title(self, text: str) -> str:
        """Extract a title from the first sentence of article text."""
        if not isinstance(text, str):
            return "Untitled"
        # Take the first sentence or first 120 chars
        first_line = text.strip().split('\n')[0].strip()
        # If the first line is very short, it's probably the actual headline
        if len(first_line) < 200:
            return first_line
        # Otherwise truncate
        return first_line[:120] + "..."

    def get_article(self, article_id: int) -> dict | None:
        """Get a single article by ID with computed NLP features."""
        if not self.is_loaded or self.df is None:
            return None
        row = self.df[self.df['id'] == article_id]
        if row.empty:
            return None
        row = row.iloc[0]

        # Compute sentiment on-demand for individual articles
        sentiment = analyze_sentiment(row['text'])

        return {
            "id": int(row['id']),
            "title": row['title'],
            "text": row['text'],
            "category": row['category'],
            "word_count": int(row['word_count']),
            "char_count": int(row['char_count']),
            "sentiment": sentiment,
            "processed_text": row['processed_text']
        }

    def search_articles(self, query: str = "", category: str = "",
                        page: int = 1, per_page: int = 20,
                        sort_by: str = "id") -> dict:
        """
        Search and filter articles with pagination.

        Args:
            query: Search text (matches title and article text)
            category: Filter by category (empty = all)
            page: Page number (1-indexed)
            per_page: Articles per page
            sort_by: Column to sort by

        Returns:
            Dict with articles list, pagination info, and total count
        """
        if not self.is_loaded or self.df is None:
            return {"articles": [], "total": 0, "page": 1, "per_page": per_page, "total_pages": 0}

        filtered = self.df.copy()

        # Category filter
        if category and category.lower() != "all":
            filtered = filtered[filtered['category'].str.lower() == category.lower()]

        # Text search (case-insensitive on title + text)
        if query:
            query_lower = query.lower()
            mask = (
                filtered['title'].str.lower().str.contains(query_lower, na=False) |
                filtered['text'].str.lower().str.contains(query_lower, na=False)
            )
            filtered = filtered[mask]

        total = len(filtered)
        total_pages = max(1, (total + per_page - 1) // per_page)
        page = max(1, min(page, total_pages))

        start = (page - 1) * per_page
        end = start + per_page
        page_data = filtered.iloc[start:end]

        articles = []
        for _, row in page_data.iterrows():
            articles.append({
                "id": int(row['id']),
                "title": row['title'],
                "category": row['category'],
                "word_count": int(row['word_count']),
                "summary": str(row['text'])[:200] + "..." if len(str(row['text'])) > 200 else str(row['text']),
            })

        return {
            "articles": articles,
            "total": total,
            "page": page,
            "per_page": per_page,
            "total_pages": total_pages
        }

    def get_dataset_stats(self) -> dict:
        """
        Compute and cache dataset-level statistics.
        Called once, results are cached for subsequent requests.
        """
        if self._stats_cache is not None:
            return self._stats_cache

        if not self.is_loaded or self.df is None:
            return {}

        categories = self.df['category'].value_counts().to_dict()

        self._stats_cache = {
            "total_articles": len(self.df),
            "total_categories": self.df['category'].nunique(),
            "categories": categories,
            "avg_word_count": round(float(self.df['word_count'].mean()), 1),
            "median_word_count": round(float(self.df['word_count'].median()), 1),
            "min_word_count": int(self.df['word_count'].min()),
            "max_word_count": int(self.df['word_count'].max()),
            "avg_char_count": round(float(self.df['char_count'].mean()), 1),
            "word_count_distribution": self._get_length_distribution(),
        }
        return self._stats_cache

    def _get_length_distribution(self) -> list[dict]:
        """Compute article length distribution for histogram display."""
        bins = [0, 100, 200, 300, 400, 500, 750, 1000, 1500, 2000, float('inf')]
        labels = ['0-100', '100-200', '200-300', '300-400', '400-500',
                  '500-750', '750-1000', '1000-1500', '1500-2000', '2000+']
        counts = pd.cut(self.df['word_count'], bins=bins, labels=labels).value_counts().sort_index()
        return [{"range": label, "count": int(count)} for label, count in counts.items()]

    def get_word_frequencies_cached(self, top_n: int = 50) -> list[dict]:
        """Get cached word frequency data."""
        if self._word_freq_cache is not None:
            return self._word_freq_cache[:top_n]

        if not self.is_loaded or self.df is None:
            return []

        self._word_freq_cache = get_word_frequencies(
            self.df['processed_text'].dropna().tolist(), top_n=100
        )
        return self._word_freq_cache[:top_n]

    def compute_all_sentiments(self) -> dict:
        """
        Compute sentiment for all articles and cache the distribution.
        This is expensive (~2225 TextBlob calls) so we do it once.
        """
        if self._sentiment_cache is not None:
            return self._sentiment_cache

        if not self.is_loaded or self.df is None:
            return {}

        print("Computing sentiment for all articles...")
        sentiments = []
        for text in self.df['text']:
            sentiments.append(analyze_sentiment(str(text)))

        # Store sentiment labels on the dataframe for per-category analysis
        self.df['sentiment_label'] = [s['label'] for s in sentiments]
        self.df['sentiment_score'] = [s['score'] for s in sentiments]

        dist = sentiment_distribution(sentiments)

        # Per-category sentiment
        per_category = {}
        for cat in self.df['category'].unique():
            cat_sentiments = [s for s, c in zip(sentiments, self.df['category']) if c == cat]
            per_category[cat] = sentiment_distribution(cat_sentiments)

        self._sentiment_cache = {
            "overall": dist,
            "per_category": per_category,
            "avg_score": round(float(np.mean([s['score'] for s in sentiments])), 4),
            "avg_subjectivity": round(float(np.mean([s['subjectivity'] for s in sentiments])), 4),
        }
        print("Sentiment analysis complete.")
        return self._sentiment_cache

    def get_categories(self) -> list[str]:
        """Get list of unique categories."""
        if not self.is_loaded or self.df is None:
            return []
        return sorted(self.df['category'].unique().tolist())

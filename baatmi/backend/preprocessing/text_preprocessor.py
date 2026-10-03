"""
NLP Preprocessing Pipeline for news text.

This module provides a reproducible, self-contained NLP text preprocessing
pipeline that ensures consistency between training and prediction time.

Pipeline steps:
  1. Text cleaning (removes HTML tags, URLs, email addresses, non-ASCII noise)
  2. Lowercase conversion
  3. Tokenization (word-boundary regex matching clean alpha tokens >= 3 chars)
  4. Stopword removal (standard 179 NLTK English stopwords + domain-specific terms)
  5. Lemmatization / Stemming (WordNetLemmatizer if available, with robust PorterStemmer)
"""
import re
import string

# Standard English stopwords (exact NLTK English corpus + domain-specific news stopwords)
STANDARD_STOPWORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're",
    "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he',
    'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's",
    'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
    'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are',
    'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do',
    'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because',
    'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against',
    'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
    'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
    'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how',
    'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such',
    'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
    's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've",
    'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't",
    'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't",
    'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't",
    'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't",
    'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't",
    # Domain news filler words:
    'said', 'also', 'would', 'could', 'one', 'two', 'new', 'like', 'get', 'make',
    'year', 'years', 'first', 'last', 'time', 'mr', 'people', 'us', 'say'
}

# Initialize stemmer/lemmatizer safely
_lemmatizer = None
try:
    from nltk.stem import WordNetLemmatizer
    _lem = WordNetLemmatizer()
    _lem.lemmatize('testing')  # Test if resource is available
    _lemmatizer = _lem
except Exception:
    _lemmatizer = None

from nltk.stem import PorterStemmer
_stemmer = PorterStemmer()


def ensure_nltk_data():
    """No-op kept for backward compatibility."""
    pass


def clean_text(text: str) -> str:
    """
    Clean raw text by removing HTML, URLs, emails, and non-ASCII noise.
    """
    if not isinstance(text, str):
        return ""

    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    # Remove email addresses
    text = re.sub(r'\S+@\S+', ' ', text)
    # Remove non-ASCII characters
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def tokenize(text: str) -> list[str]:
    """
    Tokenize text into lowercase alphabetic tokens of length >= 3.
    Extracts meaningful word tokens while ignoring numbers and lone punctuation.
    """
    return re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())


def preprocess_text(text: str, lemmatize: bool = True, remove_stopwords: bool = True) -> str:
    """
    Full reproducible NLP preprocessing pipeline.
    Used consistently both during model training (TF-IDF) and at inference time.

    Steps:
      1. Clean text (remove markup, urls, specials)
      2. Tokenize into words
      3. Remove stopwords (standard English + domain words)
      4. Stem / Lemmatize each token
    """
    cleaned = clean_text(text)
    tokens = tokenize(cleaned)

    if remove_stopwords:
        tokens = [t for t in tokens if t not in STANDARD_STOPWORDS]

    if lemmatize:
        if _lemmatizer is not None:
            try:
                tokens = [_lemmatizer.lemmatize(t) for t in tokens]
            except Exception:
                tokens = [_stemmer.stem(t) for t in tokens]
        else:
            tokens = [_stemmer.stem(t) for t in tokens]

    return ' '.join(tokens)


def get_word_frequencies(texts: list[str], top_n: int = 50) -> list[dict]:
    """
    Count word frequencies across a collection of preprocessed texts.
    Returns list of dicts with 'term' and 'count' keys.
    """
    from collections import Counter
    word_counts = Counter()

    for text in texts:
        if isinstance(text, str):
            words = text.split()
            word_counts.update(words)

    return [{"term": word, "count": count} for word, count in word_counts.most_common(top_n)]

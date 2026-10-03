"""
Sentiment Analysis module using TextBlob.

TextBlob uses a pattern-based approach (built on the Pattern library) that
assigns polarity scores ranging from -1.0 (most negative) to +1.0 (most positive).

This is a lightweight, interpretable approach suitable for academic projects.
It does NOT require model training — it uses a pre-built lexicon.
"""
from textblob import TextBlob


def analyze_sentiment(text: str) -> dict:
    """
    Analyze the sentiment of a text using TextBlob.

    TextBlob's sentiment analysis is based on a pre-trained sentiment lexicon
    where each word has a polarity score. The overall polarity is the average
    of word-level polarities.

    Args:
        text: The text to analyze

    Returns:
        Dictionary with:
          - label: "positive", "negative", or "neutral"
          - score: polarity value from -1.0 to 1.0
          - subjectivity: 0.0 (objective) to 1.0 (subjective)
    """
    if not text or not isinstance(text, str):
        return {"label": "neutral", "score": 0.0, "subjectivity": 0.0}

    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity

    # Classify based on polarity thresholds
    if polarity > 0.05:
        label = "positive"
    elif polarity < -0.05:
        label = "negative"
    else:
        label = "neutral"

    return {
        "label": label,
        "score": round(polarity, 4),
        "subjectivity": round(subjectivity, 4)
    }


def batch_sentiment(texts: list[str]) -> list[dict]:
    """Analyze sentiment for a batch of texts."""
    return [analyze_sentiment(t) for t in texts]


def sentiment_distribution(sentiments: list[dict]) -> dict:
    """
    Calculate distribution of sentiment labels.

    Args:
        sentiments: List of sentiment dicts from analyze_sentiment()

    Returns:
        Dictionary with counts and percentages for each label
    """
    total = len(sentiments)
    if total == 0:
        return {"positive": 0, "negative": 0, "neutral": 0,
                "positive_pct": 0, "negative_pct": 0, "neutral_pct": 0}

    counts = {"positive": 0, "negative": 0, "neutral": 0}
    for s in sentiments:
        label = s.get("label", "neutral")
        counts[label] = counts.get(label, 0) + 1

    return {
        "positive": counts["positive"],
        "negative": counts["negative"],
        "neutral": counts["neutral"],
        "positive_pct": round(counts["positive"] / total * 100, 1),
        "negative_pct": round(counts["negative"] / total * 100, 1),
        "neutral_pct": round(counts["neutral"] / total * 100, 1),
        "total": total
    }

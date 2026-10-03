/**
 * API service for Baatmi Frontend
 * Connects directly to the FastAPI NLP/ML backend.
 * Uses NEXT_PUBLIC_API_URL environment variable with fallback to http://127.0.0.1:8000
 */

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000'

export interface Article {
  id: number
  title: string
  category: string
  word_count: number
  summary: string
  text?: string
  char_count?: number
  sentiment?: {
    label: string
    score: number
    subjectivity: number
  }
  top_terms?: Array<{
    term: string
    score: number
  }>
}

export interface NewsResponse {
  articles: Article[]
  total: number
  page: number
  per_page: number
  total_pages: number
}

export interface DatasetStats {
  total_articles: number
  total_categories: number
  categories: Record<string, number>
  avg_word_count: number
  median_word_count: number
  min_word_count: number
  max_word_count: number
  avg_char_count: number
  word_count_distribution: Array<{
    range: string
    count: number
  }>
}

export interface SentimentData {
  overall: {
    positive: number
    negative: number
    neutral: number
    positive_pct: number
    negative_pct: number
    neutral_pct: number
    total: number
  }
  per_category: Record<string, {
    positive: number
    negative: number
    neutral: number
    positive_pct: number
    negative_pct: number
    neutral_pct: number
    total: number
  }>
  avg_score: number
  avg_subjectivity: number
}

export interface ModelMetrics {
  name: string
  accuracy: number
  precision_macro: number
  recall_macro: number
  f1_macro: number
  per_class: Array<{
    category: string
    precision: number
    recall: number
    f1: number
  }>
  confusion_matrix: number[][]
  categories: string[]
}

export interface ModelsResponse {
  models: ModelMetrics[]
  categories: string[]
}

export interface PredictionResult {
  predicted_category: string
  confidence: number
  probabilities: Record<string, number>
  sentiment: {
    label: string
    score: number
    subjectivity: number
  }
  top_terms: Array<{
    term: string
    score: number
  }>
  model_used: string
}

export interface HealthStatus {
  status: string
  dataset_loaded: boolean
  models_loaded: boolean
  total_articles: number
}

export interface MethodologyData {
  dataset: {
    name: string
    source: string
    size: number
    categories: string[]
    description: string
  }
  preprocessing: {
    steps: string[]
    library: string
  }
  feature_extraction: {
    method: string
    parameters: Record<string, any>
    library: string
  }
  sentiment_analysis: {
    method: string
    range: string
    thresholds: Record<string, string>
  }
  classification: {
    models: string[]
    train_test_split: string
    evaluation_metrics: string[]
    library: string
  }
  pipeline_diagram: string[]
}

export const api = {
  getHealth: async (): Promise<HealthStatus> => {
    const res = await fetch(`${API_BASE_URL}/api/health`)
    if (!res.ok) throw new Error(`Health check failed: ${res.statusText}`)
    return res.json()
  },

  getNews: async (search = '', category = '', page = 1, perPage = 20): Promise<NewsResponse> => {
    const params = new URLSearchParams()
    if (search) params.set('search', search)
    if (category && category !== 'All stories' && category !== 'all') params.set('category', category)
    params.set('page', page.toString())
    params.set('per_page', perPage.toString())

    const res = await fetch(`${API_BASE_URL}/api/news?${params.toString()}`)
    if (!res.ok) throw new Error(`Failed to fetch news: ${res.statusText}`)
    return res.json()
  },

  getArticle: async (id: number): Promise<Article> => {
    const res = await fetch(`${API_BASE_URL}/api/news/${id}`)
    if (!res.ok) throw new Error(`Failed to fetch article: ${res.statusText}`)
    return res.json()
  },

  getAnalytics: async (): Promise<DatasetStats> => {
    const res = await fetch(`${API_BASE_URL}/api/analytics`)
    if (!res.ok) throw new Error(`Failed to fetch analytics: ${res.statusText}`)
    return res.json()
  },

  getSentiment: async (): Promise<SentimentData> => {
    const res = await fetch(`${API_BASE_URL}/api/analytics/sentiment`)
    if (!res.ok) throw new Error(`Failed to fetch sentiment: ${res.statusText}`)
    return res.json()
  },

  getCategoryAnalytics: async () => {
    const res = await fetch(`${API_BASE_URL}/api/analytics/categories`)
    if (!res.ok) throw new Error(`Failed to fetch category analytics: ${res.statusText}`)
    return res.json()
  },

  getTopWords: async (n = 30): Promise<{ terms: Array<{ term: string; count: number }> }> => {
    const res = await fetch(`${API_BASE_URL}/api/analytics/top-words?n=${n}`)
    if (!res.ok) throw new Error(`Failed to fetch top words: ${res.statusText}`)
    return res.json()
  },

  getTfidfTerms: async (n = 30): Promise<{ terms: Array<{ term: string; score: number }> }> => {
    const res = await fetch(`${API_BASE_URL}/api/analytics/tfidf?n=${n}`)
    if (!res.ok) throw new Error(`Failed to fetch TF-IDF terms: ${res.statusText}`)
    return res.json()
  },

  getModels: async (): Promise<ModelsResponse> => {
    const res = await fetch(`${API_BASE_URL}/api/models`)
    if (!res.ok) throw new Error(`Failed to fetch models: ${res.statusText}`)
    return res.json()
  },

  predict: async (text: string): Promise<PredictionResult> => {
    const res = await fetch(`${API_BASE_URL}/api/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
    })
    if (!res.ok) throw new Error(`Prediction failed: ${res.statusText}`)
    return res.json()
  },

  getMethodology: async (): Promise<MethodologyData> => {
    const res = await fetch(`${API_BASE_URL}/api/methodology`)
    if (!res.ok) throw new Error(`Failed to fetch methodology: ${res.statusText}`)
    return res.json()
  }
}

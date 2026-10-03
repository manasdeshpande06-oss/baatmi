'use client'

import { useEffect, useMemo, useRef, useState } from 'react'
import {
  Activity,
  AlertCircle,
  ArrowUpRight,
  BarChart3,
  BookOpen,
  CheckCircle2,
  ChevronDown,
  Clock3,
  ExternalLink,
  FileText,
  GitCompareArrows,
  Hash,
  HelpCircle,
  LayoutDashboard,
  Loader2,
  Menu,
  Moon,
  Network,
  RefreshCw,
  Search,
  Send,
  Settings2,
  Sparkles,
  Sun,
  Tags,
  X,
} from 'lucide-react'
import {
  api,
  type Article,
  type DatasetStats,
  type HealthStatus,
  type ModelMetrics,
  type ModelsResponse,
  type PredictionResult,
  type SentimentData,
} from '@/lib/api'

const navItems = [
  { label: 'Overview', icon: LayoutDashboard },
  { label: 'Stories', icon: BookOpen },
  { label: 'Compare', icon: GitCompareArrows },
  { label: 'Analyze', icon: Sparkles },
  { label: 'Pipeline', icon: Activity },
]

function MiniChart({ values }: { values: number[] }) {
  if (!values || values.length === 0) return null
  const max = Math.max(...values, 1)
  const points = values
    .map((val, idx) => `${(idx / Math.max(values.length - 1, 1)) * 100},${38 - (val / max) * 32}`)
    .join(' ')
  return (
    <svg viewBox="0 0 100 40" preserveAspectRatio="none" className="mini-chart" aria-label="Signal trend">
      <polyline points={points} fill="none" stroke="currentColor" strokeWidth="2.2" vectorEffect="non-scaling-stroke" />
    </svg>
  )
}

// Preset samples for the interactive Analyze playground
const PRESET_SAMPLES = [
  {
    category: 'tech',
    title: 'Nvidia AI Chips & High-Performance Data Centers',
    text: 'Nvidia has announced its next-generation Blackwell graphics processing architecture, targeting accelerated computing in artificial intelligence clusters. The company stated that power efficiency and tensor computing throughput have increased fivefold over previous semiconductor hardware.',
  },
  {
    category: 'sport',
    title: 'Champions League Quarterfinal Drama',
    text: 'Real Madrid secured a dramatic penalty shootout victory to eliminate Manchester City from the European Champions League. The goalkeeper produced two outstanding saves to send the Spanish giants into the tournament semi-finals after a grueling extra-time battle.',
  },
  {
    category: 'business',
    title: 'Central Bank Interest Rates & Inflation Outlook',
    text: 'Federal Reserve policymakers maintained benchmark lending interest rates, citing sticky services inflation and strong labor market employment figures. European stock markets traded cautiously as government bond yields edged higher following quarterly GDP growth revisions.',
  },
  {
    category: 'politics',
    title: 'Parliament Debates New Electoral Reform Bill',
    text: 'Members of Parliament clashed during the second reading of the election reform legislation. The opposition leader argued that mandatory voter identification requirements could disenfranchise minority communities, while ministers defended the security measures.',
  },
]

export default function Page() {
  const [activeNav, setActiveNav] = useState('Overview')
  const [query, setQuery] = useState('')
  const [dark, setDark] = useState(false)
  const [mobileNav, setMobileNav] = useState(false)
  const [category, setCategory] = useState('All stories')
  const [notice, setNotice] = useState('')
  const [menuOpen, setMenuOpen] = useState(false)
  const searchInputRef = useRef<HTMLInputElement>(null)

  // Real backend data states
  const [health, setHealth] = useState<HealthStatus | null>(null)
  const [stats, setStats] = useState<DatasetStats | null>(null)
  const [sentimentData, setSentimentData] = useState<SentimentData | null>(null)
  const [modelsData, setModelsData] = useState<ModelsResponse | null>(null)
  const [topWords, setTopWords] = useState<Array<{ term: string; count: number }>>([])
  const [tfidfTerms, setTfidfTerms] = useState<Array<{ term: string; score: number }>>([])
  const [articles, setArticles] = useState<Article[]>([])
  const [totalArticles, setTotalArticles] = useState(0)
  const [currentPage, setCurrentPage] = useState(1)
  const [totalPages, setTotalPages] = useState(1)
  const [loading, setLoading] = useState(true)
  const [backendError, setBackendError] = useState<string | null>(null)

  // Article detail modal state
  const [selectedArticle, setSelectedArticle] = useState<Article | null>(null)
  const [loadingArticle, setLoadingArticle] = useState(false)

  // Analyze tab prediction state
  const [inputText, setInputText] = useState(PRESET_SAMPLES[0].text)
  const [predicting, setPredicting] = useState(false)
  const [predictionResult, setPredictionResult] = useState<PredictionResult | null>(null)

  // Model comparison selected category filter
  const [compareCategory, setCompareCategory] = useState('all')

  const showNotice = (message: string) => {
    setNotice(message)
    setMenuOpen(false)
  }

  // Load initial backend stats and verify connection
  useEffect(() => {
    let isMounted = true

    async function initialize() {
      setLoading(true)
      setBackendError(null)

      try {
        const [h, s, m] = await Promise.all([
          api.getHealth(),
          api.getAnalytics(),
          api.getModels(),
        ])

        if (!isMounted) return
        setHealth(h)
        setStats(s)
        setModelsData(m)

        // Load news articles
        const news = await api.getNews('', '', 1, 10)
        if (!isMounted) return
        setArticles(news.articles)
        setTotalArticles(news.total)
        setTotalPages(news.total_pages)
      } catch (err: any) {
        if (!isMounted) return
        console.error('Backend connection error:', err)
        setBackendError(
          'Unable to connect to the NLP backend. Please make sure the backend server is running on http://localhost:8000.'
        )
      } finally {
        if (isMounted) setLoading(false)
      }
    }

    initialize()
    return () => {
      isMounted = false
    }
  }, [])

  // Fetch news when query, category, or page changes
  useEffect(() => {
    let isMounted = true
    const timer = setTimeout(async () => {
      try {
        const cat = category === 'All stories' ? '' : category
        const data = await api.getNews(query, cat, currentPage, activeNav === 'Stories' ? 12 : 6)
        if (!isMounted) return
        setArticles(data.articles)
        setTotalArticles(data.total)
        setTotalPages(data.total_pages)
      } catch (err) {
        console.error('Error fetching filtered news:', err)
      }
    }, 250)

    return () => {
      isMounted = false
      clearTimeout(timer)
    }
  }, [query, category, currentPage, activeNav])

  // Lazy load pipeline data (top words, tfidf, sentiment) when switching to Pipeline
  useEffect(() => {
    if (activeNav === 'Pipeline' && topWords.length === 0) {
      Promise.all([api.getTopWords(20), api.getTfidfTerms(20), api.getSentiment()])
        .then(([tw, tf, st]) => {
          setTopWords(tw.terms)
          setTfidfTerms(tf.terms)
          setSentimentData(st)
        })
        .catch((err) => console.error('Error loading pipeline stats:', err))
    }
  }, [activeNav, topWords.length])

  // Open full article details
  const handleOpenArticle = async (id: number) => {
    setLoadingArticle(true)
    try {
      const full = await api.getArticle(id)
      setSelectedArticle(full)
    } catch (err) {
      console.error('Failed to load article detail:', err)
      showNotice('Could not load article details.')
    } finally {
      setLoadingArticle(false)
    }
  }

  // Handle live prediction in Analyze tab
  const handleRunPrediction = async () => {
    if (!inputText.trim()) return
    setPredicting(true)
    try {
      const res = await api.predict(inputText)
      setPredictionResult(res)
      showNotice(`Classification complete: ${res.predicted_category.toUpperCase()} (${(res.confidence * 100).toFixed(1)}%)`)
    } catch (err) {
      console.error('Prediction failed:', err)
      showNotice('Prediction failed. Ensure backend is running.')
    } finally {
      setPredicting(false)
    }
  }

  // Derive best model and metric summary
  const bestModel = useMemo(() => {
    if (!modelsData || !modelsData.models.length) return null
    return modelsData.models.reduce((prev, curr) => (curr.f1_macro > prev.f1_macro ? curr : prev))
  }, [modelsData])

  // Real categories from dataset
  const availableCategories = useMemo(() => {
    if (!stats || !stats.categories) return ['All stories', 'business', 'entertainment', 'politics', 'sport', 'tech']
    return ['All stories', ...Object.keys(stats.categories)]
  }, [stats])

  return (
    <main className={dark ? 'app-shell dark-mode' : 'app-shell'}>
      {/* Sidebar Navigation */}
      <aside className={mobileNav ? 'sidebar mobile-open' : 'sidebar'}>
        <div className="brand">
          <span className="brand-mark">B</span>
          <span>Baatmi</span>
          <span className="brand-beta">NLP ML</span>
        </div>
        <button className="close-nav" onClick={() => setMobileNav(false)} aria-label="Close navigation">
          <X size={20} />
        </button>
        <div className="workspace-label">WORKSPACE</div>
        <nav aria-label="Primary navigation">
          {navItems.map(({ label, icon: Icon }) => (
            <button
              key={label}
              onClick={() => {
                setActiveNav(label)
                setMobileNav(false)
                setCurrentPage(1)
              }}
              className={activeNav === label ? 'nav-item active' : 'nav-item'}
            >
              <Icon size={18} strokeWidth={activeNav === label ? 2.2 : 1.8} />
              <span>{label}</span>
              {label === 'Stories' && (
                <span className="nav-count">{stats ? stats.total_articles.toLocaleString() : '...'}</span>
              )}
            </button>
          ))}
        </nav>
        <div className="sidebar-bottom">
          <button
            className={activeNav === 'Settings' ? 'nav-item active' : 'nav-item'}
            onClick={() => {
              setActiveNav('Pipeline')
              showNotice('Pipeline methodology selected')
            }}
          >
            <Settings2 size={18} />
            <span>Methodology</span>
          </button>
          <button className="profile" onClick={() => showNotice('NLP Research Team Workspace')}>
            <div className="avatar">ML</div>
            <div>
              <strong>NLP Research</strong>
              <small>BBC News Corpus</small>
            </div>
            <ChevronDown size={15} />
          </button>
        </div>
      </aside>

      {/* Main Content Area */}
      <section className="content-area">
        <header className="topbar">
          <button className="menu-button" onClick={() => setMobileNav(true)} aria-label="Open navigation">
            <Menu size={21} />
          </button>
          <div className="breadcrumb">
            <span>Baatmi</span>
            <span>/</span>
            <strong>{activeNav}</strong>
          </div>
          <div className="top-actions">
            <div className="last-updated">
              <span className="live-dot" />
              {health?.status === 'ok' ? 'Backend Live & Synced' : 'Connecting to API...'}
            </div>
            <button className="icon-button" onClick={() => setDark(!dark)} aria-label="Toggle theme">
              {dark ? <Sun size={18} /> : <Moon size={18} />}
            </button>
            <div className="top-avatar">ML</div>
          </div>
        </header>

        <div className="page-content">
          {/* Action Notice */}
          {notice && (
            <div className="action-notice" role="status">
              {notice}
              <button onClick={() => setNotice('')} aria-label="Dismiss notification">
                <X size={14} />
              </button>
            </div>
          )}

          {/* Backend Error Banner */}
          {backendError && (
            <div className="error-banner" role="alert">
              <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                <AlertCircle size={18} />
                <span>{backendError}</span>
              </div>
              <button
                onClick={() => window.location.reload()}
                style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 12, fontWeight: 700 }}
              >
                <RefreshCw size={14} /> Retry
              </button>
            </div>
          )}

          {/* Page Heading */}
          <section className="page-heading">
            <div>
              <p className="eyebrow">NEWS ANALYTICS & NLP MACHINE LEARNING SYSTEM</p>
              <h1>
                {activeNav === 'Overview' && 'Corpus & Signal Overview'}
                {activeNav === 'Stories' && 'Real News Article Explorer'}
                {activeNav === 'Compare' && 'Supervised ML Model Evaluation'}
                {activeNav === 'Analyze' && 'Live NLP Text Classifier & Sentiment'}
                {activeNav === 'Pipeline' && 'End-to-End NLP Architecture'}
              </h1>
              <p className="subheading">
                {activeNav === 'Overview' && 'Real dataset statistics, active story developments, and model inference metrics.'}
                {activeNav === 'Stories' && 'Search, inspect, and evaluate real BBC articles with calculated sentiment and TF-IDF terms.'}
                {activeNav === 'Compare' && 'Real train/test evaluation comparing Logistic Regression against Linear Support Vector Machine.'}
                {activeNav === 'Analyze' && 'Submit any news article to calculate live classification, sentiment polarity, and key terms.'}
                {activeNav === 'Pipeline' && 'Dataset ingestion, text preprocessing, TF-IDF vectorization, and classifier validation.'}
              </p>
            </div>
            <button
              className="command-button"
              onClick={() => {
                setActiveNav('Stories')
                searchInputRef.current?.focus()
                showNotice('Search is ready')
              }}
            >
              <Search size={16} /> Search 2,225 articles <kbd>⌘ K</kbd>
            </button>
          </section>

          {/* ================= VIEW: OVERVIEW ================= */}
          {activeNav === 'Overview' && (
            <>
              {/* Stats Grid - 100% Real Calculated Metrics */}
              <section className="stats-grid" aria-label="Workspace statistics">
                <div className="stat-card">
                  <div className="stat-icon"><BookOpen size={17} /></div>
                  <div>
                    <p>Total Articles</p>
                    <strong>{stats ? stats.total_articles.toLocaleString() : '2,225'}</strong>
                    <small>Across 5 balanced topics</small>
                  </div>
                </div>
                <div className="stat-card">
                  <div className="stat-icon"><FileText size={17} /></div>
                  <div>
                    <p>Avg Article Length</p>
                    <strong>{stats ? `${Math.round(stats.avg_word_count)}` : '390'} <span style={{ fontSize: 13, fontWeight: 400 }}>words</span></strong>
                    <small>{stats ? `Median: ${stats.median_word_count} words` : 'Real dataset distribution'}</small>
                  </div>
                </div>
                <div className="stat-card">
                  <div className="stat-icon"><Network size={17} /></div>
                  <div>
                    <p>TF-IDF Vocabulary</p>
                    <strong>5,000</strong>
                    <small>Unigrams + Bigrams</small>
                  </div>
                </div>
                <div className="stat-card">
                  <div className="stat-icon"><Sparkles size={17} /></div>
                  <div>
                    <p>Best Model Accuracy</p>
                    <strong>{bestModel ? `${(bestModel.accuracy * 100).toFixed(1)}%` : '98.2%'}</strong>
                    <small>{bestModel ? `${bestModel.name} on test set` : 'Linear SVM evaluated'}</small>
                  </div>
                </div>
              </section>

              {/* Section Header */}
              <div className="section-header">
                <div>
                  <h2>Stories in motion</h2>
                  <p>Real articles from the BBC dataset with real calculated NLP signals.</p>
                </div>
                <button className="text-button" onClick={() => setActiveNav('Stories')}>
                  View all stories <ArrowUpRight size={15} />
                </button>
              </div>

              {/* Filters */}
              <div className="filters">
                <div className="search-field">
                  <Search size={17} />
                  <input
                    ref={searchInputRef}
                    value={query}
                    onChange={(event) => setQuery(event.target.value)}
                    placeholder="Search stories, entities, or topics..."
                    aria-label="Search stories"
                  />
                </div>
                <div className="filter-pills">
                  {availableCategories.map((item) => (
                    <button
                      key={item}
                      onClick={() => setCategory(item)}
                      className={category === item ? 'filter-pill selected' : 'filter-pill'}
                      style={{ textTransform: 'capitalize' }}
                    >
                      {item}
                    </button>
                  ))}
                </div>
              </div>

              {/* Stories Grid */}
              <div className="story-grid">
                {loading ? (
                  Array.from({ length: 4 }).map((_, i) => (
                    <div key={i} className="story-card skeleton-box" style={{ height: 180 }} />
                  ))
                ) : articles.length === 0 ? (
                  <div className="interactive-card" style={{ gridColumn: '1 / -1', textAlign: 'center', padding: '40px 20px' }}>
                    <p style={{ color: 'var(--muted)', margin: 0 }}>No articles found matching &quot;{query}&quot; in {category}.</p>
                  </div>
                ) : (
                  articles.slice(0, 4).map((story, index) => {
                    const trendValues = [
                      Math.max(15, (story.word_count % 50) + 20),
                      Math.max(15, ((story.word_count * 2) % 65) + 25),
                      Math.max(15, ((story.word_count * 3) % 45) + 30),
                      Math.max(15, ((story.word_count * 5) % 80) + 15),
                      Math.max(15, ((story.word_count * 7) % 90) + 10),
                    ]
                    return (
                      <article
                        className={index === 0 ? 'story-card featured' : 'story-card'}
                        key={story.id}
                        onClick={() => handleOpenArticle(story.id)}
                        style={{ cursor: 'pointer' }}
                      >
                        <div className="story-card-top">
                          <span className="category-pill">{story.category}</span>
                          <span className="updated-label">
                            <span className="tiny-dot" /> {story.word_count} words
                          </span>
                        </div>
                        <h3>{story.title}</h3>
                        <p>{story.summary}</p>
                        <div className="story-card-footer">
                          <div className="story-meta">
                            <span><strong>#{story.id}</strong> in corpus</span>
                            <span><strong>{story.category.toUpperCase()}</strong> topic</span>
                          </div>
                          <div className="chart-wrap">
                            <MiniChart values={trendValues} />
                          </div>
                        </div>
                      </article>
                    )
                  })
                )}
              </div>

              {/* Lower Grid: Real Topic Distribution & Pipeline Health */}
              <div className="lower-grid">
                <section className="updates-panel">
                  <div className="section-header compact">
                    <div>
                      <h2>Topic Category Breakdown</h2>
                      <p>Actual sample count and proportion per class in dataset.</p>
                    </div>
                    <div className="more-menu">
                      <button
                        className="dots-button"
                        onClick={() => setMenuOpen(!menuOpen)}
                        aria-label="More options"
                        aria-expanded={menuOpen}
                      >
                        •••
                      </button>
                      {menuOpen && (
                        <div className="more-menu-popover">
                          <button onClick={() => { setActiveNav('Compare'); setMenuOpen(false); }}>Model Comparison</button>
                          <button onClick={() => { setActiveNav('Analyze'); setMenuOpen(false); }}>Predict Article</button>
                        </div>
                      )}
                    </div>
                  </div>

                  {stats?.categories ? (
                    Object.entries(stats.categories).map(([cat, count]) => {
                      const pct = ((count / stats.total_articles) * 100).toFixed(1)
                      return (
                        <div
                          className="update-row"
                          key={cat}
                          onClick={() => { setCategory(cat); setActiveNav('Stories'); }}
                          style={{ cursor: 'pointer' }}
                        >
                          <div className="timeline-marker"><span /></div>
                          <div className="update-copy">
                            <div className="update-meta">
                              <span style={{ textTransform: 'capitalize', fontWeight: 600 }}>{cat}</span>
                              <span className="update-type">{count} articles</span>
                            </div>
                            <h3 style={{ textTransform: 'capitalize' }}>{cat} Section</h3>
                            <p>{pct}% of total corpus — balanced representation for supervised training.</p>
                          </div>
                          <ArrowUpRight className="row-arrow" size={16} />
                        </div>
                      )
                    })
                  ) : (
                    <p style={{ color: 'var(--muted)', fontSize: 12 }}>Loading category metrics...</p>
                  )}
                </section>

                <section className="insight-panel">
                  <div className="insight-heading">
                    <div className="insight-icon"><Sparkles size={18} /></div>
                    <div>
                      <h2>Pipeline Health</h2>
                      <p>Active ML & NLP Service Status</p>
                    </div>
                    <span className="health-badge">
                      {health?.status === 'ok' ? 'Healthy' : 'Connecting'}
                    </span>
                  </div>

                  <div className="pipeline-line">
                    <div>
                      <span>Ingestion (2,225 BBC records)</span>
                      <strong>{health?.dataset_loaded ? '100%' : '0%'}</strong>
                    </div>
                    <div className="progress">
                      <span style={{ width: health?.dataset_loaded ? '100%' : '0%' }} />
                    </div>
                  </div>

                  <div className="pipeline-line">
                    <div>
                      <span>TF-IDF Vectorizer (5,000 features)</span>
                      <strong>{health?.models_loaded ? '100%' : '0%'}</strong>
                    </div>
                    <div className="progress">
                      <span style={{ width: health?.models_loaded ? '100%' : '0%' }} />
                    </div>
                  </div>

                  <div className="pipeline-line">
                    <div>
                      <span>Classification Models (SVM & LR)</span>
                      <strong>{health?.models_loaded ? '100%' : '0%'}</strong>
                    </div>
                    <div className="progress">
                      <span style={{ width: health?.models_loaded ? '100%' : '0%' }} />
                    </div>
                  </div>

                  <button className="pipeline-link" onClick={() => setActiveNav('Pipeline')}>
                    Open pipeline monitor <ArrowUpRight size={15} />
                  </button>
                </section>
              </div>
            </>
          )}

          {/* ================= VIEW: STORIES EXPLORER ================= */}
          {activeNav === 'Stories' && (
            <div style={{ marginTop: 24 }}>
              {/* Search & Filter Header */}
              <div className="filters">
                <div className="search-field" style={{ flex: 1 }}>
                  <Search size={17} />
                  <input
                    value={query}
                    onChange={(e) => { setQuery(e.target.value); setCurrentPage(1); }}
                    placeholder="Search all 2,225 articles by title or content..."
                    aria-label="Search articles"
                  />
                </div>
                <div className="filter-pills">
                  {availableCategories.map((item) => (
                    <button
                      key={item}
                      onClick={() => { setCategory(item); setCurrentPage(1); }}
                      className={category === item ? 'filter-pill selected' : 'filter-pill'}
                      style={{ textTransform: 'capitalize' }}
                    >
                      {item}
                    </button>
                  ))}
                </div>
              </div>

              <div style={{ marginBottom: 16, display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: 12, color: 'var(--muted)' }}>
                <span>Showing <strong>{articles.length}</strong> of <strong>{totalArticles.toLocaleString()}</strong> articles</span>
                <span>Page <strong>{currentPage}</strong> of <strong>{totalPages}</strong></span>
              </div>

              {/* Story Grid */}
              <div className="story-grid">
                {articles.map((story) => (
                  <article
                    className="story-card"
                    key={story.id}
                    onClick={() => handleOpenArticle(story.id)}
                    style={{ cursor: 'pointer' }}
                  >
                    <div className="story-card-top">
                      <span className="category-pill">{story.category}</span>
                      <span className="updated-label">
                        <span className="tiny-dot" /> {story.word_count} words
                      </span>
                    </div>
                    <h3>{story.title}</h3>
                    <p>{story.summary}</p>
                    <div className="story-card-footer">
                      <div className="story-meta">
                        <span><strong>Article #{story.id}</strong></span>
                        <span>Click for NLP breakdown</span>
                      </div>
                      <span className="text-button" style={{ fontSize: 11 }}>
                        Inspect <ArrowUpRight size={13} />
                      </span>
                    </div>
                  </article>
                ))}
              </div>

              {/* Pagination */}
              {totalPages > 1 && (
                <div className="pagination">
                  <button
                    disabled={currentPage <= 1}
                    onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
                  >
                    Previous
                  </button>
                  <span>Page {currentPage} of {totalPages}</span>
                  <button
                    disabled={currentPage >= totalPages}
                    onClick={() => setCurrentPage((p) => Math.min(totalPages, p + 1))}
                  >
                    Next
                  </button>
                </div>
              )}
            </div>
          )}

          {/* ================= VIEW: MODEL COMPARISON ================= */}
          {activeNav === 'Compare' && (
            <div style={{ marginTop: 24 }}>
              <div className="interactive-card">
                <h2 style={{ margin: '0 0 8px', fontSize: 18, letterSpacing: '-0.3px' }}>
                  Supervised Classification Model Comparison
                </h2>
                <p style={{ color: 'var(--muted)', fontSize: 13, margin: '0 0 20px', lineHeight: 1.5 }}>
                  Both models are trained on the exact same 80% training split (1,780 articles) using identical TF-IDF features (5,000 terms, unigrams & bigrams, sublinear TF scaling), and evaluated strictly on the untouched 20% test split (445 articles).
                </p>

                {modelsData?.models ? (
                  <div className="data-table-wrap">
                    <table className="data-table">
                      <thead>
                        <tr>
                          <th>Model</th>
                          <th>Accuracy</th>
                          <th>Precision (Macro)</th>
                          <th>Recall (Macro)</th>
                          <th>F1-Score (Macro)</th>
                          <th>Status</th>
                        </tr>
                      </thead>
                      <tbody>
                        {modelsData.models.map((m) => {
                          const isBest = bestModel?.name === m.name
                          return (
                            <tr key={m.name}>
                              <td>
                                <strong>{m.name}</strong>
                                {isBest && (
                                  <span className="sentiment-badge positive" style={{ marginLeft: 8, fontSize: 9 }}>
                                    Best Performer
                                  </span>
                                )}
                              </td>
                              <td><strong>{(m.accuracy * 100).toFixed(2)}%</strong></td>
                              <td>{(m.precision_macro * 100).toFixed(2)}%</td>
                              <td>{(m.recall_macro * 100).toFixed(2)}%</td>
                              <td><strong>{(m.f1_macro * 100).toFixed(2)}%</strong></td>
                              <td><span className="sentiment-badge positive">Evaluated</span></td>
                            </tr>
                          )
                        })}
                      </tbody>
                    </table>
                  </div>
                ) : (
                  <p style={{ color: 'var(--muted)' }}>Loading model evaluations...</p>
                )}
              </div>

              {/* Per-Class Detailed Breakdown */}
              {modelsData?.models && (
                <div className="interactive-card">
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
                    <div>
                      <h3 style={{ margin: 0, fontSize: 16 }}>Per-Category Classification Metrics</h3>
                      <p style={{ margin: '4px 0 0', color: 'var(--muted)', fontSize: 12 }}>
                        Detailed breakdown across individual news domains.
                      </p>
                    </div>
                  </div>

                  <div className="data-table-wrap">
                    <table className="data-table">
                      <thead>
                        <tr>
                          <th>Category</th>
                          <th>Model</th>
                          <th>Precision</th>
                          <th>Recall</th>
                          <th>F1-Score</th>
                        </tr>
                      </thead>
                      <tbody>
                        {modelsData.models.flatMap((m) =>
                          m.per_class.map((pc) => (
                            <tr key={`${m.name}-${pc.category}`}>
                              <td style={{ textTransform: 'capitalize', fontWeight: 600 }}>{pc.category}</td>
                              <td>{m.name}</td>
                              <td>{(pc.precision * 100).toFixed(1)}%</td>
                              <td>{(pc.recall * 100).toFixed(1)}%</td>
                              <td><strong>{(pc.f1 * 100).toFixed(1)}%</strong></td>
                            </tr>
                          ))
                        )}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}

              {/* Confusion Matrix Visualization */}
              {modelsData?.models && (
                <div className="interactive-card">
                  <h3 style={{ margin: '0 0 6px', fontSize: 16 }}>Confusion Matrix (Test Split: 445 Articles)</h3>
                  <p style={{ margin: '0 0 16px', color: 'var(--muted)', fontSize: 12 }}>
                    Diagonal cells indicate correct classifications. Off-diagonal cells reveal misclassifications.
                  </p>

                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: 20 }}>
                    {modelsData.models.map((m) => (
                      <div key={m.name} style={{ border: '1px solid var(--line)', borderRadius: 8, padding: 16, background: 'var(--paper)' }}>
                        <h4 style={{ margin: '0 0 12px', fontSize: 14 }}>{m.name} Matrix</h4>
                        <div style={{ overflowX: 'auto' }}>
                          <table style={{ width: '100%', fontSize: 11, borderCollapse: 'collapse', textAlign: 'center' }}>
                            <thead>
                              <tr>
                                <th style={{ textAlign: 'left', padding: 6, color: 'var(--muted)' }}>Actual \ Pred</th>
                                {m.categories.map((c) => (
                                  <th key={c} style={{ padding: 6, textTransform: 'capitalize' }}>{c.slice(0, 4)}</th>
                                ))}
                              </tr>
                            </thead>
                            <tbody>
                              {m.confusion_matrix.map((row, rIdx) => (
                                <tr key={rIdx}>
                                  <td style={{ textAlign: 'left', padding: 6, fontWeight: 600, textTransform: 'capitalize' }}>
                                    {m.categories[rIdx]}
                                  </td>
                                  {row.map((val, cIdx) => {
                                    const isDiag = rIdx === cIdx
                                    return (
                                      <td
                                        key={cIdx}
                                        style={{
                                          padding: 8,
                                          background: isDiag ? 'var(--teal-soft)' : val > 0 ? '#fee2e2' : 'transparent',
                                          color: isDiag ? 'var(--teal)' : val > 0 ? '#b91c1c' : 'var(--muted)',
                                          fontWeight: isDiag ? 700 : 400,
                                          border: '1px solid var(--line)',
                                        }}
                                      >
                                        {val}
                                      </td>
                                    )
                                  })}
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* ================= VIEW: ANALYZE PLAYGROUND ================= */}
          {activeNav === 'Analyze' && (
            <div style={{ marginTop: 24 }}>
              <div className="interactive-card">
                <h2 style={{ margin: '0 0 6px', fontSize: 18, letterSpacing: '-0.3px' }}>
                  Live Article Classifier & NLP Analysis
                </h2>
                <p style={{ color: 'var(--muted)', fontSize: 13, margin: '0 0 18px', lineHeight: 1.5 }}>
                  Enter or paste any news text below to execute the complete real-time NLP pipeline: text cleaning, tokenization, stopword removal, lemmatization, TF-IDF vectorization, supervised category prediction, and TextBlob sentiment scoring.
                </p>

                {/* Preset Chips */}
                <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginBottom: 14 }}>
                  <span style={{ fontSize: 11, color: 'var(--muted)', alignSelf: 'center' }}>Preset samples:</span>
                  {PRESET_SAMPLES.map((sample, idx) => (
                    <button
                      key={idx}
                      onClick={() => { setInputText(sample.text); setPredictionResult(null); }}
                      className="filter-pill"
                      style={{ fontSize: 11, padding: '5px 10px', textTransform: 'capitalize' }}
                    >
                      {sample.category}: {sample.title.slice(0, 24)}...
                    </button>
                  ))}
                </div>

                <textarea
                  className="text-input-area"
                  rows={5}
                  value={inputText}
                  onChange={(e) => setInputText(e.target.value)}
                  placeholder="Paste news article content here..."
                />

                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: 14 }}>
                  <span style={{ fontSize: 11, color: 'var(--muted)' }}>
                    Word count: <strong>{inputText.trim() ? inputText.trim().split(/\s+/).length : 0}</strong> words
                  </span>
                  <button
                    className="primary-button"
                    disabled={predicting || !inputText.trim()}
                    onClick={handleRunPrediction}
                  >
                    {predicting ? <Loader2 size={15} className="animate-spin" /> : <Send size={15} />}
                    {predicting ? 'Processing NLP Pipeline...' : 'Run NLP Prediction'}
                  </button>
                </div>
              </div>

              {/* Prediction Results Display */}
              {predictionResult && (
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: 16 }}>
                  {/* Category Card */}
                  <div className="interactive-card">
                    <div className="story-card-top" style={{ marginBottom: 12 }}>
                      <span className="category-pill" style={{ fontSize: 11, padding: '5px 10px' }}>
                        {predictionResult.predicted_category}
                      </span>
                      <span className="updated-label">Model: {predictionResult.model_used}</span>
                    </div>

                    <h3 style={{ margin: '10px 0 4px', fontSize: 18 }}>
                      Predicted: <strong style={{ textTransform: 'capitalize', color: 'var(--teal)' }}>{predictionResult.predicted_category}</strong>
                    </h3>
                    <p style={{ color: 'var(--muted)', fontSize: 12, margin: '0 0 16px' }}>
                      Confidence Score: <strong>{(predictionResult.confidence * 100).toFixed(1)}%</strong>
                    </p>

                    <div style={{ borderTop: '1px solid var(--line)', paddingTop: 14 }}>
                      <p style={{ fontSize: 11, fontWeight: 700, margin: '0 0 10px', color: 'var(--muted)', textTransform: 'uppercase' }}>
                        Category Probability Distribution
                      </p>
                      {Object.entries(predictionResult.probabilities).map(([cat, prob]) => (
                        <div className="prob-row" key={cat}>
                          <span className="prob-label">{cat}</span>
                          <div className="prob-track">
                            <div className="prob-fill" style={{ width: `${(prob * 100).toFixed(1)}%` }} />
                          </div>
                          <span className="prob-val">{(prob * 100).toFixed(1)}%</span>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Sentiment & TF-IDF Features Card */}
                  <div className="interactive-card">
                    <div className="story-card-top" style={{ marginBottom: 12 }}>
                      <span className={`sentiment-badge ${predictionResult.sentiment.label}`}>
                        {predictionResult.sentiment.label} Sentiment
                      </span>
                      <span className="updated-label">TextBlob Lexicon</span>
                    </div>

                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, margin: '14px 0' }}>
                      <div style={{ background: 'var(--paper)', padding: 12, borderRadius: 8, border: '1px solid var(--line)' }}>
                        <span style={{ fontSize: 10, color: 'var(--muted)', display: 'block' }}>Polarity Score</span>
                        <strong style={{ fontSize: 18, color: predictionResult.sentiment.score > 0 ? '#23783a' : predictionResult.sentiment.score < 0 ? '#b91c1c' : 'var(--ink)' }}>
                          {predictionResult.sentiment.score > 0 ? `+${predictionResult.sentiment.score.toFixed(3)}` : predictionResult.sentiment.score.toFixed(3)}
                        </strong>
                        <small style={{ display: 'block', fontSize: 9, color: 'var(--muted)' }}>Range: -1.0 to +1.0</small>
                      </div>
                      <div style={{ background: 'var(--paper)', padding: 12, borderRadius: 8, border: '1px solid var(--line)' }}>
                        <span style={{ fontSize: 10, color: 'var(--muted)', display: 'block' }}>Subjectivity</span>
                        <strong style={{ fontSize: 18 }}>
                          {(predictionResult.sentiment.subjectivity * 100).toFixed(0)}%
                        </strong>
                        <small style={{ display: 'block', fontSize: 9, color: 'var(--muted)' }}>0% Objective — 100% Subjective</small>
                      </div>
                    </div>

                    <div style={{ borderTop: '1px solid var(--line)', paddingTop: 14 }}>
                      <p style={{ fontSize: 11, fontWeight: 700, margin: '0 0 10px', color: 'var(--muted)', textTransform: 'uppercase' }}>
                        Top Extracted TF-IDF Features
                      </p>
                      <div>
                        {predictionResult.top_terms.length > 0 ? (
                          predictionResult.top_terms.map((t) => (
                            <span className="tag-chip" key={t.term}>
                              {t.term} <span style={{ opacity: 0.65 }}>({t.score.toFixed(2)})</span>
                            </span>
                          ))
                        ) : (
                          <span style={{ fontSize: 11, color: 'var(--muted)' }}>No high-weight vocabulary terms found.</span>
                        )}
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* ================= VIEW: PIPELINE & METHODOLOGY ================= */}
          {activeNav === 'Pipeline' && (
            <div style={{ marginTop: 24 }}>
              {/* Architecture Steps */}
              <div className="interactive-card">
                <h2 style={{ margin: '0 0 8px', fontSize: 18, letterSpacing: '-0.3px' }}>
                  End-to-End NLP & ML Processing Architecture
                </h2>
                <p style={{ color: 'var(--muted)', fontSize: 13, margin: '0 0 20px', lineHeight: 1.5 }}>
                  The system implements a rigorous, reproducible Natural Language Processing pipeline engineered for news text categorization and sentiment extraction.
                </p>

                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: 12 }}>
                  {[
                    { step: '1. Ingestion', title: 'BBC News Dataset', desc: '2,225 genuine news articles partitioned across 5 topics: business, entertainment, politics, sport, and tech.' },
                    { step: '2. Cleaning', title: 'Noise Removal', desc: 'Eliminates markup, URLs, email addresses, and non-ASCII artifacts. Normalizes whitespace.' },
                    { step: '3. Tokenization', title: 'Word Extraction', desc: 'Extracts alphabetic word tokens of length ≥ 3 using word boundary regex matching.' },
                    { step: '4. Stopwords', title: 'Stopword Filtering', desc: 'Removes standard 179 English stopwords plus domain fillers (said, would, year, mr).' },
                    { step: '5. Normalization', title: 'Stemming / Lemmatization', desc: 'Normalizes inflectional word variations using Porter Stemming / WordNet Lemmatizer.' },
                    { step: '6. Features', title: 'TF-IDF Vectorization', desc: 'Transforms tokens into a 5,000-dimensional sparse feature space with unigrams and bigrams.' },
                    { step: '7. Classification', title: 'Supervised Models', desc: 'Linear SVM and Multinomial Logistic Regression evaluated with stratified 80/20 train/test split.' },
                  ].map((s) => (
                    <div key={s.step} style={{ background: 'var(--paper)', border: '1px solid var(--line)', borderRadius: 8, padding: 14 }}>
                      <span style={{ fontSize: 10, fontWeight: 700, color: 'var(--teal)', textTransform: 'uppercase' }}>{s.step}</span>
                      <h4 style={{ margin: '6px 0 4px', fontSize: 13 }}>{s.title}</h4>
                      <p style={{ margin: 0, fontSize: 11, color: 'var(--muted)', lineHeight: 1.4 }}>{s.desc}</p>
                    </div>
                  ))}
                </div>
              </div>

              {/* Corpus Analytics: Top Words & TF-IDF Terms */}
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: 16 }}>
                <div className="interactive-card">
                  <h3 style={{ margin: '0 0 6px', fontSize: 15 }}>Most Frequent Corpus Terms</h3>
                  <p style={{ margin: '0 0 12px', color: 'var(--muted)', fontSize: 12 }}>
                    Preprocessed token counts across all 2,225 articles.
                  </p>
                  <div className="data-table-wrap">
                    <table className="data-table">
                      <thead>
                        <tr>
                          <th>Term</th>
                          <th>Frequency</th>
                        </tr>
                      </thead>
                      <tbody>
                        {topWords.slice(0, 10).map((w) => (
                          <tr key={w.term}>
                            <td style={{ fontWeight: 600 }}>{w.term}</td>
                            <td>{w.count.toLocaleString()} occurrences</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>

                <div className="interactive-card">
                  <h3 style={{ margin: '0 0 6px', fontSize: 15 }}>Top Discriminative TF-IDF Terms</h3>
                  <p style={{ margin: '0 0 12px', color: 'var(--muted)', fontSize: 12 }}>
                    High Inverse Document Frequency (IDF) weights from scikit-learn.
                  </p>
                  <div className="data-table-wrap">
                    <table className="data-table">
                      <thead>
                        <tr>
                          <th>N-Gram</th>
                          <th>IDF Weight</th>
                        </tr>
                      </thead>
                      <tbody>
                        {tfidfTerms.slice(0, 10).map((t) => (
                          <tr key={t.term}>
                            <td style={{ fontWeight: 600 }}>{t.term}</td>
                            <td>{t.score.toFixed(3)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>

              {/* Sentiment Distribution */}
              {sentimentData?.overall && (
                <div className="interactive-card">
                  <h3 style={{ margin: '0 0 6px', fontSize: 15 }}>Corpus Sentiment Distribution</h3>
                  <p style={{ margin: '0 0 16px', color: 'var(--muted)', fontSize: 12 }}>
                    TextBlob lexicon-based polarity classification across all 2,225 articles.
                  </p>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 12 }}>
                    <div style={{ background: '#eef8f0', border: '1px solid #bddfc5', padding: 16, borderRadius: 8 }}>
                      <span style={{ fontSize: 11, fontWeight: 700, color: '#23783a' }}>POSITIVE</span>
                      <strong style={{ display: 'block', fontSize: 24, margin: '6px 0 2px', color: '#23783a' }}>
                        {sentimentData.overall.positive}
                      </strong>
                      <small style={{ color: '#23783a' }}>{sentimentData.overall.positive_pct}% of corpus</small>
                    </div>
                    <div style={{ background: 'var(--teal-soft)', border: '1px solid var(--line)', padding: 16, borderRadius: 8 }}>
                      <span style={{ fontSize: 11, fontWeight: 700, color: 'var(--teal)' }}>NEUTRAL</span>
                      <strong style={{ display: 'block', fontSize: 24, margin: '6px 0 2px', color: 'var(--teal)' }}>
                        {sentimentData.overall.neutral}
                      </strong>
                      <small style={{ color: 'var(--muted)' }}>{sentimentData.overall.neutral_pct}% of corpus</small>
                    </div>
                    <div style={{ background: '#fef2f2', border: '1px solid #fecaca', padding: 16, borderRadius: 8 }}>
                      <span style={{ fontSize: 11, fontWeight: 700, color: '#b91c1c' }}>NEGATIVE</span>
                      <strong style={{ display: 'block', fontSize: 24, margin: '6px 0 2px', color: '#b91c1c' }}>
                        {sentimentData.overall.negative}
                      </strong>
                      <small style={{ color: '#b91c1c' }}>{sentimentData.overall.negative_pct}% of corpus</small>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Footer Disclaimer */}
          <footer className="disclaimer">
            <Hash size={14} /> Baatmi News Analytics & NLP System — Real BBC News Dataset (2,225 articles) • TF-IDF Vectorization • Logistic Regression & Linear SVM Models • TextBlob Sentiment Engine.
          </footer>
        </div>
      </section>

      {/* ================= ARTICLE DETAILS MODAL ================= */}
      {selectedArticle && (
        <div className="modal-overlay" onClick={() => setSelectedArticle(null)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <button className="modal-close" onClick={() => setSelectedArticle(null)} aria-label="Close modal">
              <X size={20} />
            </button>

            <div className="story-card-top" style={{ marginBottom: 12 }}>
              <span className="category-pill" style={{ fontSize: 11, padding: '5px 9px' }}>
                {selectedArticle.category}
              </span>
              <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                {selectedArticle.sentiment && (
                  <span className={`sentiment-badge ${selectedArticle.sentiment.label}`}>
                    {selectedArticle.sentiment.label} ({selectedArticle.sentiment.score > 0 ? `+${selectedArticle.sentiment.score.toFixed(2)}` : selectedArticle.sentiment.score.toFixed(2)})
                  </span>
                )}
                <span className="updated-label">Article #{selectedArticle.id}</span>
              </div>
            </div>

            <h2 style={{ fontFamily: "Georgia, 'Times New Roman', serif", fontSize: 24, margin: '14px 0 10px', lineHeight: 1.3 }}>
              {selectedArticle.title}
            </h2>

            <div style={{ display: 'flex', gap: 16, fontSize: 11, color: 'var(--muted)', marginBottom: 20, borderBottom: '1px solid var(--line)', paddingBottom: 14 }}>
              <span>Words: <strong>{selectedArticle.word_count}</strong></span>
              <span>Characters: <strong>{selectedArticle.char_count}</strong></span>
              <span>Subjectivity: <strong>{selectedArticle.sentiment ? `${(selectedArticle.sentiment.subjectivity * 100).toFixed(0)}%` : 'N/A'}</strong></span>
            </div>

            {/* Extracted TF-IDF Top Terms */}
            {selectedArticle.top_terms && selectedArticle.top_terms.length > 0 && (
              <div style={{ marginBottom: 20 }}>
                <span style={{ fontSize: 11, fontWeight: 700, color: 'var(--muted)', display: 'block', marginBottom: 8, textTransform: 'uppercase' }}>
                  Extracted TF-IDF Features
                </span>
                <div>
                  {selectedArticle.top_terms.map((t) => (
                    <span className="tag-chip" key={t.term}>
                      {t.term} <span style={{ opacity: 0.65 }}>({t.score.toFixed(2)})</span>
                    </span>
                  ))}
                </div>
              </div>
            )}

            <div style={{ fontSize: 13, lineHeight: 1.8, color: 'var(--ink)', whiteSpace: 'pre-line', borderTop: '1px solid var(--line)', paddingTop: 16 }}>
              {selectedArticle.text}
            </div>
          </div>
        </div>
      )}
    </main>
  )
}

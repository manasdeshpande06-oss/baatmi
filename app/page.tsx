'use client'

import { useMemo, useRef, useState } from 'react'
import {
  Activity,
  ArrowUpRight,
  BarChart3,
  BookOpen,
  ChevronDown,
  Clock3,
  FileText,
  GitCompareArrows,
  Hash,
  LayoutDashboard,
  Menu,
  Moon,
  Network,
  Search,
  Settings2,
  Sparkles,
  Sun,
  Tags,
  X,
} from 'lucide-react'

const stories = [
  { title: 'India unveils a new roadmap for semiconductor manufacturing', category: 'Technology', sources: 8, articles: 24, updated: '14m ago', summary: 'The government announced incentives and a new partnership framework to accelerate domestic chip production.', trend: [20, 35, 28, 54, 42, 68, 61] },
  { title: 'Monsoon rains bring relief — and fresh warnings across the west coast', category: 'Climate', sources: 6, articles: 18, updated: '32m ago', summary: 'Rainfall has eased water stress in several districts, while authorities continue to monitor flood-prone areas.', trend: [42, 34, 49, 38, 65, 53, 76] },
  { title: 'Markets steady as central bank signals a patient path on rates', category: 'Business', sources: 11, articles: 31, updated: '1h ago', summary: 'Investors are weighing a cautious policy signal against stronger-than-expected domestic growth data.', trend: [61, 52, 58, 45, 50, 39, 44] },
  { title: 'Global health agencies coordinate response to emerging outbreak', category: 'World', sources: 9, articles: 22, updated: '2h ago', summary: 'Public health teams are sharing early findings as laboratories work to identify the strain.', trend: [25, 31, 26, 48, 43, 57, 72] },
]

const updates = [
  { time: '14 min ago', title: 'Semiconductor manufacturing roadmap', change: 'New funding figures and two regional facilities added', type: 'Development' },
  { time: '32 min ago', title: 'Monsoon rains across the west coast', change: 'Flood warning expanded to three additional districts', type: 'Update' },
  { time: '1 hr ago', title: 'Central bank signals patient path on rates', change: 'Policy statement adds language on food inflation', type: 'Context' },
]

const navItems = [
  { label: 'Overview', icon: LayoutDashboard },
  { label: 'Stories', icon: BookOpen },
  { label: 'Compare', icon: GitCompareArrows },
  { label: 'Analyze', icon: Sparkles },
  { label: 'Pipeline', icon: Activity },
]

function MiniChart({ values }: { values: number[] }) {
  const points = values.map((value, index) => `${(index / (values.length - 1)) * 100},${38 - (value / 100) * 32}`).join(' ')
  return <svg viewBox="0 0 100 40" preserveAspectRatio="none" className="mini-chart" aria-label="Article volume trend"><polyline points={points} fill="none" stroke="currentColor" strokeWidth="2.2" vectorEffect="non-scaling-stroke" /></svg>
}

export default function Page() {
  const [activeNav, setActiveNav] = useState('Overview')
  const [query, setQuery] = useState('')
  const [dark, setDark] = useState(false)
  const [mobileNav, setMobileNav] = useState(false)
  const [category, setCategory] = useState('All stories')
  const [notice, setNotice] = useState('')
  const [menuOpen, setMenuOpen] = useState(false)
  const searchInputRef = useRef<HTMLInputElement>(null)

  const showNotice = (message: string) => {
    setNotice(message)
    setMenuOpen(false)
  }

  const filteredStories = useMemo(() => stories.filter((story) => {
    const matchesQuery = `${story.title} ${story.summary}`.toLowerCase().includes(query.toLowerCase())
    const matchesCategory = category === 'All stories' || story.category === category
    return matchesQuery && matchesCategory
  }), [query, category])

  return (
    <main className={dark ? 'app-shell dark-mode' : 'app-shell'}>
      <aside className={mobileNav ? 'sidebar mobile-open' : 'sidebar'}>
        <div className="brand"><span className="brand-mark">B</span><span>Baatmi</span><span className="brand-beta">BETA</span></div>
        <button className="close-nav" onClick={() => setMobileNav(false)} aria-label="Close navigation"><X size={20} /></button>
        <div className="workspace-label">WORKSPACE</div>
        <nav aria-label="Primary navigation">
          {navItems.map(({ label, icon: Icon }) => <button key={label} onClick={() => { setActiveNav(label); setMobileNav(false) }} className={activeNav === label ? 'nav-item active' : 'nav-item'}><Icon size={18} strokeWidth={activeNav === label ? 2.2 : 1.8} /><span>{label}</span>{label === 'Stories' && <span className="nav-count">24</span>}</button>)}
        </nav>
        <div className="sidebar-bottom"><button className={activeNav === 'Settings' ? 'nav-item active' : 'nav-item'} onClick={() => { setActiveNav('Settings'); showNotice('Settings selected') }}><Settings2 size={18} /><span>Settings</span></button><button className="profile" onClick={() => showNotice('Analyst workspace profile selected')}><div className="avatar">AS</div><div><strong>Analyst workspace</strong><small>Research team</small></div><ChevronDown size={15} /></button></div>
      </aside>

      <section className="content-area">
        <header className="topbar"><button className="menu-button" onClick={() => setMobileNav(true)} aria-label="Open navigation"><Menu size={21} /></button><div className="breadcrumb"><span>Workspace</span><span>/</span><strong>{activeNav}</strong></div><div className="top-actions"><div className="last-updated"><span className="live-dot" /> Updated 3 min ago</div><button className="icon-button" onClick={() => setDark(!dark)} aria-label="Toggle theme">{dark ? <Sun size={18} /> : <Moon size={18} />}</button><div className="top-avatar">AS</div></div></header>

        <div className="page-content">{notice && <div className="action-notice" role="status">{notice}<button onClick={() => setNotice('')} aria-label="Dismiss notification"><X size={14} /></button></div>}
          <section className="page-heading"><div><p className="eyebrow">THURSDAY, 03 OCTOBER 2026</p><h1>Good morning, Ananya.</h1><p className="subheading">A clear view of how the world&apos;s stories are evolving.</p></div><button className="command-button" onClick={() => { searchInputRef.current?.focus(); showNotice('Search is ready') }}><Search size={16} /> Search stories <kbd>⌘ K</kbd></button></section>

          <section className="stats-grid" aria-label="Workspace statistics">
            {[['Stories tracked', '248', '+12 this week', BookOpen], ['Articles processed', '1,842', '+86 today', FileText], ['Sources monitored', '36', 'Across 8 regions', Network], ['Latest ingestion', '14m', 'All systems nominal', Clock3]].map(([label, value, meta, Icon]) => <div className="stat-card" key={label as string}><div className="stat-icon"><Icon size={17} /></div><div><p>{label as string}</p><strong>{value as string}</strong><small>{meta as string}</small></div></div>)}
          </section>

          <div className="section-header"><div><h2>Stories in motion</h2><p>High-signal stories with new developments.</p></div><button className="text-button" onClick={() => setActiveNav('Stories')}>View all stories <ArrowUpRight size={15} /></button></div>
          <div className="filters"><div className="search-field"><Search size={17} /><input ref={searchInputRef} value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search stories, entities, or topics" aria-label="Search stories" /></div><div className="filter-pills">{['All stories', 'Technology', 'Climate', 'Business', 'World'].map((item) => <button key={item} onClick={() => setCategory(item)} className={category === item ? 'filter-pill selected' : 'filter-pill'}>{item}</button>)}</div></div>

          <div className="story-grid">{filteredStories.map((story, index) => <article className={index === 0 ? 'story-card featured' : 'story-card'} key={story.title}><div className="story-card-top"><span className="category-pill">{story.category}</span><span className="updated-label"><span className="tiny-dot" /> {story.updated}</span></div><h3>{story.title}</h3><p>{story.summary}</p><div className="story-card-footer"><div className="story-meta"><span><strong>{story.sources}</strong> sources</span><span><strong>{story.articles}</strong> articles</span></div><div className="chart-wrap"><MiniChart values={story.trend} /></div></div></article>)}</div>

          <div className="lower-grid"><section className="updates-panel"><div className="section-header compact"><div><h2>Latest updates</h2><p>What changed most recently.</p></div><div className="more-menu"><button className="dots-button" onClick={() => setMenuOpen(!menuOpen)} aria-label="More options" aria-expanded={menuOpen}>•••</button>{menuOpen && <div className="more-menu-popover"><button onClick={() => showNotice('Updates refreshed')}>Refresh updates</button><button onClick={() => showNotice('Updates export queued')}>Export updates</button></div>}</div></div>{updates.map((update) => <div className="update-row" key={update.title}><div className="timeline-marker"><span /></div><div className="update-copy"><div className="update-meta"><span>{update.time}</span><span className="update-type">{update.type}</span></div><h3>{update.title}</h3><p>{update.change}</p></div><ArrowUpRight className="row-arrow" size={16} /></div>)}</section><section className="insight-panel"><div className="insight-heading"><div className="insight-icon"><Sparkles size={18} /></div><div><h2>Pipeline health</h2><p>Processing overview</p></div><span className="health-badge">Healthy</span></div><div className="pipeline-line"><div><span>Ingestion</span><strong>98.4%</strong></div><div className="progress"><span style={{ width: '98.4%' }} /></div></div><div className="pipeline-line"><div><span>Enrichment</span><strong>94.1%</strong></div><div className="progress"><span style={{ width: '94.1%' }} /></div></div><div className="pipeline-line"><div><span>Story linking</span><strong>91.8%</strong></div><div className="progress"><span style={{ width: '91.8%' }} /></div></div><button className="pipeline-link" onClick={() => setActiveNav('Pipeline')}>Open pipeline monitor <ArrowUpRight size={15} /></button></section></div>
          <footer className="disclaimer"><Hash size={14} /> Baatmi surfaces model-generated signals for research. Detected techniques are not proof of bias or falsehood.</footer>
        </div>
      </section>
    </main>
  )
}

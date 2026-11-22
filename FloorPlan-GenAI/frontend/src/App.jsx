import React, { useEffect, useState } from 'react'
import {
  Activity, Box, Database, FlaskConical, GitBranch, History, LayoutDashboard,
  ScanLine, Sparkles, SquareChartGantt,
} from 'lucide-react'
import floorPlanLogo from './assets/floorplan-logo.png'
import { useWorkspace } from './api'
import {
  GenerationForm, MetricStrip, Notice, PlanPreview, ResultDetails, StatusPill,
} from './components/Workspace'
import { Architecture, Dataset, Training } from './pages/Research'
import { Evaluation, HistoryPage } from './pages/Results'

const navigation = [
  ['dashboard', 'Dashboard', LayoutDashboard],
  ['generate', 'Generate Floor Plan', Sparkles],
  ['history', 'Previous Generations', History],
  ['architecture', 'Model Architecture', Box],
  ['dataset', 'Dataset Explorer', Database],
  ['training', 'Training Pipeline', GitBranch],
  ['evaluation', 'Evaluation', FlaskConical],
]

function currentPage() {
  const hash = window.location.hash.slice(1)
  return navigation.some(([key]) => key === hash) ? hash : 'dashboard'
}

function NavLinks({ items, page }) {
  return items.map(([key, label, Icon]) => <a className={'nav-item ' + (key === page ? 'selected' : '')}
    key={key} href={'#' + key} aria-current={key === page ? 'page' : undefined}>
    <Icon size={17} /> <span>{label}</span>
  </a>)
}

function Sidebar({ page, health }) {
  return <aside className="sidebar">
    <a href="#dashboard" className="brand" aria-label="FloorPlan AI dashboard">
      <img className="brand-mark" src={floorPlanLogo} width={40} height={40} alt="" />
      <span><strong>FloorPlan <em>AI</em></strong><small>Generation studio</small></span>
    </a>
    <nav aria-label="Main navigation">
      <div className="nav-group">
        <div className="nav-caption">WORKSPACE</div><NavLinks items={navigation.slice(0, 3)} page={page} />
      </div>
      <div className="nav-group">
        <div className="nav-caption">RESEARCH</div><NavLinks items={navigation.slice(3)} page={page} />
      </div>
    </nav>
    <div className="sidebar-footer">
      <div className="runtime-card"><Activity size={17} /><div><strong>Local runtime</strong>
        <small>{health?.device ? health.device.toUpperCase() + ' inference' : 'Checking engine'}</small></div></div>
    </div>
  </aside>
}

function Workbench({ page, job, jobs, health, ready, active, onSubmit }) {
  const dashboard = page === 'dashboard'
  return <>
    <div className="hero">
      <div className="hero-copy">
        <div className="eyebrow"><ScanLine size={15} /> CONDITIONAL FLOW MATCHING</div>
        <h1>{dashboard ? 'FloorPlan AI' : 'Generate floor plan'}</h1>
        <p>Generate an editable apartment floor plan from room counts and apartment size.</p>
      </div>
      <div className="hero-decor" aria-hidden="true"><SquareChartGantt strokeWidth={0.7} /></div>
    </div>
    {dashboard && <MetricStrip health={health} jobs={jobs} />}
    <div className="workbench">
      <GenerationForm ready={ready} active={active} onSubmit={onSubmit} />
      <PlanPreview key={job?.id || 'new'} job={job} />
    </div>
    <ResultDetails job={job} />
  </>
}

export default function App() {
  const [page, setPage] = useState(currentPage)
  const [selection, setSelection] = useState(null)
  const { health, jobs, connectionError } = useWorkspace()
  const title = navigation.find(([key]) => key === page)[1]
  const ready = health?.state === 'ready' && !connectionError
  const activeJob = jobs.find(job => job.state === 'queued' || job.state === 'running')
  const active = Boolean(activeJob ||
    (selection && !jobs.some(job => job.id === selection.id) && selection.state === 'queued'))
  const selected = selection && (jobs.find(job => job.id === selection.id) || selection)
  const displayed = selected || activeJob || jobs[0]

  useEffect(() => {
    function onNavigate() {
      if (window.location.hash === '#main-content') return
      setPage(currentPage())
      window.scrollTo(0, 0)
    }
    window.addEventListener('hashchange', onNavigate)
    return () => window.removeEventListener('hashchange', onNavigate)
  }, [])

  function viewJob(job) {
    setSelection(job)
    window.location.hash = 'generate'
  }

  const healthLabel = connectionError || health?.state === 'error'
    ? 'Engine unavailable' : ready ? 'Engine ready' : 'Loading engine'

  return <div className="app-shell">
    <a href="#main-content" className="skip-link">Skip to content</a>
    <Sidebar page={page} health={health} />
    <div className="main-shell">
      <header className="topbar">
        <div className="breadcrumb"><span>Workspace</span><span className="crumb-divider">/</span><strong>{title}</strong></div>
        <StatusPill label={healthLabel} tone={ready ? 'green' : healthLabel === 'Engine unavailable' ? 'red' : 'amber'} />
      </header>
      <main id="main-content" className="content" tabIndex={-1}>
        {connectionError && <Notice>{connectionError}</Notice>}
        {health?.state === 'error' && !connectionError && <Notice>{health.message}</Notice>}
        {(page === 'dashboard' || page === 'generate') &&
          <Workbench page={page} job={displayed} jobs={jobs} health={health} ready={ready} active={active} onSubmit={setSelection} />}
        {page === 'history' && <HistoryPage jobs={jobs} viewJob={viewJob} />}
        {page === 'architecture' && <Architecture />}
        {page === 'dataset' && <Dataset />}
        {page === 'training' && <Training />}
        {page === 'evaluation' && <Evaluation jobs={jobs} viewJob={viewJob} />}
      </main>
    </div>
  </div>
}

import React, { useState } from 'react'
import {
  Activity, Box, Check, CircleAlert, Cpu, Download, FileJson, Focus, Layers3,
  LoaderCircle, Maximize, Minus, Plus, ScanLine, Sparkles, ZoomIn, ZoomOut,
} from 'lucide-react'
import { api, fileUrl, formatArea, roomNames } from '../api'

const defaults = { size: 'medium', bedrooms: 3, bathrooms: 2, balconies: 1, seed: '' }
const sizes = ['small', 'medium', 'large']
const countFields = [
  ['bedrooms', 'Bedrooms', 1],
  ['bathrooms', 'Bathrooms', 1],
  ['balconies', 'Balconies', 0],
]

export function StatusPill({ label, tone = 'neutral' }) {
  return <span className={'status-pill ' + tone}><span className="status-dot" />{label}</span>
}

export function JobStatus({ job }) {
  const labels = { queued: 'Queued', running: 'Generating', completed: 'Completed', failed: 'Failed' }
  const tones = { queued: 'amber', running: 'amber', completed: 'green', failed: 'red' }
  return <StatusPill label={labels[job.state]} tone={tones[job.state]} />
}

export function Notice({ children }) {
  return <div className="notice" role="alert"><CircleAlert size={17} /> <div>{children}</div></div>
}

export function MetricStrip({ health, jobs }) {
  const completed = jobs.filter(job => job.state === 'completed').length
  const metrics = [
    [Layers3, 'MODEL', 'Flow Matching', 'Text2Floorplan v1.1.0'],
    [Cpu, 'COMPUTE DEVICE', health?.device?.toUpperCase() || 'Detecting',
      health?.device === 'cpu' && health.threads ? health.threads + ' CPU threads' : 'Local inference'],
    [Box, 'CHECKPOINT', health?.checkpoint === 'loaded' ? 'Loaded' : health?.checkpoint === 'verified' ? 'Verified' : health?.checkpoint === 'unavailable' ? 'Missing or invalid' : 'Pending',
      health?.clip === 'loaded' ? 'CLIP text encoder loaded' : 'CLIP ' + (health?.clip || 'pending')],
    [Activity, 'COMPLETED PLANS', completed, 'Saved on this device'],
  ]
  return <div className="metrics">
    {metrics.map(([Icon, label, value, detail]) => <div className="metric panel" key={label}>
      <div className="metric-icon"><Icon size={19} /></div>
      <div><div className="metric-label">{label}</div><div className="metric-value">{value}</div>
      <div className="metric-detail">{detail}</div></div>
    </div>)}
  </div>
}

function CountInput({ label, value, minimum, maximum, onChange }) {
  return <div className="count-row">
    <span>{label}</span>
    <div className="stepper" role="group" aria-label={label}>
      <button type="button" aria-label={'Decrease ' + label.toLowerCase()} disabled={value <= minimum} onClick={() => onChange(value - 1)}><Minus size={15} /></button>
      <output aria-label={label + ' count'}>{value}</output>
      <button type="button" aria-label={'Increase ' + label.toLowerCase()} disabled={value >= maximum} onClick={() => onChange(value + 1)}><Plus size={15} /></button>
    </div>
  </div>
}

export function GenerationForm({ ready, active, onSubmit }) {
  const [condition, setCondition] = useState(defaults)
  const [sending, setSending] = useState(false)
  const [error, setError] = useState('')
  const sum = condition.bedrooms + condition.bathrooms + condition.balconies
  const setField = (field, value) => setCondition(current => ({ ...current, [field]: value }))

  async function submit(event) {
    event.preventDefault()
    setError('')
    const seed = condition.seed === '' ? null : Number(condition.seed)
    if (seed !== null && (!Number.isInteger(seed) || seed < 0 || seed > 4294967295)) {
      setError('Enter a whole seed from 0 to 4,294,967,295.')
      return
    }
    setSending(true)
    try {
      const job = await api('/jobs', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ...condition, seed }),
      })
      onSubmit(job)
    } catch (failure) {
      setError(failure.message)
    } finally {
      setSending(false)
    }
  }

  return <form className="panel settings" onSubmit={submit}>
    <div className="panel-heading"><div className="section-icon"><ScanLine size={19} /></div>
      <div><h2>Generation settings</h2><p>Define the apartment program.</p></div></div>
    <fieldset>
      <legend>Apartment size</legend>
      <div className="segmented">
        {sizes.map(size => <label key={size}>
          <input type="radio" name="size" value={size} checked={condition.size === size}
            onChange={() => setField('size', size)} />
          <span>{size}</span>
        </label>)}
      </div>
    </fieldset>
    <div className="count-fields">
      {countFields.map(([key, label, minimum]) => <CountInput key={key} label={label} value={condition[key]}
        minimum={minimum} maximum={condition[key] + 17 - sum} onChange={value => setField(key, value)} />)}
    </div>
    <div className="seed-field">
      <label htmlFor="seed">Random seed <span className="muted">(optional)</span></label>
      <input id="seed" type="number" min="0" max="4294967295" step="1" placeholder="Random on each run"
        value={condition.seed} onChange={event => setField('seed', event.target.value)} />
    </div>
    {error && <Notice>{error}</Notice>}
    <button type="submit" className="button primary generate" disabled={!ready || active || sending}>
      {active || sending ? <LoaderCircle className="spin" size={17} /> : <Sparkles size={17} />}
      {active ? 'Generation in progress' : sending ? 'Submitting…' : 'Generate floor plan'}
    </button>
    <p className="form-hint">Living room and kitchen included.</p>
  </form>
}

export function PlanPreview({ job }) {
  const [zoom, setZoom] = useState(1)
  const [imageError, setImageError] = useState(false)
  const ready = job?.state === 'completed'
  const busy = job?.state === 'running' || job?.state === 'queued'
  const canZoom = ready && !imageError

  return <section className="panel preview">
    <header className="panel-heading preview-heading">
      <div className="section-icon"><Focus size={19} /></div>
      <div className="preview-heading-copy"><h2>Plan preview</h2><p>2D floor plan</p></div>
      {job && <JobStatus job={job} />}
    </header>
    <div className="preview-frame">
      {canZoom ? <div className="canvas-scroll">
        <div className="canvas-image" style={{ '--zoom': zoom }}>
          <img src={fileUrl(job, 'png')} alt="Generated apartment floor plan with room labels and estimated areas"
            onError={() => setImageError(true)} />
        </div>
      </div> : <div className="canvas-empty" aria-live="polite">
        {busy ? <LoaderCircle className="spin canvas-icon" size={35} /> : <ScanLine className="canvas-icon" size={38} />}
        <strong>{imageError ? 'Preview unavailable' : busy ? 'Generating floor plan' : job?.state === 'failed' ? 'Generation failed' : 'No floor plan yet'}</strong>
        <p>{imageError ? 'The saved PNG could not be loaded.' : busy ? job.message : job?.state === 'failed' ? job.message : 'Choose the room counts and generate a plan to see it here.'}</p>
        {busy && job.rejections.length > 0 && <span>{job.rejections.length} rejected {job.rejections.length === 1 ? 'candidate' : 'candidates'}</span>}
      </div>}
    </div>
    <footer className="preview-footer">
      <div className="zoom" aria-label="Preview zoom">
        <button type="button" title="Zoom out" aria-label="Zoom out" disabled={!canZoom || zoom <= 0.75}
          onClick={() => setZoom(current => current - 0.25)}><ZoomOut size={16} /></button>
        <button type="button" title="Reset zoom" aria-label="Reset zoom" disabled={!canZoom}
          onClick={() => setZoom(1)}><Maximize size={15} /></button>
        <button type="button" title="Zoom in" aria-label="Zoom in" disabled={!canZoom || zoom >= 2}
          onClick={() => setZoom(current => current + 0.25)}><ZoomIn size={16} /></button>
        <span>{Math.round(100 * zoom)}%</span>
      </div>
      {ready && <div className="downloads">
        <a className="button secondary small" href={fileUrl(job, 'png', true)} aria-label="Download PNG"><Download size={15} /> PNG</a>
        <a className="button secondary small" href={fileUrl(job, 'json', true)} aria-label="Download editable JSON"><FileJson size={15} /> JSON</a>
      </div>}
    </footer>
  </section>
}

export function ResultDetails({ job }) {
  if (job?.state !== 'completed') return null
  const { result, conditions } = job
  const tally = {}
  const roomLabels = result.rooms.map(room => {
    tally[room.type] = (tally[room.type] || 0) + 1
    return { ...room, label: roomNames[room.type] || room.type, index: tally[room.type] }
  })
  return <section className="panel result-details">
    <header className="split-heading"><div><h2>Generation details</h2><p>{job.prompt}</p></div>
      <StatusPill label={result.validation.counts_match && result.validation.areas_valid ? 'Checks passed' : 'Review diagnostics'}
        tone={result.validation.counts_match && result.validation.areas_valid ? 'green' : 'amber'} /></header>
    <dl className="result-metadata">
      <div><dt>Estimated clear area</dt><dd>{formatArea(result.total_area_m2)}</dd></div>
      <div><dt>Final seed</dt><dd>{result.seed}</dd></div>
      <div><dt>Initial seed</dt><dd>{conditions.seed ?? 'Random'}</dd></div>
      <div><dt>Device and time</dt><dd>{job.device?.toUpperCase()} · {job.duration_seconds.toFixed(1)} s</dd></div>
    </dl>
    <div className="room-grid">
      {roomLabels.map(room => <div className="room-tile" key={room.id}>
        <span className={'room-swatch ' + room.type} />
        <span>{room.label}{tally[room.type] > 1 ? ' ' + room.index : ''}</span>
        <strong>{formatArea(room.area_m2)}</strong>
      </div>)}
    </div>
    <p className="form-hint"><Check size={15} /> {result.area_definition}</p>
  </section>
}

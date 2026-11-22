import React, { useState } from 'react'
import { ScanLine } from 'lucide-react'
import { evaluatedRooms, fileUrl, formatArea, formatDate, roomNames } from '../api'
import { JobStatus, StatusPill } from '../components/Workspace'
import { PageHeading } from './Research'

function EmptyPlans({ children }) {
  return <section className="panel empty-panel">
    <ScanLine className="canvas-icon" size={28} />
    <h2>{children}</h2>
    <p>Generate a floor plan to add a record to this workspace.</p>
    <a className="button secondary" href="#generate">Generate floor plan</a>
  </section>
}

export function HistoryPage({ jobs, viewJob }) {
  return <>
    <PageHeading title="Previous generations">Review generation attempts and reopen saved results.</PageHeading>
    {jobs.length === 0 ? <EmptyPlans>No generations yet</EmptyPlans> : <div className="history-grid">
      {jobs.map(job => <article className="panel history-card" key={job.id}>
        {job.state === 'completed' && <img src={fileUrl(job, 'png')} alt={job.prompt} loading="lazy" />}
        <div className="history-card-body">
          <div className="split-heading"><h2>{job.prompt}</h2><JobStatus job={job} /></div>
          <p>{formatDate(job.created_at)}{job.result ? ' · Seed ' + job.result.seed + ' · ' + formatArea(job.result.total_area_m2) : ''}</p>
          {job.state !== 'completed' && <p>{job.message}</p>}
          <button type="button" className="button secondary small" onClick={() => viewJob(job)}>View generation</button>
        </div>
      </article>)}
    </div>}
  </>
}

function CountTable({ validation }) {
  return <table className="eval-table">
    <thead><tr><th scope="col">Room type</th><th scope="col">Requested</th><th scope="col">Generated</th><th scope="col">Match</th></tr></thead>
    <tbody>{evaluatedRooms.map(kind => {
      const matches = validation.requested[kind] === validation.generated[kind]
      return <tr key={kind}><th scope="row">{roomNames[kind]}</th><td>{validation.requested[kind]}</td>
        <td>{validation.generated[kind]}</td><td className={matches ? 'pass' : 'fail'}>{matches ? 'Yes' : 'No'}</td></tr>
    })}</tbody>
  </table>
}

function Rejections({ job }) {
  if (!job.rejections.length) return null
  return <details className="diagnostics">
    <summary>Rejection details</summary><ul>{job.rejections.map((message, index) => <li key={index}>{message}</li>)}</ul>
  </details>
}

function PlanEvaluation({ job, viewJob }) {
  const { result } = job
  return <article className="panel eval-card">
    <img src={fileUrl(job, 'png')} alt={job.prompt} loading="lazy" />
    <div className="eval-card-body">
      <h2>{job.prompt}</h2>
      <p>Seed {result.seed} · {formatArea(result.total_area_m2)} · {job.duration_seconds.toFixed(1)} s</p>
      <CountTable validation={result.validation} />
      <div className="status-row">
        <StatusPill label="Upstream geometry accepted" tone="green" />
        <StatusPill label={result.validation.areas_valid ? 'Room areas passed' : 'Room area flagged'}
          tone={result.validation.areas_valid ? 'green' : 'red'} />
      </div>
      <p>{result.rooms.filter(room => room.area_valid).length} / {result.rooms.length} rooms with valid clear area. Kitchen and bathroom thresholds come from the upstream policy.</p>
      <p>{job.rejections.length} {job.rejections.length === 1 ? 'candidate' : 'candidates'} rejected before acceptance.</p>
      <Rejections job={job} />
      <button type="button" className="button secondary small" onClick={() => viewJob(job)}>Open plan details</button>
    </div>
  </article>
}

export function Evaluation({ jobs, viewJob }) {
  const [selected, setSelected] = useState(null)
  const completed = jobs.filter(job => job.state === 'completed')
  const failed = jobs.filter(job => job.state === 'failed')
  const selectedIds = selected ?? completed.slice(0, 2).map(job => job.id)
  const compared = completed.filter(job => selectedIds.includes(job.id))

  function toggle(id) {
    setSelected(selectedIds.includes(id) ? selectedIds.filter(value => value !== id) : [...selectedIds, id])
  }

  return <>
    <PageHeading title="Evaluation">Compare requested conditions, measured room areas and rejected candidates from actual runs.</PageHeading>
    <div className="section-stack">
      {completed.length ? <section className="panel detail-panel">
        <div className="split-heading"><div><h2>Compare saved plans</h2><p>These diagnostics cover generated geometry and room counts.</p></div></div>
        <fieldset className="eval-choices">
          <legend>Select plans</legend>
          {completed.map(job => <label className="eval-choice" key={job.id}>
            <input type="checkbox" checked={selectedIds.includes(job.id)} onChange={() => toggle(job.id)} />
            Seed {job.result.seed} · {job.id.slice(0, 6)}
          </label>)}
        </fieldset>
        {compared.length ? <div className="eval-grid">
          {compared.map(job => <PlanEvaluation key={job.id} job={job} viewJob={viewJob} />)}
        </div> : <p>Select a saved plan to inspect its diagnostics.</p>}
      </section> : <EmptyPlans>No completed plans to compare</EmptyPlans>}
      {failed.length > 0 && <section className="panel detail-panel">
        <h2>Failed generations</h2>
        {failed.map(job => <div className="data-card failed-record" key={job.id}>
          <h3>{job.prompt}</h3><p>{job.message}</p><Rejections job={job} />
        </div>)}
      </section>}
    </div>
  </>
}

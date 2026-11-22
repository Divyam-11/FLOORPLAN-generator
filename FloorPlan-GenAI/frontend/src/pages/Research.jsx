import React from 'react'
import { Box, Braces, Cpu, FileText, GitBranch, Layers3, ScanLine, Workflow } from 'lucide-react'
import { StatusPill } from '../components/Workspace'

const architecture = [
  [FileText, 'Text prompt', 'Apartment size and room counts produce an English prompt. A rule-based sampler uses the same counts to construct room adjacency and sample corner counts.'],
  [Braces, 'CLIP text encoder', 'CLIP ViT-L/14 tokenizes up to 77 tokens and pools the text sequence into a 768-dimensional conditioning vector. It runs from local pretrained weights.'],
  [Layers3, 'Flow Matching Transformer', 'The released checkpoint has hidden width 512, eight attention heads, two room-node layers and four corner layers. Corner layers use timestep, text and room embeddings with masked attention. The outputs include XY velocity and 16 coordinate bit logits.'],
  [Cpu, 'Flow solver', 'The Heun solver integrates the learned velocity from Gaussian noise at σ = 1 toward geometry at σ = 0. The production runtime uses a 200-step schedule setting and discrete coordinate refinement near the clean endpoint.'],
  [Workflow, 'Geometry reconstruction', 'The pipeline rebuilds polygon rings, restores canvas aspect and physical scale, repairs boundaries and places doors and windows. It checks doorway connections and minimum clear areas.'],
  [ScanLine, 'Editable floor plan', 'The accepted layout is exported as a labelled PNG and editable JSON with rooms, walls, openings, railings, floor areas and the final generation seed.'],
]

const dataStages = [
  ['Clean and reconstruct sources', 'ResPlan vector or graph records go through wall reconstruction, snapping and canvas fitting. RPLAN annotated images are converted into room geometry, filtered and area-calibrated against ResPlan for prompt labels.'],
  ['Extract and normalize', 'The feature builder extracts corners, room identities and adjacency in aligned plan-ID order. Polygon rings are canonicalized and padded to 128 corners of 52 channels with up to 20 room slots.'],
  ['Encode topology', 'Room slots, semantic categories, random-walk and Laplacian graph features form the 61-channel node representation. Adjacency also creates local, connected and global attention masks, with next-corner indices for polygon edges.'],
  ['Generate and encode prompts', 'Room counts and area quantiles produce English prompts. Frozen CLIP encodes deduplicated prompts as 768-dimensional vectors, expanded back into the common plan-ID order. The final validator checks cross-file IDs and structural alignment.'],
]

const priors = [
  ['combined_area_stats_natural.json', 'Quantiles for size labels in training prompts'],
  ['corner_count_distributions.json', 'Corner-count distributions for topology sampling'],
  ['geometry_policy.json', 'Area multipliers and minimum clear areas'],
  ['raw_recovery_scale_stats.json', 'Coordinate-to-metre calibration statistics'],
  ['room_aspect_ranges.json', 'Room aspect ranges for geometry recovery'],
  ['room_aspect_samples.npz', 'Aspect-ratio samples for geometry recovery'],
]

const training = [
  ['Load aligned datasets', 'The trainer reads normalized plan GT, topology GT, masks, next-corner mapping and saved CLIP embeddings. It checks common plan IDs, then creates a seeded train/validation split.'],
  ['Build room and corner conditions', 'Padded corner tokens provide ring and room identities. Attention masks encode within-room, connected-room and global relationships. The model also receives the precomputed CLIP prompt vector.'],
  ['Mix geometry and noise', 'For clean corner XY x₀ and Gaussian noise x₁, the trainer samples σ and forms xσ = (1 − σ)x₀ + σx₁. Training supports D4 rotations and reflections plus a small noise-level-dependent perturbation.'],
  ['Predict a velocity and coordinate bits', 'The node and corner transformer processes the conditioned tokens. Its XY head estimates x₁ − x₀. The discrete head uses the estimated clean geometry and polygon-edge features to predict eight bits per axis.'],
  ['Calculate the objective', 'The total is weighted velocity MSE on valid corners plus low-σ coordinate-bit MSE on valid polygon edges. The trainer reports bit accuracy separately.'],
  ['Optimize and checkpoint', 'The training source uses AdamW, a warmup and cosine learning-rate schedule, and exponential moving-average tracking. It records model, EMA, optimizer and scheduler state and saves the best validation loss checkpoint.'],
]

export function PageHeading({ title, children }) {
  return <div className="page-heading"><h1>{title}</h1><p>{children}</p></div>
}

function Stages({ entries }) {
  return <ol className="pipeline-list">
    {entries.map(([title, text]) => <li key={title}><h3>{title}</h3><p>{text}</p></li>)}
  </ol>
}

export function Architecture() {
  return <>
    <PageHeading title="Model architecture">Follow the production path from prompt conditioning to editable geometry.</PageHeading>
    <div className="section-stack">
      <section className="panel detail-panel">
        <div className="panel-heading"><div className="section-icon"><Box size={19} /></div>
          <div><h2>Inference data flow</h2><p>Expand a stage to inspect its role.</p></div></div>
        <ol className="flow-grid" aria-label="Inference stages">
          {architecture.map(([Icon, title, text], index) => <li key={title}>
            <details className="flow-node" open={index === 2 || undefined}>
              <summary><Icon size={17} />{title}</summary><p>{text}</p>
            </details>
          </li>)}
        </ol>
      </section>
      <section className="panel detail-panel">
        <div className="panel-heading"><div className="section-icon"><GitBranch size={19} /></div>
          <div><h2>Transformer inputs</h2><p>Two conditioning paths meet at the corner layers.</p></div></div>
        <div className="data-grid">
          <article className="data-card"><h3>Text and time</h3><p>A projected CLIP vector is added to the timestep embedding. Adaptive layer normalization injects this combined condition into the corner blocks.</p></article>
          <article className="data-card"><h3>Room graph and corner rings</h3><p>Masked node attention encodes room identities, types and graph positional features. Each corner gathers its room embedding and adds its ring position to the evolving XY geometry.</p></article>
        </div>
      </section>
    </div>
  </>
}

export function Dataset() {
  return <>
    <PageHeading title="Dataset explorer">Inspect the documented data sources, feature pipeline and released examples.</PageHeading>
    <div className="section-stack">
      <section className="panel detail-panel">
        <div className="split-heading"><h2>Original data sources</h2></div>
        <div className="data-grid">
          <article className="data-card"><h3>ResPlan</h3><p>Vector polygons and connectivity graphs for residential floor plans. Its authors provide a separate data grant covering their contributions.</p><a href="https://github.com/m-agour/ResPlan" target="_blank" rel="noreferrer">ResPlan project and terms</a></article>
          <article className="data-card"><h3>RPLAN</h3><p>Annotated raster floor plans introduced by Wu et al. at SIGGRAPH Asia 2019. The raw annotated PNGs must be obtained separately under their access terms.</p><a href="https://ustc-gcl-f.github.io/projects/DeepLayout/index.html" target="_blank" rel="noreferrer">RPLAN original project</a></article>
        </div>
        <p>The upstream repository includes statistical priors and example exports. The raw and prepared training datasets are separate assets.</p>
      </section>
      <section className="panel detail-panel">
        <div className="split-heading"><h2>Preprocessing and model inputs</h2></div>
        <Stages entries={dataStages} />
      </section>
      <section className="panel detail-panel">
        <div className="split-heading"><div><h2>Repository examples</h2><p>Released demonstration outputs. Training records are documented above.</p></div></div>
        <div className="data-grid">
          {[['browser', 'Browser workstation example'], ['revit', 'Native Revit example']].map(([kind, label]) => (
            <article className="data-card" key={kind}>
              <h3>{label}</h3>
              <img className="sample-image" src={'/api/dataset/samples/' + kind + '.png'} alt={label} loading="lazy" />
              <a href={'/api/dataset/samples/' + kind + '.json'} download={kind + '-example.json'}>Download upstream JSON</a>
            </article>
          ))}
        </div>
      </section>
      <section className="panel detail-panel">
        <h2>Available statistical priors</h2>
        <ul className="file-list">
          {priors.map(([filename, purpose]) => <li key={filename}><a href={'/api/dataset/priors/' + filename}>{filename}</a><span>{purpose}</span></li>)}
        </ul>
      </section>
    </div>
  </>
}

export function Training() {
  return <>
    <PageHeading title="Training pipeline">Walk through the preserved upstream implementation.</PageHeading>
    <div className="status-row">
      <StatusPill label="Training code available" tone="green" />
      <StatusPill label="Training execution disabled" tone="amber" />
      <StatusPill label="Inference uses pretrained weights" />
    </div>
    <div className="section-stack">
      <section className="panel detail-panel">
        <Stages entries={training} />
      </section>
      <section className="panel detail-panel">
        <h2>Training records</h2><p>The released inference checkpoint contains learned model tensors and configuration. Training curves and run metrics are unavailable in the source checkout.</p>
      </section>
    </div>
  </>
}

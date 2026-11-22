import { useEffect, useState } from 'react'

export async function api(path, options) {
  const response = await fetch('/api' + path, options)
  if (response.ok) return response.json()
  const error = await response.json().catch(() => ({}))
  const detail = error.detail
  const message = Array.isArray(detail) ? detail.map(entry => entry.msg).join(' ') : detail
  throw new Error(message || 'The local API returned HTTP ' + response.status + '.')
}

export function useWorkspace() {
  const [health, setHealth] = useState(null)
  const [jobs, setJobs] = useState([])
  const [connectionError, setConnectionError] = useState('')

  useEffect(() => {
    const controller = new AbortController()
    let timer
    async function refresh() {
      try {
        const [status, history] = await Promise.all([
          api('/health', { signal: controller.signal }),
          api('/jobs', { signal: controller.signal }),
        ])
        setHealth(status)
        setJobs(history)
        setConnectionError('')
      } catch {
        if (!controller.signal.aborted) {
          setConnectionError('The local API is unreachable. Check that the server is running.')
        }
      } finally {
        if (!controller.signal.aborted) timer = setTimeout(refresh, 1400)
      }
    }
    refresh()
    return () => {
      controller.abort()
      clearTimeout(timer)
    }
  }, [])

  return { health, jobs, connectionError }
}

export function fileUrl(job, type, download = false) {
  return '/api/jobs/' + job.id + '/files/' + type + (download ? '?download=true' : '')
}

export function formatArea(area) {
  return Number.isFinite(area) ? area.toFixed(1) + ' m²' : 'Unavailable'
}

export function formatDate(date) {
  return new Intl.DateTimeFormat(undefined, { dateStyle: 'medium', timeStyle: 'short' }).format(new Date(date))
}

export const roomNames = {
  living: 'Living room', kitchen: 'Kitchen', bedroom: 'Bedroom',
  bathroom: 'Bathroom', balcony: 'Balcony', storage: 'Storage', stair: 'Stair',
}

export const evaluatedRooms = ['bedroom', 'bathroom', 'balcony']

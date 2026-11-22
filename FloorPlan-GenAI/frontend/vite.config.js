import { defineConfig } from 'vite'

const backend = process.env.FLOORPLAN_API_URL || 'http://127.0.0.1:8000'

export default defineConfig({
  server: { host: '127.0.0.1', proxy: { '/api': backend } },
  preview: { host: '127.0.0.1', proxy: { '/api': backend } },
})

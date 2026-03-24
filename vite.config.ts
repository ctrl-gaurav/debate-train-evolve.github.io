import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import path from 'path'
import { cpSync, existsSync } from 'fs'

export default defineConfig({
  plugins: [
    react(),
    tailwindcss(),
    {
      name: 'copy-old-ui',
      closeBundle() {
        const src = path.resolve(__dirname, 'old_ui')
        const dest = path.resolve(__dirname, 'dist/old_ui')
        if (existsSync(src)) {
          cpSync(src, dest, { recursive: true })
        }
      },
    },
  ],
  base: '/debate-train-evolve.github.io/',
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: false,
  },
})

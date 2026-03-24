import { useEffect, useRef } from 'react'
import { useTheme } from '../context/ThemeContext'

interface Particle {
  x: number
  y: number
  vx: number
  vy: number
  radius: number
  opacity: number
  hue: number
}

export default function ParticleBackground() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const { isDark } = useTheme()

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    let animationId: number
    const particles: Particle[] = []
    const particleCount = window.innerWidth < 768 ? 30 : 60

    const resize = () => {
      canvas.width = window.innerWidth
      canvas.height = window.innerHeight
    }
    resize()
    window.addEventListener('resize', resize)

    for (let i = 0; i < particleCount; i++) {
      particles.push({
        x: Math.random() * canvas.width,
        y: Math.random() * canvas.height,
        vx: (Math.random() - 0.5) * 0.35,
        vy: (Math.random() - 0.5) * 0.35,
        radius: Math.random() * 1.5 + 0.5,
        opacity: Math.random() * 0.35 + 0.1,
        hue: Math.random() > 0.5 ? 210 : 195,  // blue or cyan
      })
    }

    const animate = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height)

      particles.forEach((p, i) => {
        p.x += p.vx
        p.y += p.vy

        if (p.x < 0 || p.x > canvas.width) p.vx *= -1
        if (p.y < 0 || p.y > canvas.height) p.vy *= -1

        ctx.beginPath()
        ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2)
        if (isDark) {
          // Blue-cyan particles in dark mode
          ctx.fillStyle = `hsla(${p.hue}, 80%, 70%, ${p.opacity})`
        } else {
          // Subtle blue particles in light mode
          ctx.fillStyle = `hsla(${p.hue}, 60%, 55%, ${p.opacity * 0.3})`
        }
        ctx.fill()

        // Draw connections
        for (let j = i + 1; j < particles.length; j++) {
          const dx = p.x - particles[j].x
          const dy = p.y - particles[j].y
          const distance = Math.sqrt(dx * dx + dy * dy)

          if (distance < 150) {
            ctx.beginPath()
            ctx.moveTo(p.x, p.y)
            ctx.lineTo(particles[j].x, particles[j].y)
            const lineOpacity = (1 - distance / 150) * 0.1
            if (isDark) {
              ctx.strokeStyle = `hsla(220, 70%, 60%, ${lineOpacity})`
            } else {
              ctx.strokeStyle = `hsla(220, 60%, 55%, ${lineOpacity * 0.25})`
            }
            ctx.lineWidth = 0.5
            ctx.stroke()
          }
        }
      })

      animationId = requestAnimationFrame(animate)
    }

    animate()

    return () => {
      cancelAnimationFrame(animationId)
      window.removeEventListener('resize', resize)
    }
  }, [isDark])

  return (
    <canvas
      ref={canvasRef}
      className="fixed inset-0 pointer-events-none z-0"
      style={{ opacity: isDark ? 0.6 : 0.4 }}
    />
  )
}

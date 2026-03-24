import { type ReactNode, useEffect } from 'react'
import { motion } from 'framer-motion'

interface PageWrapperProps {
  children: ReactNode
  className?: string
}

export default function PageWrapper({ children, className = '' }: PageWrapperProps) {
  useEffect(() => {
    window.scrollTo(0, 0)
  }, [])

  return (
    <motion.main
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      transition={{ duration: 0.5, ease: 'easeOut' }}
      className={`relative z-10 pt-16 min-h-screen ${className}`}
    >
      {children}
    </motion.main>
  )
}

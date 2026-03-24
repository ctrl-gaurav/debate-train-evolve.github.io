import { motion } from 'framer-motion'
import { useTheme } from '../context/ThemeContext'

interface SectionHeadingProps {
  badge?: string
  title: string
  subtitle?: string
}

export default function SectionHeading({ badge, title, subtitle }: SectionHeadingProps) {
  const { isDark } = useTheme()

  return (
    <motion.div
      initial={{ opacity: 0, y: 30 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: '-50px' }}
      transition={{ duration: 0.6 }}
      className="text-center mb-12"
    >
      {badge && (
        <span className={`inline-block px-4 py-1.5 rounded-full text-xs font-semibold tracking-wider uppercase mb-4 ${
          isDark
            ? 'bg-electric-500/10 text-electric-300 border border-electric-500/20'
            : 'bg-navy-500/10 text-navy-600 border border-navy-500/20'
        }`}>
          {badge}
        </span>
      )}
      <h2 className={`text-3xl sm:text-4xl lg:text-5xl font-display font-bold mb-4 ${
        isDark ? 'text-white' : 'text-navy-900'
      }`}>
        {title}
      </h2>
      {/* Gradient underline */}
      <div className="flex justify-center mb-4">
        <div className={`h-px w-24 ${
          isDark
            ? 'bg-gradient-to-r from-transparent via-electric-500/50 to-transparent'
            : 'bg-gradient-to-r from-transparent via-navy-400/40 to-transparent'
        }`} />
      </div>
      {subtitle && (
        <p className={`text-lg max-w-2xl mx-auto ${
          isDark ? 'text-gray-400' : 'text-gray-600'
        }`}>
          {subtitle}
        </p>
      )}
    </motion.div>
  )
}

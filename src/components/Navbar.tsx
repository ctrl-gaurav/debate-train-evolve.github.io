import { useState, useEffect } from 'react'
import { Link, useLocation } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import { useTheme } from '../context/ThemeContext'
import { HiOutlineSun, HiOutlineMoon, HiOutlineMenu, HiOutlineX } from 'react-icons/hi'
import { FaFileAlt } from 'react-icons/fa'

const navLinks = [
  { path: '/', label: 'Home' },
  { path: '/method', label: 'Method' },
  { path: '/results', label: 'Results' },
  { path: '/docs', label: 'Docs' },
  { path: '/team', label: 'Team' },
  { path: '/citation', label: 'Citation' },
]

export default function Navbar() {
  const { isDark, toggleTheme } = useTheme()
  const location = useLocation()
  const [isScrolled, setIsScrolled] = useState(false)
  const [isMobileOpen, setIsMobileOpen] = useState(false)

  useEffect(() => {
    const handleScroll = () => setIsScrolled(window.scrollY > 20)
    window.addEventListener('scroll', handleScroll)
    return () => window.removeEventListener('scroll', handleScroll)
  }, [])

  useEffect(() => {
    setIsMobileOpen(false)
  }, [location.pathname])

  return (
    <motion.nav
      initial={{ y: -100 }}
      animate={{ y: 0 }}
      transition={{ duration: 0.6, ease: 'easeOut' }}
      className={`fixed top-0 left-0 right-0 z-50 transition-all duration-500 ${
        isScrolled
          ? isDark
            ? 'glass shadow-lg shadow-navy-950/60'
            : 'glass-light shadow-lg shadow-navy-200/30'
          : 'bg-transparent'
      }`}
    >
      {/* Gradient top line accent */}
      {isScrolled && (
        <div className="absolute top-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-electric-500/30 to-transparent" />
      )}

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          <Link to="/" className="flex items-center space-x-2.5 group">
            <div className={`w-9 h-9 rounded-xl flex items-center justify-center font-bold text-sm transition-all duration-300 ${
              isDark
                ? 'bg-gradient-to-br from-electric-400 to-navy-500 text-white group-hover:shadow-lg group-hover:shadow-electric-400/25'
                : 'bg-gradient-to-br from-navy-600 to-electric-600 text-white group-hover:shadow-lg group-hover:shadow-navy-500/25'
            }`}>
              D
            </div>
            <span className={`font-display font-bold text-lg tracking-tight ${
              isDark ? 'text-white' : 'text-navy-900'
            }`}>
              DTE
            </span>
          </Link>

          <div className="hidden md:flex items-center space-x-0.5">
            {navLinks.map((link) => {
              const isActive = location.pathname === link.path
              return (
                <Link
                  key={link.path}
                  to={link.path}
                  className={`relative px-4 py-2 rounded-lg text-sm font-medium transition-all duration-300 ${
                    isActive
                      ? isDark
                        ? 'text-electric-300'
                        : 'text-navy-700'
                      : isDark
                        ? 'text-gray-400 hover:text-white'
                        : 'text-gray-500 hover:text-navy-900'
                  }`}
                >
                  {link.label}
                  {isActive && (
                    <motion.div
                      layoutId="activeNav"
                      className={`absolute inset-0 rounded-lg -z-10 ${
                        isDark
                          ? 'bg-white/[0.06] border border-electric-500/15 shadow-sm shadow-electric-500/10'
                          : 'bg-navy-500/[0.06] border border-navy-500/12'
                      }`}
                      transition={{ type: 'spring', stiffness: 350, damping: 30 }}
                    />
                  )}
                  {isActive && (
                    <motion.div
                      layoutId="activeNavBar"
                      className={`absolute -bottom-px left-2 right-2 h-px ${
                        isDark
                          ? 'bg-gradient-to-r from-transparent via-electric-400 to-transparent'
                          : 'bg-gradient-to-r from-transparent via-navy-500 to-transparent'
                      }`}
                      transition={{ type: 'spring', stiffness: 350, damping: 30 }}
                    />
                  )}
                </Link>
              )
            })}
          </div>

          <div className="flex items-center space-x-2">
            <a
              href="https://github.com/ctrl-gaurav/Debate-Train-Evolve"
              target="_blank"
              rel="noopener noreferrer"
              className={`hidden sm:flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-all duration-300 ${
                isDark
                  ? 'text-gray-400 hover:text-white bg-white/[0.04] hover:bg-white/[0.08] border border-white/[0.08] hover:border-electric-500/20'
                  : 'text-gray-600 hover:text-navy-900 bg-navy-50 hover:bg-navy-100 border border-navy-200'
              }`}
            >
              <svg className="w-3.5 h-3.5" fill="currentColor" viewBox="0 0 24 24"><path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0024 12c0-6.63-5.37-12-12-12z"/></svg>
              GitHub
            </a>

            <a
              href="https://aclanthology.org/2025.emnlp-main.1666/"
              target="_blank"
              rel="noopener noreferrer"
              className={`hidden sm:flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-all duration-300 ${
                isDark
                  ? 'text-gray-400 hover:text-white bg-white/[0.04] hover:bg-white/[0.08] border border-white/[0.08] hover:border-electric-500/20'
                  : 'text-gray-600 hover:text-navy-900 bg-navy-50 hover:bg-navy-100 border border-navy-200'
              }`}
            >
              <FaFileAlt className="w-3.5 h-3.5" />
              Paper
            </a>

            <button
              onClick={toggleTheme}
              className={`p-2 rounded-lg transition-all duration-300 ${
                isDark
                  ? 'text-gray-400 hover:text-yellow-300 hover:bg-white/[0.05]'
                  : 'text-gray-600 hover:text-navy-600 hover:bg-navy-500/10'
              }`}
              aria-label="Toggle theme"
            >
              {isDark ? <HiOutlineSun size={18} /> : <HiOutlineMoon size={18} />}
            </button>

            <button
              onClick={() => setIsMobileOpen(!isMobileOpen)}
              className={`md:hidden p-2 rounded-lg transition-all duration-300 ${
                isDark
                  ? 'text-gray-400 hover:text-white hover:bg-white/[0.05]'
                  : 'text-gray-600 hover:text-navy-900 hover:bg-navy-500/10'
              }`}
              aria-label="Toggle menu"
            >
              {isMobileOpen ? <HiOutlineX size={20} /> : <HiOutlineMenu size={20} />}
            </button>
          </div>
        </div>
      </div>

      <AnimatePresence>
        {isMobileOpen && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.3 }}
            className={`md:hidden overflow-hidden border-t ${
              isDark ? 'glass border-white/[0.06]' : 'glass-light border-navy-200/20'
            }`}
          >
            <div className="px-4 py-3 space-y-1">
              {navLinks.map((link) => {
                const isActive = location.pathname === link.path
                return (
                  <Link
                    key={link.path}
                    to={link.path}
                    className={`block px-4 py-2.5 rounded-lg text-sm font-medium transition-all ${
                      isActive
                        ? isDark
                          ? 'text-electric-300 bg-electric-500/10 border-l-2 border-electric-400'
                          : 'text-navy-600 bg-navy-500/10 border-l-2 border-navy-500'
                        : isDark
                          ? 'text-gray-400 hover:text-white hover:bg-white/[0.04]'
                          : 'text-gray-600 hover:text-navy-900 hover:bg-navy-100'
                    }`}
                  >
                    {link.label}
                  </Link>
                )
              })}
              <a
                href="https://aclanthology.org/2025.emnlp-main.1666/"
                target="_blank"
                rel="noopener noreferrer"
                className={`flex items-center gap-2 px-4 py-2.5 rounded-lg text-sm font-medium transition-all ${
                  isDark
                    ? 'text-gray-400 hover:text-white hover:bg-white/[0.04]'
                    : 'text-gray-600 hover:text-navy-900 hover:bg-navy-100'
                }`}
              >
                <FaFileAlt className="w-3.5 h-3.5" />
                Paper
              </a>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.nav>
  )
}

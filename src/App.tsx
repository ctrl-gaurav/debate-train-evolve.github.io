import { Routes, Route, useLocation } from 'react-router-dom'
import { AnimatePresence } from 'framer-motion'
import { useTheme } from './context/ThemeContext'
import Navbar from './components/Navbar'
import Footer from './components/Footer'
import ParticleBackground from './components/ParticleBackground'
import Home from './pages/Home'
import Results from './pages/Results'
import Method from './pages/Method'
import Docs from './pages/Docs'
import Team from './pages/Team'
import CitationPage from './pages/Citation'

export default function App() {
  const location = useLocation()
  const { isDark } = useTheme()

  return (
    <div className={`min-h-screen transition-colors duration-500 ${
      isDark
        ? 'bg-[#060b18] text-white'
        : 'bg-[#f8f9fd] text-navy-900'
    }`}>
      {/* Background layers */}
      <ParticleBackground />

      {/* Subtle grid pattern overlay */}
      <div className={`fixed inset-0 pointer-events-none z-0 ${
        isDark ? 'grid-pattern' : 'grid-pattern-light'
      }`} style={{ opacity: isDark ? 0.5 : 0.3 }} />

      {/* Top ambient gradient */}
      <div className={`fixed top-0 left-0 right-0 h-[600px] pointer-events-none z-0 ${
        isDark
          ? 'bg-gradient-to-b from-navy-900/40 via-transparent to-transparent'
          : 'bg-gradient-to-b from-navy-50/50 via-transparent to-transparent'
      }`} />

      <Navbar />
      <AnimatePresence mode="wait">
        <Routes location={location} key={location.pathname}>
          <Route path="/" element={<Home />} />
          <Route path="/method" element={<Method />} />
          <Route path="/results" element={<Results />} />
          <Route path="/docs" element={<Docs />} />
          <Route path="/team" element={<Team />} />
          <Route path="/citation" element={<CitationPage />} />
        </Routes>
      </AnimatePresence>
      <Footer />
    </div>
  )
}

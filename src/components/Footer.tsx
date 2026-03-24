import { Link } from 'react-router-dom'
import { useTheme } from '../context/ThemeContext'
import { FaGithub, FaFileAlt, FaGlobe } from 'react-icons/fa'

export default function Footer() {
  const { isDark } = useTheme()

  return (
    <footer className={`relative z-10 ${
      isDark
        ? 'bg-navy-950/80'
        : 'bg-white/80'
    }`}>
      {/* Gradient top border */}
      <div className="h-px bg-gradient-to-r from-transparent via-electric-500/30 to-transparent" />

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
          {/* Brand */}
          <div className="md:col-span-1">
            <div className="flex items-center space-x-2 mb-4">
              <div className={`w-8 h-8 rounded-lg bg-gradient-to-br from-electric-400 to-navy-500 flex items-center justify-center font-bold text-sm text-white shadow-md ${
                isDark ? 'shadow-electric-500/15' : 'shadow-navy-500/15'
              }`}>
                D
              </div>
              <span className={`font-display font-bold text-lg ${isDark ? 'text-white' : 'text-navy-900'}`}>
                DTE Framework
              </span>
            </div>
            <p className={`text-sm leading-relaxed ${isDark ? 'text-gray-500' : 'text-gray-600'}`}>
              Debate, Train, Evolve: Self-Evolution of Language Model Reasoning.
              Published at EMNLP 2025.
            </p>
          </div>

          {/* Quick Links */}
          <div>
            <h4 className={`font-semibold text-sm mb-4 ${isDark ? 'text-gray-300' : 'text-navy-800'}`}>
              Navigation
            </h4>
            <ul className="space-y-2.5">
              {[
                { to: '/', label: 'Home' },
                { to: '/method', label: 'Method' },
                { to: '/results', label: 'Results' },
                { to: '/docs', label: 'Documentation' },
              ].map((link) => (
                <li key={link.to}>
                  <Link
                    to={link.to}
                    className={`text-sm transition-colors duration-200 ${
                      isDark ? 'text-gray-500 hover:text-electric-300' : 'text-gray-600 hover:text-navy-600'
                    }`}
                  >
                    {link.label}
                  </Link>
                </li>
              ))}
            </ul>
          </div>

          {/* Resources */}
          <div>
            <h4 className={`font-semibold text-sm mb-4 ${isDark ? 'text-gray-300' : 'text-navy-800'}`}>
              Resources
            </h4>
            <ul className="space-y-2.5">
              <li>
                <a
                  href="https://github.com/ctrl-gaurav/Debate-Train-Evolve"
                  target="_blank"
                  rel="noopener noreferrer"
                  className={`text-sm flex items-center gap-2 transition-colors duration-200 ${
                    isDark ? 'text-gray-500 hover:text-electric-300' : 'text-gray-600 hover:text-navy-600'
                  }`}
                >
                  <FaGithub size={14} /> GitHub
                </a>
              </li>
              <li>
                <a
                  href="https://aclanthology.org/2025.emnlp-main.1666/"
                  target="_blank"
                  rel="noopener noreferrer"
                  className={`text-sm flex items-center gap-2 transition-colors duration-200 ${
                    isDark ? 'text-gray-500 hover:text-electric-300' : 'text-gray-600 hover:text-navy-600'
                  }`}
                >
                  <FaFileAlt size={14} /> Paper (ACL Anthology)
                </a>
              </li>
              <li>
                <a
                  href="https://ctrl-gaurav.github.io/debate-train-evolve.github.io/"
                  target="_blank"
                  rel="noopener noreferrer"
                  className={`text-sm flex items-center gap-2 transition-colors duration-200 ${
                    isDark ? 'text-gray-500 hover:text-electric-300' : 'text-gray-600 hover:text-navy-600'
                  }`}
                >
                  <FaGlobe size={14} /> Website
                </a>
              </li>
            </ul>
          </div>

          {/* Affiliations */}
          <div>
            <h4 className={`font-semibold text-sm mb-4 ${isDark ? 'text-gray-300' : 'text-navy-800'}`}>
              Affiliations
            </h4>
            <p className={`text-sm ${isDark ? 'text-gray-500' : 'text-gray-600'}`}>
              Virginia Tech<br />
              Department of Computer Science
            </p>
            <div className={`mt-4 pt-4 border-t ${isDark ? 'border-white/[0.05]' : 'border-navy-200/20'}`}>
              <p className={`text-xs ${isDark ? 'text-gray-600' : 'text-gray-500'}`}>
                Supported by NSF and Virginia Tech
              </p>
            </div>
          </div>
        </div>

        {/* Bottom bar */}
        <div className={`mt-8 pt-6 flex flex-col sm:flex-row justify-between items-center gap-4`}>
          <div className="h-px flex-1 bg-gradient-to-r from-transparent via-electric-500/10 to-transparent hidden sm:block" />
          <p className={`text-xs ${isDark ? 'text-gray-600' : 'text-gray-500'} shrink-0 px-4`}>
            &copy; 2025 DTE Research Team, Virginia Tech. All rights reserved.
          </p>
          <div className="h-px flex-1 bg-gradient-to-r from-transparent via-electric-500/10 to-transparent hidden sm:block" />
          <div className="flex items-center space-x-4 shrink-0">
            <a
              href="https://github.com/ctrl-gaurav/Debate-Train-Evolve"
              target="_blank"
              rel="noopener noreferrer"
              className={`transition-colors duration-200 ${
                isDark ? 'text-gray-600 hover:text-electric-300' : 'text-gray-500 hover:text-navy-600'
              }`}
            >
              <FaGithub size={18} />
            </a>
          </div>
        </div>
      </div>
    </footer>
  )
}

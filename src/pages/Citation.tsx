import { useState } from 'react'
import { motion } from 'framer-motion'
import { useTheme } from '../context/ThemeContext'
import PageWrapper from '../components/PageWrapper'
import SectionHeading from '../components/SectionHeading'
import CodeBlock from '../components/CodeBlock'
import { FaGithub } from 'react-icons/fa'

const bibtex = `@inproceedings{srivastava2025debate,
  title={Debate, Train, Evolve: Self-Evolution of Language Model Reasoning},
  author={Srivastava, Gaurav and Bi, Zhenyu and Lu, Meng and Wang, Xuan},
  booktitle={Proceedings of the 2025 Conference on Empirical Methods
             in Natural Language Processing (EMNLP)},
  year={2025},
  url={https://aclanthology.org/2025.emnlp-main.1666/}
}`

export default function CitationPage() {
  const { isDark } = useTheme()
  const [copied, setCopied] = useState(false)

  const handleCopy = () => {
    navigator.clipboard.writeText(bibtex)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <PageWrapper>
      <section className="pt-24 pb-20">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
          <SectionHeading
            badge="Citation"
            title="Cite Our Work"
            subtitle="If you find DTE useful in your research, please cite our paper."
          />

          {/* Paper info card */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            className={`relative p-6 sm:p-8 rounded-2xl mb-8 overflow-hidden ${isDark ? 'glass-card' : 'glass-card-light'}`}
          >
            {/* Decorative accent */}
            <div className={`absolute top-0 left-0 right-0 h-px ${
              isDark
                ? 'bg-gradient-to-r from-transparent via-electric-500/30 to-transparent'
                : 'bg-gradient-to-r from-transparent via-navy-400/20 to-transparent'
            }`} />

            <div className={`text-xs font-mono uppercase tracking-wider mb-3 ${isDark ? 'text-electric-400' : 'text-navy-500'}`}>
              EMNLP 2025 -- Main Conference
            </div>
            <h2 className={`text-xl sm:text-2xl font-display font-bold mb-3 ${isDark ? 'text-white' : 'text-navy-900'}`}>
              Debate, Train, Evolve: Self-Evolution of Language Model Reasoning
            </h2>
            <p className={`text-sm mb-4 ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>
              Gaurav Srivastava, Zhenyu Bi, Meng Lu, Xuan Wang
            </p>
            <p className={`text-sm mb-6 leading-relaxed ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>
              We propose DTE, a novel ground truth-free training framework that uses multi-agent debate traces
              to evolve a single language model. With a new Reflect-Critique-Refine prompting strategy and GRPO training,
              DTE achieves an average +8.92% accuracy gain on GSM-Plus and strong cross-domain generalization
              across 7 reasoning benchmarks.
            </p>

            <div className="flex flex-wrap gap-3">
              <a
                href="https://aclanthology.org/2025.emnlp-main.1666/"
                target="_blank"
                rel="noopener noreferrer"
                className={`inline-flex items-center gap-2 px-5 py-2.5 rounded-xl text-sm font-semibold transition-all duration-300 ${
                  isDark
                    ? 'bg-gradient-to-r from-electric-500 to-electric-600 text-white hover:shadow-lg hover:shadow-electric-500/25 hover:scale-[1.02]'
                    : 'bg-gradient-to-r from-navy-600 to-electric-600 text-white hover:shadow-lg hover:shadow-navy-500/25 hover:scale-[1.02]'
                }`}
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
                  <path d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
                </svg>
                ACL Anthology
              </a>
              <a
                href="https://github.com/ctrl-gaurav/Debate-Train-Evolve"
                target="_blank"
                rel="noopener noreferrer"
                className={`inline-flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-medium transition-all duration-300 ${
                  isDark
                    ? 'bg-white/[0.04] text-white border border-white/[0.08] hover:bg-white/[0.08] hover:border-electric-500/20'
                    : 'bg-navy-50 text-navy-900 border border-navy-200 hover:bg-navy-100'
                }`}
              >
                <FaGithub size={16} />
                Code Repository
              </a>
              <a
                href="https://ctrl-gaurav.github.io/debate-train-evolve.github.io/"
                target="_blank"
                rel="noopener noreferrer"
                className={`inline-flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-medium transition-all duration-300 ${
                  isDark
                    ? 'bg-white/[0.04] text-white border border-white/[0.08] hover:bg-white/[0.08] hover:border-electric-500/20'
                    : 'bg-navy-50 text-navy-900 border border-navy-200 hover:bg-navy-100'
                }`}
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
                  <path d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
                </svg>
                Project Website
              </a>
            </div>
          </motion.div>

          {/* BibTeX */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.1 }}
          >
            <div className="flex items-center justify-between mb-3">
              <h3 className={`text-lg font-display font-semibold ${isDark ? 'text-white' : 'text-navy-900'}`}>
                BibTeX
              </h3>
              <button
                onClick={handleCopy}
                className={`px-4 py-1.5 rounded-lg text-xs font-mono transition-all duration-200 ${
                  copied
                    ? isDark ? 'bg-green-500/15 text-green-400' : 'bg-green-50 text-green-600'
                    : isDark ? 'bg-white/[0.04] text-gray-400 hover:text-white hover:bg-white/[0.08]' : 'bg-navy-50 text-gray-500 hover:text-navy-900 hover:bg-navy-100'
                }`}
              >
                {copied ? 'Copied!' : 'Copy BibTeX'}
              </button>
            </div>
            <CodeBlock code={bibtex} language="bibtex" title="citation.bib" />
          </motion.div>

          {/* Key results for citing */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.5, delay: 0.2 }}
            className={`mt-8 p-6 rounded-2xl ${isDark ? 'glass-card' : 'glass-card-light'}`}
          >
            <h3 className={`text-lg font-display font-semibold mb-4 ${isDark ? 'text-white' : 'text-navy-900'}`}>
              Key Results for Reference
            </h3>
            <ul className={`space-y-3 text-sm ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>
              {[
                '+8.92% average accuracy improvement on GSM-Plus',
                '+5.8% average cross-domain generalization',
                '7 benchmarks, 6 open-weight models (1.5B to 14B parameters)',
                'RCR prompting reduces sycophancy by 50%',
                'GRPO consistently outperforms SFT and DPO training methods',
              ].map((item, i) => (
                <li key={i} className="flex items-start gap-3">
                  <span className={`mt-1.5 w-1.5 h-1.5 rounded-full shrink-0 ${
                    isDark ? 'bg-electric-400' : 'bg-navy-500'
                  }`} />
                  {item}
                </li>
              ))}
            </ul>
          </motion.div>
        </div>
      </section>
    </PageWrapper>
  )
}

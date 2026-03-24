import { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import { motion } from 'framer-motion'
import { useTheme } from '../context/ThemeContext'
import { useScrollAnimation, useAnimatedCounter } from '../hooks/useScrollAnimation'
import PageWrapper from '../components/PageWrapper'
import SectionHeading from '../components/SectionHeading'
import { FaGithub, FaArrowRight, FaGraduationCap, FaExternalLinkAlt, FaFileAlt } from 'react-icons/fa'

/* ---------- Typing Effect ---------- */
function TypingEffect() {
  const phrases = [
    'Self-Evolution of Language Model Reasoning',
    'Multi-Agent Debate + GRPO Training',
    'Zero Ground Truth Labels Required',
    '+13.92% Best Accuracy Gain (GSM-Plus)',
    'Cross-Domain Generalization to Science',
    '6 Models, 7 Benchmarks, 1 Framework',
  ]
  const [phraseIdx, setPhraseIdx] = useState(0)
  const [charIdx, setCharIdx] = useState(0)
  const [isDeleting, setIsDeleting] = useState(false)

  useEffect(() => {
    const cur = phrases[phraseIdx]
    let timeout: ReturnType<typeof setTimeout>
    if (!isDeleting && charIdx < cur.length) {
      timeout = setTimeout(() => setCharIdx(charIdx + 1), 45)
    } else if (!isDeleting && charIdx === cur.length) {
      timeout = setTimeout(() => setIsDeleting(true), 2200)
    } else if (isDeleting && charIdx > 0) {
      timeout = setTimeout(() => setCharIdx(charIdx - 1), 25)
    } else if (isDeleting && charIdx === 0) {
      setIsDeleting(false)
      setPhraseIdx((phraseIdx + 1) % phrases.length)
    }
    return () => clearTimeout(timeout)
  }, [charIdx, isDeleting, phraseIdx, phrases])

  return (
    <span className="typing-cursor font-mono text-electric-400">
      {phrases[phraseIdx].substring(0, charIdx)}
    </span>
  )
}

/* ---------- Authors ---------- */
const authors = [
  { name: 'Gaurav Srivastava', email: 'gks@vt.edu', note: '*' },
  { name: 'Zhenyu Bi', email: 'zhenyub@vt.edu' },
  { name: 'Meng Lu', email: 'menglu@vt.edu' },
  { name: 'Xuan Wang', email: 'xuanw@vt.edu', note: '\u2020' },
]

/* ---------- Stat Card ---------- */
function StatCard({ value, suffix, label, delay }: {
  value: number; suffix: string; label: string; delay: number
}) {
  const { isDark } = useTheme()
  const { ref, isVisible } = useScrollAnimation(0.3)
  const count = useAnimatedCounter(value, 2000, isVisible)

  return (
    <motion.div
      ref={ref}
      initial={{ opacity: 0, y: 30 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      transition={{ duration: 0.6, delay }}
      whileHover={{ y: -4 }}
      className={`relative group p-6 rounded-2xl transition-all duration-300 overflow-hidden ${
        isDark
          ? 'glass-card hover:shadow-lg hover:shadow-electric-500/10'
          : 'glass-card-light hover:shadow-lg hover:shadow-navy-500/10'
      }`}
    >
      {/* Hover gradient */}
      <div className={`absolute inset-0 rounded-2xl opacity-0 group-hover:opacity-100 transition-opacity duration-500 ${
        isDark ? 'bg-gradient-to-br from-electric-500/[0.06] to-cyan-500/[0.04]' : 'bg-gradient-to-br from-navy-500/[0.04] to-electric-500/[0.04]'
      }`} />
      <div className="relative">
        <div className={`text-3xl sm:text-4xl font-bold font-display mb-1 ${
          isDark ? 'text-white' : 'text-navy-900'
        }`}>
          {value > 0 ? '+' : ''}{count.toFixed(value % 1 !== 0 ? 2 : 0)}{suffix}
        </div>
        <div className={`text-sm ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>{label}</div>
      </div>
      {/* Bottom accent line */}
      <div className={`absolute bottom-0 left-4 right-4 h-px opacity-0 group-hover:opacity-100 transition-opacity duration-500 ${
        isDark
          ? 'bg-gradient-to-r from-transparent via-electric-500/30 to-transparent'
          : 'bg-gradient-to-r from-transparent via-navy-400/30 to-transparent'
      }`} />
    </motion.div>
  )
}

/* ---------- Pipeline Step ---------- */
function PipelineStep({ title, desc, icon, index, isDark }: {
  title: string; desc: string; icon: React.ReactNode; index: number; isDark: boolean
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 30 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      transition={{ duration: 0.5, delay: index * 0.15 }}
      className="flex flex-col items-center text-center group"
    >
      <motion.div
        whileHover={{ scale: 1.08 }}
        className={`w-20 h-20 rounded-2xl flex items-center justify-center mb-4 transition-all duration-300 ${
          isDark
            ? 'bg-gradient-to-br from-electric-500/15 to-cyan-500/10 border border-electric-500/20 text-electric-400 group-hover:border-electric-500/35 group-hover:shadow-lg group-hover:shadow-electric-500/10'
            : 'bg-gradient-to-br from-navy-500/10 to-electric-500/8 border border-navy-300/30 text-navy-600 group-hover:border-navy-400/40 group-hover:shadow-lg group-hover:shadow-navy-500/10'
        }`}
      >
        {icon}
      </motion.div>
      <div className={`text-sm font-mono uppercase tracking-wider mb-1 ${
        isDark ? 'text-electric-300' : 'text-navy-600'
      }`}>Step {index + 1}</div>
      <h3 className={`text-xl font-display font-bold mb-2 ${isDark ? 'text-white' : 'text-navy-900'}`}>
        {title}
      </h3>
      <p className={`text-sm max-w-xs ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>{desc}</p>
    </motion.div>
  )
}

/* ---------- Feature Card ---------- */
function FeatureCard({ title, description, delay }: {
  title: string; description: string; delay: number
}) {
  const { isDark } = useTheme()
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      transition={{ duration: 0.5, delay }}
      whileHover={{ y: -3 }}
      className={`relative group p-6 rounded-2xl transition-all duration-300 overflow-hidden ${
        isDark
          ? 'glass-card hover:border-electric-500/20'
          : 'glass-card-light hover:border-navy-300/40'
      }`}
    >
      {/* Top accent line */}
      <div className={`absolute top-0 left-6 right-6 h-px transition-opacity duration-500 opacity-0 group-hover:opacity-100 ${
        isDark
          ? 'bg-gradient-to-r from-transparent via-electric-500/40 to-transparent'
          : 'bg-gradient-to-r from-transparent via-navy-400/30 to-transparent'
      }`} />
      <h3 className={`text-lg font-display font-semibold mb-2 ${isDark ? 'text-white' : 'text-navy-900'}`}>
        {title}
      </h3>
      <p className={`text-sm leading-relaxed ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>
        {description}
      </p>
    </motion.div>
  )
}

/* ========== HOME PAGE ========== */
export default function Home() {
  const { isDark } = useTheme()

  return (
    <PageWrapper>
      {/* Hero Section */}
      <section className="relative min-h-[90vh] flex items-center justify-center overflow-hidden">
        {/* Decorative orbs */}
        <div className={`orb orb-animated w-[500px] h-[500px] top-10 -left-60 ${isDark ? 'bg-electric-500/[0.07]' : 'bg-electric-500/[0.04]'}`} />
        <div className={`orb orb-animated w-[400px] h-[400px] bottom-10 -right-48 ${isDark ? 'bg-cyan-500/[0.06]' : 'bg-cyan-500/[0.04]'}`} style={{ animationDelay: '2s' }} />
        <div className={`orb w-72 h-72 top-1/3 right-1/4 ${isDark ? 'bg-navy-500/10' : 'bg-navy-500/[0.04]'}`} />

        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 text-center relative z-10">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, ease: 'easeOut' }}
          >
            {/* Badge */}
            <motion.div
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.2 }}
              className={`inline-flex items-center gap-2.5 px-5 py-2 rounded-full text-xs font-semibold tracking-wider uppercase mb-8 ${
                isDark
                  ? 'bg-electric-500/10 text-electric-300 border border-electric-500/20'
                  : 'bg-navy-500/[0.08] text-navy-600 border border-navy-400/20'
              }`}
            >
              <span className="w-1.5 h-1.5 rounded-full bg-green-400 animate-pulse" />
              EMNLP 2025 Main Conference
            </motion.div>

            {/* Title */}
            <h1 className="text-4xl sm:text-5xl md:text-6xl lg:text-7xl font-display font-bold tracking-tight mb-6">
              <span className={isDark ? 'hero-gradient-text' : 'hero-gradient-text-light'}>Debate</span>
              <span className={isDark ? 'text-gray-500' : 'text-gray-400'}>, </span>
              <span className={isDark ? 'hero-gradient-text' : 'hero-gradient-text-light'}>Train</span>
              <span className={isDark ? 'text-gray-500' : 'text-gray-400'}>, </span>
              <span className={isDark ? 'hero-gradient-text' : 'hero-gradient-text-light'}>Evolve</span>
            </h1>

            {/* Subtitle */}
            <p className={`text-lg sm:text-xl max-w-2xl mx-auto mb-6 ${
              isDark ? 'text-gray-400' : 'text-gray-600'
            }`}>
              A ground-truth-free training framework for self-evolving LLM reasoning through multi-agent debate traces.
            </p>

            {/* Authors inline */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.4, duration: 0.6 }}
              className="mb-6"
            >
              <div className="flex flex-wrap items-center justify-center gap-x-2 gap-y-1 mb-2">
                {authors.map((a, i) => (
                  <span key={a.name} className="inline-flex items-center">
                    <a
                      href={`mailto:${a.email}`}
                      className={`text-sm sm:text-base font-display font-semibold transition-colors duration-200 ${
                        isDark ? 'text-gray-200 hover:text-electric-300' : 'text-navy-800 hover:text-navy-600'
                      }`}
                    >
                      {a.name}
                    </a>
                    {a.note && (
                      <sup className={`text-[10px] ml-0.5 font-bold ${isDark ? 'text-electric-400' : 'text-navy-500'}`}>{a.note}</sup>
                    )}
                    {i < authors.length - 1 && (
                      <span className={`mx-1 ${isDark ? 'text-gray-600' : 'text-gray-400'}`}>,</span>
                    )}
                  </span>
                ))}
              </div>
              <div className={`flex items-center justify-center gap-2 text-xs sm:text-sm ${isDark ? 'text-gray-500' : 'text-gray-500'}`}>
                <FaGraduationCap size={14} className={isDark ? 'text-electric-400/60' : 'text-navy-500/60'} />
                <span>Department of Computer Science, Virginia Tech</span>
              </div>
              <div className={`flex items-center justify-center gap-4 mt-1.5 text-[11px] font-mono ${isDark ? 'text-gray-600' : 'text-gray-400'}`}>
                <span>* Lead Author</span>
                <span>&dagger; Corresponding Author</span>
              </div>
            </motion.div>

            {/* Typing effect */}
            <div className="h-8 mb-8 flex items-center justify-center">
              <TypingEffect />
            </div>

            {/* CTA Buttons */}
            <div className="flex flex-wrap items-center justify-center gap-4">
              <a
                href="https://aclanthology.org/2025.emnlp-main.1666/"
                target="_blank"
                rel="noopener noreferrer"
                className={`group relative inline-flex items-center gap-2.5 px-7 py-3.5 rounded-xl font-semibold text-sm transition-all duration-300 overflow-hidden ${
                  isDark
                    ? 'bg-gradient-to-r from-electric-500 to-cyan-500 text-white hover:shadow-xl hover:shadow-electric-500/30 hover:scale-[1.02]'
                    : 'bg-gradient-to-r from-navy-600 to-electric-600 text-white hover:shadow-xl hover:shadow-navy-500/30 hover:scale-[1.02]'
                }`}
              >
                <div
                  className="absolute inset-0 opacity-20 group-hover:opacity-30 transition-opacity"
                  style={{
                    background: 'linear-gradient(110deg, transparent 25%, rgba(255,255,255,0.3) 50%, transparent 75%)',
                    backgroundSize: '200% 100%',
                    animation: 'shimmer 3s ease-in-out infinite',
                  }}
                />
                <span className="relative z-10 flex items-center gap-2.5">
                  <FaFileAlt size={14} />
                  Read the Paper
                  <FaExternalLinkAlt size={10} className="opacity-70" />
                </span>
              </a>
              <Link
                to="/docs"
                className={`group inline-flex items-center gap-2.5 px-7 py-3.5 rounded-xl font-semibold text-sm transition-all duration-300 ${
                  isDark
                    ? 'bg-white/[0.04] text-white border border-white/[0.08] hover:bg-white/[0.08] hover:border-electric-500/20'
                    : 'bg-white text-navy-900 border border-navy-200 hover:bg-navy-50 hover:border-navy-300'
                }`}
              >
                Get Started
                <FaArrowRight className="group-hover:translate-x-1 transition-transform" size={12} />
              </Link>
              <a
                href="https://github.com/ctrl-gaurav/Debate-Train-Evolve"
                target="_blank"
                rel="noopener noreferrer"
                className={`inline-flex items-center gap-2.5 px-7 py-3.5 rounded-xl font-semibold text-sm transition-all duration-300 ${
                  isDark
                    ? 'bg-white/[0.04] text-white border border-white/[0.08] hover:bg-white/[0.08] hover:border-electric-500/20'
                    : 'bg-white text-navy-900 border border-navy-200 hover:bg-navy-50 hover:border-navy-300'
                }`}
              >
                <FaGithub size={16} />
                GitHub
              </a>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Stats Section */}
      <section className="py-20">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 sm:gap-6">
            <StatCard value={8.92} suffix="%" label="Avg. accuracy gain on GSM-Plus" delay={0} />
            <StatCard value={5.8} suffix="%" label="Cross-domain generalization" delay={0.1} />
            <StatCard value={7} suffix="" label="Reasoning benchmarks" delay={0.2} />
            <StatCard value={6} suffix="" label="Open-weight models tested" delay={0.3} />
          </div>
        </div>
      </section>

      {/* Pipeline Visual */}
      <section className="py-20 relative">
        {/* Section background accent */}
        <div className={`absolute inset-0 pointer-events-none ${
          isDark ? 'bg-gradient-to-b from-transparent via-navy-900/20 to-transparent' : 'bg-gradient-to-b from-transparent via-navy-50/40 to-transparent'
        }`} />
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10">
          <SectionHeading
            badge="Framework"
            title="Three-Phase Pipeline"
            subtitle="DTE iteratively improves a single model by generating multi-agent debate traces, training on them, and evolving the model."
          />

          <div className="grid grid-cols-1 md:grid-cols-5 gap-4 sm:gap-6 mt-12 items-start">
            <PipelineStep
              title="Debate"
              desc="N identical LLM copies debate using RCR prompting. Each round: agents reflect, critique 2 peers, and refine their answer until consensus."
              icon={
                <svg width="32" height="32" viewBox="0 0 32 32" fill="none" stroke="currentColor" strokeWidth="1.5">
                  <circle cx="10" cy="10" r="4" />
                  <circle cx="22" cy="10" r="4" />
                  <circle cx="16" cy="22" r="4" />
                  <path d="M13.5 11.5L13 20" strokeDasharray="2 2" />
                  <path d="M18.5 11.5L19 20" strokeDasharray="2 2" />
                  <path d="M12 8.5L20 8.5" strokeDasharray="2 2" />
                </svg>
              }
              index={0}
              isDark={isDark}
            />
            {/* Arrow */}
            <div className="hidden md:flex items-center justify-center pt-12">
              <motion.div
                initial={{ scaleX: 0 }}
                whileInView={{ scaleX: 1 }}
                viewport={{ once: true }}
                transition={{ duration: 0.6, delay: 0.3 }}
                className="flex items-center w-full"
              >
                <div className={`flex-1 h-px ${isDark ? 'bg-gradient-to-r from-electric-500/20 to-electric-500/40' : 'bg-gradient-to-r from-navy-400/20 to-navy-400/40'}`} />
                <svg className={`w-5 h-5 -ml-1 ${isDark ? 'text-electric-500/50' : 'text-navy-400/50'}`} fill="currentColor" viewBox="0 0 20 20"><path d="M10.293 3.293a1 1 0 011.414 0l6 6a1 1 0 010 1.414l-6 6a1 1 0 01-1.414-1.414L14.586 11H3a1 1 0 110-2h11.586l-4.293-4.293a1 1 0 010-1.414z"/></svg>
              </motion.div>
            </div>

            <PipelineStep
              title="Train"
              desc="Debate traces fine-tune the model via GRPO with 5 shaped reward functions -- no ground truth labels needed."
              icon={
                <svg width="32" height="32" viewBox="0 0 32 32" fill="none" stroke="currentColor" strokeWidth="1.5">
                  <path d="M4 24L10 16L16 20L22 8L28 12" />
                  <circle cx="28" cy="12" r="2" fill="currentColor" />
                  <path d="M4 28H28" strokeOpacity="0.3" />
                </svg>
              }
              index={1}
              isDark={isDark}
            />

            <div className="hidden md:flex items-center justify-center pt-12">
              <motion.div
                initial={{ scaleX: 0 }}
                whileInView={{ scaleX: 1 }}
                viewport={{ once: true }}
                transition={{ duration: 0.6, delay: 0.45 }}
                className="flex items-center w-full"
              >
                <div className={`flex-1 h-px ${isDark ? 'bg-gradient-to-r from-electric-500/20 to-electric-500/40' : 'bg-gradient-to-r from-navy-400/20 to-navy-400/40'}`} />
                <svg className={`w-5 h-5 -ml-1 ${isDark ? 'text-electric-500/50' : 'text-navy-400/50'}`} fill="currentColor" viewBox="0 0 20 20"><path d="M10.293 3.293a1 1 0 011.414 0l6 6a1 1 0 010 1.414l-6 6a1 1 0 01-1.414-1.414L14.586 11H3a1 1 0 110-2h11.586l-4.293-4.293a1 1 0 010-1.414z"/></svg>
              </motion.div>
            </div>

            <PipelineStep
              title="Evolve"
              desc="The trained model replaces agents in the debate ensemble. The process repeats until convergence."
              icon={
                <svg width="32" height="32" viewBox="0 0 32 32" fill="none" stroke="currentColor" strokeWidth="1.5">
                  <path d="M16 4C22.627 4 28 9.373 28 16" />
                  <path d="M16 28C9.373 28 4 22.627 4 16" />
                  <path d="M26 12L28 16L24 16" fill="currentColor" stroke="none" />
                  <path d="M6 20L4 16L8 16" fill="currentColor" stroke="none" />
                  <circle cx="16" cy="16" r="3" />
                </svg>
              }
              index={2}
              isDark={isDark}
            />
          </div>
        </div>
      </section>

      {/* Key Features */}
      <section className="py-20">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
          <SectionHeading
            badge="Innovation"
            title="Key Contributions"
            subtitle="What makes DTE unique in the landscape of LLM training."
          />

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mt-12">
            <FeatureCard
              title="Ground-Truth-Free Training"
              description="DTE requires no labeled data. The consensus answer from multi-agent debate serves as the training signal, enabling fully autonomous model improvement."
              delay={0}
            />
            <FeatureCard
              title="Reflect-Critique-Refine (RCR)"
              description="A new prompting strategy that forces agents to identify errors, critique 2 peers, and provide novel reasoning -- reducing sycophancy by 50%."
              delay={0.08}
            />
            <FeatureCard
              title="GRPO with Shaped Rewards"
              description="Group Relative Policy Optimization with 5 reward functions (correctness, format, numeric, XML structure) eliminates the need for a separate value network."
              delay={0.16}
            />
            <FeatureCard
              title="Single-Model Inference"
              description="After training, the evolved model runs as a single model with standard inference latency -- no multi-agent overhead at deployment time."
              delay={0.24}
            />
            <FeatureCard
              title="Cross-Domain Generalization"
              description="Models trained on math benchmarks show +5.8% average improvement on unseen science and commonsense reasoning tasks."
              delay={0.32}
            />
            <FeatureCard
              title="Temperature Annealing"
              description="Controlled temperature decay from 0.7 to 0.3 for smaller models prevents catastrophic forgetting and recovers up to 76% of lost performance."
              delay={0.4}
            />
          </div>
        </div>
      </section>

      {/* Models & Benchmarks overview */}
      <section className="py-20 relative">
        <div className={`absolute inset-0 pointer-events-none ${
          isDark ? 'bg-gradient-to-b from-transparent via-navy-900/20 to-transparent' : 'bg-gradient-to-b from-transparent via-navy-50/40 to-transparent'
        }`} />
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10">
          <SectionHeading
            badge="Scale"
            title="Comprehensive Evaluation"
            subtitle="Tested across 6 open-weight models and 7 diverse reasoning benchmarks."
          />

          <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mt-12">
            {/* Models */}
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5 }}
              className={`p-6 rounded-2xl ${isDark ? 'glass-card' : 'glass-card-light'}`}
            >
              <h3 className={`text-lg font-display font-semibold mb-4 ${isDark ? 'text-white' : 'text-navy-900'}`}>
                Models
              </h3>
              <div className="space-y-2.5">
                {[
                  { name: 'Qwen-2.5-1.5B', family: 'Qwen' },
                  { name: 'Qwen-2.5-3B', family: 'Qwen' },
                  { name: 'Qwen-2.5-7B', family: 'Qwen' },
                  { name: 'Qwen-2.5-14B', family: 'Qwen' },
                  { name: 'Llama-3.2-3B', family: 'Llama' },
                  { name: 'Llama-3.1-8B', family: 'Llama' },
                ].map((m) => (
                  <div key={m.name} className={`flex items-center justify-between px-4 py-2.5 rounded-xl transition-colors duration-200 ${
                    isDark ? 'bg-white/[0.02] hover:bg-white/[0.05]' : 'bg-navy-50/40 hover:bg-navy-50/70'
                  }`}>
                    <span className={`text-sm font-medium ${isDark ? 'text-gray-300' : 'text-navy-800'}`}>
                      {m.name}
                    </span>
                    <span className={`text-xs font-mono px-2 py-0.5 rounded-md ${
                      isDark ? 'bg-electric-500/10 text-electric-300' : 'bg-navy-100 text-navy-600'
                    }`}>
                      {m.family}
                    </span>
                  </div>
                ))}
              </div>
            </motion.div>

            {/* Benchmarks */}
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5 }}
              className={`p-6 rounded-2xl ${isDark ? 'glass-card' : 'glass-card-light'}`}
            >
              <h3 className={`text-lg font-display font-semibold mb-4 ${isDark ? 'text-white' : 'text-navy-900'}`}>
                Benchmarks
              </h3>
              <div className="space-y-2.5">
                {[
                  { name: 'GSM8K', type: 'Math', color: 'electric' },
                  { name: 'GSM-Plus', type: 'Math', color: 'electric' },
                  { name: 'MATH', type: 'Math', color: 'electric' },
                  { name: 'ARC-Easy', type: 'Science', color: 'cyan' },
                  { name: 'ARC-Challenge', type: 'Science', color: 'cyan' },
                  { name: 'GPQA Main', type: 'STEM', color: 'cyan' },
                  { name: 'CommonsenseQA', type: 'Commonsense', color: 'cyan' },
                ].map((b) => (
                  <div key={b.name} className={`flex items-center justify-between px-4 py-2.5 rounded-xl transition-colors duration-200 ${
                    isDark ? 'bg-white/[0.02] hover:bg-white/[0.05]' : 'bg-navy-50/40 hover:bg-navy-50/70'
                  }`}>
                    <span className={`text-sm font-medium ${isDark ? 'text-gray-300' : 'text-navy-800'}`}>
                      {b.name}
                    </span>
                    <span className={`text-xs font-mono px-2 py-0.5 rounded-md ${
                      b.color === 'electric'
                        ? isDark ? 'bg-electric-500/10 text-electric-300' : 'bg-blue-50 text-blue-600'
                        : isDark ? 'bg-cyan-500/10 text-cyan-300' : 'bg-teal-50 text-teal-600'
                    }`}>
                      {b.type}
                    </span>
                  </div>
                ))}
              </div>
            </motion.div>
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="py-20">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className={`p-8 sm:p-12 rounded-3xl text-center relative overflow-hidden ${
              isDark
                ? 'bg-gradient-to-br from-navy-900/60 to-navy-950/60 border border-white/[0.06]'
                : 'bg-gradient-to-br from-navy-50 to-electric-50 border border-navy-200/30'
            }`}
          >
            <div className={`orb w-48 h-48 -top-24 -right-24 ${isDark ? 'bg-electric-500/10' : 'bg-electric-500/[0.06]'}`} />
            <div className={`orb w-36 h-36 -bottom-18 -left-18 ${isDark ? 'bg-cyan-500/[0.08]' : 'bg-cyan-500/[0.05]'}`} />
            <h2 className={`text-2xl sm:text-3xl font-display font-bold mb-4 relative z-10 ${
              isDark ? 'text-white' : 'text-navy-900'
            }`}>
              Ready to Evolve Your Models?
            </h2>
            <p className={`text-base mb-8 max-w-xl mx-auto relative z-10 ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>
              Get started with the DTE framework in minutes. Full documentation, examples, and pre-trained checkpoints available.
            </p>
            <div className="flex flex-wrap items-center justify-center gap-4 relative z-10">
              <Link
                to="/docs"
                className={`inline-flex items-center gap-2 px-7 py-3.5 rounded-xl font-semibold text-sm transition-all duration-300 ${
                  isDark
                    ? 'bg-electric-500 text-white hover:bg-electric-600 hover:shadow-lg hover:shadow-electric-500/20'
                    : 'bg-navy-600 text-white hover:bg-navy-700 hover:shadow-lg hover:shadow-navy-600/20'
                }`}
              >
                Documentation <FaArrowRight size={12} />
              </Link>
              <Link
                to="/results"
                className={`inline-flex items-center gap-2 px-7 py-3.5 rounded-xl font-semibold text-sm transition-all duration-300 ${
                  isDark
                    ? 'bg-white/[0.04] text-white border border-white/[0.08] hover:bg-white/[0.08]'
                    : 'bg-white text-navy-900 border border-navy-200 hover:bg-navy-50'
                }`}
              >
                View Results
              </Link>
            </div>
          </motion.div>
        </div>
      </section>
    </PageWrapper>
  )
}

import { useState, useMemo } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { useTheme } from '../context/ThemeContext'
import PageWrapper from '../components/PageWrapper'
import CodeBlock from '../components/CodeBlock'

/* ============ TYPES ============ */
interface Section {
  id: string
  title: string
  icon: string
  content: React.ReactNode
}

/* ============ COLLAPSIBLE ============ */
function Collapsible({ title, children, defaultOpen = false, isDark }: {
  title: string; children: React.ReactNode; defaultOpen?: boolean; isDark: boolean
}) {
  const [open, setOpen] = useState(defaultOpen)
  return (
    <div className={`rounded-xl overflow-hidden border transition-colors duration-300 ${isDark ? 'border-white/[0.06] bg-white/[0.02]' : 'border-navy-200/30 bg-navy-50/30'} ${open ? isDark ? 'border-electric-500/15' : 'border-navy-300/40' : ''}`}>
      <button
        onClick={() => setOpen(!open)}
        className={`w-full flex items-center justify-between px-5 py-3.5 text-left text-sm font-semibold transition-colors ${
          isDark ? 'text-gray-200 hover:bg-white/3' : 'text-navy-800 hover:bg-navy-50'
        }`}
      >
        {title}
        <svg className={`w-4 h-4 transition-transform duration-200 ${open ? 'rotate-180' : ''}`} fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><path d="M19 9l-7 7-7-7" /></svg>
      </button>
      <AnimatePresence>
        {open && (
          <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: 'auto', opacity: 1 }} exit={{ height: 0, opacity: 0 }} transition={{ duration: 0.25 }}>
            <div className={`px-5 pb-4 ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>{children}</div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

/* ============ INLINE TIP/NOTE ============ */
function Note({ children, type = 'info', isDark }: {
  children: React.ReactNode; type?: 'info' | 'warning' | 'tip'; isDark: boolean
}) {
  const styles = {
    info: isDark ? 'border-electric-500/30 bg-electric-500/[0.05]' : 'border-navy-300/40 bg-navy-50/50',
    warning: isDark ? 'border-amber-500/30 bg-amber-500/[0.05]' : 'border-amber-300/40 bg-amber-50/50',
    tip: isDark ? 'border-green-500/30 bg-green-500/[0.05]' : 'border-green-300/40 bg-green-50/50',
  }
  const icons = { info: 'i', warning: '!', tip: '*' }
  const labels = { info: 'Note', warning: 'Warning', tip: 'Tip' }
  const labelColors = {
    info: isDark ? 'text-electric-300' : 'text-navy-600',
    warning: isDark ? 'text-amber-300' : 'text-amber-600',
    tip: isDark ? 'text-green-300' : 'text-green-600',
  }
  return (
    <div className={`rounded-xl border-l-4 p-4 ${styles[type]}`}>
      <div className={`text-xs font-mono font-bold uppercase tracking-wider mb-1 flex items-center gap-1.5 ${labelColors[type]}`}><span className="opacity-60">{icons[type]}</span> {labels[type]}</div>
      <div className={`text-sm leading-relaxed ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>{children}</div>
    </div>
  )
}

/* ============ ARCHITECTURE DIAGRAM (SVG) ============ */
function ArchitectureDiagram({ isDark }: { isDark: boolean }) {
  const boxFill = isDark ? 'rgba(255,255,255,0.04)' : 'rgba(0,0,0,0.02)'
  const boxStroke = isDark ? 'rgba(77,195,255,0.3)' : 'rgba(42,77,255,0.25)'
  const textColor = isDark ? '#e5e7eb' : '#1e293b'
  const subTextColor = isDark ? '#9ca3af' : '#64748b'
  const arrowColor = isDark ? 'rgba(77,195,255,0.5)' : 'rgba(42,77,255,0.4)'
  const accentColor = isDark ? '#4dc3ff' : '#2a4dff'

  return (
    <svg viewBox="0 0 800 320" className="w-full" style={{ maxWidth: 800 }}>
      <defs>
        <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="10" refY="3.5" orient="auto">
          <polygon points="0 0, 10 3.5, 0 7" fill={arrowColor} />
        </marker>
      </defs>

      {/* Phase 1: Debate */}
      <rect x="20" y="40" width="220" height="240" rx="16" fill={boxFill} stroke={boxStroke} strokeWidth="1.5" />
      <text x="130" y="75" textAnchor="middle" fill={accentColor} fontSize="13" fontWeight="700" fontFamily="monospace">PHASE 1</text>
      <text x="130" y="100" textAnchor="middle" fill={textColor} fontSize="16" fontWeight="700">Debate</text>
      <circle cx="80" cy="150" r="22" fill={boxFill} stroke={boxStroke} strokeWidth="1" />
      <text x="80" y="155" textAnchor="middle" fill={subTextColor} fontSize="10" fontWeight="600">Agent 1</text>
      <circle cx="180" cy="150" r="22" fill={boxFill} stroke={boxStroke} strokeWidth="1" />
      <text x="180" y="155" textAnchor="middle" fill={subTextColor} fontSize="10" fontWeight="600">Agent 2</text>
      <circle cx="130" cy="210" r="22" fill={boxFill} stroke={boxStroke} strokeWidth="1" />
      <text x="130" y="215" textAnchor="middle" fill={subTextColor} fontSize="10" fontWeight="600">Agent 3</text>
      <line x1="100" y1="145" x2="158" y2="145" stroke={arrowColor} strokeWidth="1" strokeDasharray="4 3" />
      <line x1="92" y1="168" x2="118" y2="195" stroke={arrowColor} strokeWidth="1" strokeDasharray="4 3" />
      <line x1="168" y1="168" x2="142" y2="195" stroke={arrowColor} strokeWidth="1" strokeDasharray="4 3" />
      <text x="130" y="260" textAnchor="middle" fill={subTextColor} fontSize="11">RCR Prompting</text>

      {/* Arrow 1->2 */}
      <line x1="250" y1="160" x2="295" y2="160" stroke={arrowColor} strokeWidth="2" markerEnd="url(#arrowhead)" />

      {/* Phase 2: Train */}
      <rect x="305" y="40" width="220" height="240" rx="16" fill={boxFill} stroke={boxStroke} strokeWidth="1.5" />
      <text x="415" y="75" textAnchor="middle" fill={accentColor} fontSize="13" fontWeight="700" fontFamily="monospace">PHASE 2</text>
      <text x="415" y="100" textAnchor="middle" fill={textColor} fontSize="16" fontWeight="700">Train (GRPO)</text>
      <rect x="340" y="125" width="150" height="35" rx="8" fill={boxFill} stroke={boxStroke} strokeWidth="1" />
      <text x="415" y="147" textAnchor="middle" fill={subTextColor} fontSize="11">Debate Traces</text>
      <rect x="340" y="175" width="150" height="35" rx="8" fill={boxFill} stroke={boxStroke} strokeWidth="1" />
      <text x="415" y="197" textAnchor="middle" fill={subTextColor} fontSize="11">5 Reward Functions</text>
      <rect x="340" y="225" width="150" height="35" rx="8" fill={boxFill} stroke={boxStroke} strokeWidth="1" />
      <text x="415" y="247" textAnchor="middle" fill={subTextColor} fontSize="11">LoRA Fine-tuning</text>

      {/* Arrow 2->3 */}
      <line x1="535" y1="160" x2="580" y2="160" stroke={arrowColor} strokeWidth="2" markerEnd="url(#arrowhead)" />

      {/* Phase 3: Evolve */}
      <rect x="590" y="40" width="190" height="240" rx="16" fill={boxFill} stroke={boxStroke} strokeWidth="1.5" />
      <text x="685" y="75" textAnchor="middle" fill={accentColor} fontSize="13" fontWeight="700" fontFamily="monospace">PHASE 3</text>
      <text x="685" y="100" textAnchor="middle" fill={textColor} fontSize="16" fontWeight="700">Evolve</text>
      <rect x="620" y="130" width="130" height="50" rx="10" fill={boxFill} stroke={accentColor} strokeWidth="1.5" />
      <text x="685" y="152" textAnchor="middle" fill={textColor} fontSize="12" fontWeight="600">Evolved Model</text>
      <text x="685" y="167" textAnchor="middle" fill={subTextColor} fontSize="10">replaces agents</text>
      <path d="M685 195 C685 230, 750 250, 750 200 C750 180, 720 190, 720 195" fill="none" stroke={arrowColor} strokeWidth="1.5" strokeDasharray="4 3" />
      <text x="685" y="260" textAnchor="middle" fill={subTextColor} fontSize="11">Repeat until</text>
      <text x="685" y="274" textAnchor="middle" fill={subTextColor} fontSize="11">convergence</text>

      {/* Evolution loop arrow (back from Evolve to Debate) */}
      <path d="M 685 290 Q 685 310, 400 310 Q 130 310, 130 288" fill="none" stroke={arrowColor} strokeWidth="1.5" strokeDasharray="6 4" markerEnd="url(#arrowhead)" />
      <text x="400" y="306" textAnchor="middle" fill={accentColor} fontSize="10" fontWeight="600">Next Evolution Round</text>
    </svg>
  )
}

/* ============ RCR FLOW DIAGRAM (SVG) ============ */
function RCRFlowDiagram({ isDark }: { isDark: boolean }) {
  const boxFill = isDark ? 'rgba(255,255,255,0.04)' : 'rgba(0,0,0,0.02)'
  const boxStroke = isDark ? 'rgba(77,195,255,0.3)' : 'rgba(42,77,255,0.25)'
  const textColor = isDark ? '#e5e7eb' : '#1e293b'
  const subTextColor = isDark ? '#9ca3af' : '#64748b'
  const arrowColor = isDark ? 'rgba(77,195,255,0.5)' : 'rgba(42,77,255,0.4)'
  const accentColor = isDark ? '#4dc3ff' : '#2a4dff'
  const glowColor = isDark ? 'rgba(77,195,255,0.15)' : 'rgba(42,77,255,0.08)'

  return (
    <svg viewBox="0 0 780 260" className="w-full" style={{ maxWidth: 780 }}>
      <defs>
        <marker id="rcr-arrow" markerWidth="10" markerHeight="7" refX="10" refY="3.5" orient="auto">
          <polygon points="0 0, 10 3.5, 0 7" fill={arrowColor} />
        </marker>
        <linearGradient id="rcr-grad1" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stopColor={isDark ? 'rgba(77,195,255,0.12)' : 'rgba(42,77,255,0.06)'} />
          <stop offset="100%" stopColor={isDark ? 'rgba(139,92,246,0.12)' : 'rgba(139,92,246,0.06)'} />
        </linearGradient>
      </defs>

      {/* Background glow */}
      <rect x="0" y="0" width="780" height="260" rx="20" fill="url(#rcr-grad1)" />

      {/* Step 1: Reflect */}
      <rect x="30" y="50" width="200" height="110" rx="14" fill={boxFill} stroke={boxStroke} strokeWidth="1.5" />
      <rect x="30" y="50" width="200" height="4" rx="2" fill={accentColor} opacity="0.6" />
      <text x="130" y="82" textAnchor="middle" fill={accentColor} fontSize="11" fontWeight="700" fontFamily="monospace">STEP 1</text>
      <text x="130" y="106" textAnchor="middle" fill={textColor} fontSize="15" fontWeight="700">Reflect</text>
      <text x="130" y="126" textAnchor="middle" fill={subTextColor} fontSize="11">(Self-Critique)</text>
      <text x="130" y="148" textAnchor="middle" fill={subTextColor} fontSize="10">Each agent evaluates</text>
      <text x="130" y="161" textAnchor="middle" fill={subTextColor} fontSize="10">its own reasoning</text>

      {/* Arrow 1->2 */}
      <line x1="240" y1="105" x2="285" y2="105" stroke={arrowColor} strokeWidth="2" markerEnd="url(#rcr-arrow)" />

      {/* Step 2: Critique */}
      <rect x="295" y="50" width="200" height="110" rx="14" fill={boxFill} stroke={boxStroke} strokeWidth="1.5" />
      <rect x="295" y="50" width="200" height="4" rx="2" fill={isDark ? '#a78bfa' : '#7c3aed'} opacity="0.6" />
      <text x="395" y="82" textAnchor="middle" fill={isDark ? '#a78bfa' : '#7c3aed'} fontSize="11" fontWeight="700" fontFamily="monospace">STEP 2</text>
      <text x="395" y="106" textAnchor="middle" fill={textColor} fontSize="15" fontWeight="700">Critique</text>
      <text x="395" y="126" textAnchor="middle" fill={subTextColor} fontSize="11">(2 Peers)</text>
      <text x="395" y="148" textAnchor="middle" fill={subTextColor} fontSize="10">Explicitly critique</text>
      <text x="395" y="161" textAnchor="middle" fill={subTextColor} fontSize="10">two peer responses</text>

      {/* Arrow 2->3 */}
      <line x1="505" y1="105" x2="550" y2="105" stroke={arrowColor} strokeWidth="2" markerEnd="url(#rcr-arrow)" />

      {/* Step 3: Refine */}
      <rect x="560" y="50" width="200" height="110" rx="14" fill={boxFill} stroke={boxStroke} strokeWidth="1.5" />
      <rect x="560" y="50" width="200" height="4" rx="2" fill={isDark ? '#34d399' : '#059669'} opacity="0.6" />
      <text x="660" y="82" textAnchor="middle" fill={isDark ? '#34d399' : '#059669'} fontSize="11" fontWeight="700" fontFamily="monospace">STEP 3</text>
      <text x="660" y="106" textAnchor="middle" fill={textColor} fontSize="15" fontWeight="700">Refine</text>
      <text x="660" y="126" textAnchor="middle" fill={subTextColor} fontSize="11">(Update Answer)</text>
      <text x="660" y="148" textAnchor="middle" fill={subTextColor} fontSize="10">Revise with novel</text>
      <text x="660" y="161" textAnchor="middle" fill={subTextColor} fontSize="10">reasoning required</text>

      {/* Cycle arrow from Refine back to Reflect */}
      <path d="M 660 170 Q 660 220, 390 225 Q 130 225, 130 168" fill="none" stroke={arrowColor} strokeWidth="1.5" strokeDasharray="6 4" markerEnd="url(#rcr-arrow)" />
      <text x="390" y="218" textAnchor="middle" fill={accentColor} fontSize="10" fontWeight="600">Next Round</text>

      {/* Annotation */}
      <rect x="280" y="230" width="230" height="24" rx="12" fill={glowColor} stroke={accentColor} strokeWidth="0.8" />
      <text x="395" y="246" textAnchor="middle" fill={accentColor} fontSize="10" fontWeight="600">Novel Reasoning Required on Change</text>
    </svg>
  )
}

/* ============ REWARD STACK DIAGRAM (SVG) ============ */
function RewardStackDiagram({ isDark }: { isDark: boolean }) {
  const textColor = isDark ? '#e5e7eb' : '#1e293b'
  const subTextColor = isDark ? '#9ca3af' : '#64748b'
  const bgFill = isDark ? 'rgba(255,255,255,0.02)' : 'rgba(0,0,0,0.01)'

  const rewards = [
    { label: 'Correctness', weight: 2.0, fraction: 0.5, color: isDark ? '#34d399' : '#059669', lightBg: isDark ? 'rgba(52,211,153,0.15)' : 'rgba(5,150,105,0.1)' },
    { label: 'Int Format', weight: 0.5, fraction: 0.125, color: isDark ? '#4dc3ff' : '#2a4dff', lightBg: isDark ? 'rgba(77,195,255,0.15)' : 'rgba(42,77,255,0.1)' },
    { label: 'Strict XML', weight: 0.5, fraction: 0.125, color: isDark ? '#67e8f9' : '#0891b2', lightBg: isDark ? 'rgba(103,232,249,0.15)' : 'rgba(8,145,178,0.1)' },
    { label: 'Soft XML', weight: 0.5, fraction: 0.125, color: isDark ? '#a78bfa' : '#7c3aed', lightBg: isDark ? 'rgba(167,139,250,0.15)' : 'rgba(124,58,237,0.1)' },
    { label: 'XML Count', weight: 0.5, fraction: 0.125, color: isDark ? '#fbbf24' : '#d97706', lightBg: isDark ? 'rgba(251,191,36,0.15)' : 'rgba(217,119,6,0.1)' },
  ]

  const barStartX = 80
  const barWidth = 580
  const barY = 55
  const barHeight = 50

  let cumulativeX = barStartX

  return (
    <svg viewBox="0 0 750 180" className="w-full" style={{ maxWidth: 750 }}>
      {/* Background */}
      <rect x="0" y="0" width="750" height="180" rx="16" fill={bgFill} />

      {/* Title */}
      <text x="375" y="30" textAnchor="middle" fill={textColor} fontSize="14" fontWeight="700">Reward Distribution (max = 4.0)</text>

      {/* Stacked horizontal bar */}
      {rewards.map((r, i) => {
        const segWidth = r.fraction * barWidth
        const x = cumulativeX
        cumulativeX += segWidth
        const isFirst = i === 0
        const isLast = i === rewards.length - 1
        return (
          <g key={r.label}>
            <rect
              x={x} y={barY} width={segWidth} height={barHeight}
              rx={isFirst ? 10 : isLast ? 10 : 0}
              fill={r.lightBg}
              stroke={r.color}
              strokeWidth="1.5"
            />
            <text x={x + segWidth / 2} y={barY + 22} textAnchor="middle" fill={r.color} fontSize="10" fontWeight="700">{r.label}</text>
            <text x={x + segWidth / 2} y={barY + 40} textAnchor="middle" fill={r.color} fontSize="13" fontWeight="800">{r.weight}</text>
          </g>
        )
      })}

      {/* Total label */}
      <text x={barStartX + barWidth + 30} y={barY + 32} textAnchor="middle" fill={textColor} fontSize="16" fontWeight="800">= 4.0</text>

      {/* Bottom legend */}
      {rewards.map((r, i) => (
        <g key={`legend-${r.label}`}>
          <rect x={30 + i * 145} y={130} width="12" height="12" rx="3" fill={r.lightBg} stroke={r.color} strokeWidth="1" />
          <text x={48 + i * 145} y={141} fill={subTextColor} fontSize="10">{r.label} ({r.weight})</text>
        </g>
      ))}
    </svg>
  )
}

/* ============ GRPO TRAINING DIAGRAM (SVG) ============ */
function GRPOTrainingDiagram({ isDark }: { isDark: boolean }) {
  const boxFill = isDark ? 'rgba(255,255,255,0.04)' : 'rgba(0,0,0,0.02)'
  const boxStroke = isDark ? 'rgba(77,195,255,0.3)' : 'rgba(42,77,255,0.25)'
  const textColor = isDark ? '#e5e7eb' : '#1e293b'
  const subTextColor = isDark ? '#9ca3af' : '#64748b'
  const arrowColor = isDark ? 'rgba(77,195,255,0.5)' : 'rgba(42,77,255,0.4)'
  const accentColor = isDark ? '#4dc3ff' : '#2a4dff'
  const warnColor = isDark ? '#fbbf24' : '#d97706'

  return (
    <svg viewBox="0 0 820 280" className="w-full" style={{ maxWidth: 820 }}>
      <defs>
        <marker id="grpo-arrow" markerWidth="10" markerHeight="7" refX="10" refY="3.5" orient="auto">
          <polygon points="0 0, 10 3.5, 0 7" fill={arrowColor} />
        </marker>
        <marker id="grpo-arrow-warn" markerWidth="10" markerHeight="7" refX="10" refY="3.5" orient="auto">
          <polygon points="0 0, 10 3.5, 0 7" fill={warnColor} />
        </marker>
        <linearGradient id="grpo-bg" x1="0" y1="0" x2="1" y2="0.5">
          <stop offset="0%" stopColor={isDark ? 'rgba(77,195,255,0.06)' : 'rgba(42,77,255,0.03)'} />
          <stop offset="100%" stopColor={isDark ? 'rgba(139,92,246,0.06)' : 'rgba(139,92,246,0.03)'} />
        </linearGradient>
      </defs>

      <rect x="0" y="0" width="820" height="280" rx="20" fill="url(#grpo-bg)" />

      {/* Title */}
      <text x="410" y="30" textAnchor="middle" fill={textColor} fontSize="14" fontWeight="700">GRPO Training Loop</text>

      {/* Box 1: Debate Traces */}
      <rect x="20" y="60" width="130" height="80" rx="12" fill={boxFill} stroke={boxStroke} strokeWidth="1.5" />
      <text x="85" y="95" textAnchor="middle" fill={textColor} fontSize="12" fontWeight="700">Debate</text>
      <text x="85" y="112" textAnchor="middle" fill={textColor} fontSize="12" fontWeight="700">Traces</text>
      <text x="85" y="132" textAnchor="middle" fill={subTextColor} fontSize="9">JSONL data</text>

      {/* Arrow */}
      <line x1="155" y1="100" x2="185" y2="100" stroke={arrowColor} strokeWidth="2" markerEnd="url(#grpo-arrow)" />

      {/* Box 2: Group Sampling */}
      <rect x="190" y="60" width="140" height="80" rx="12" fill={boxFill} stroke={boxStroke} strokeWidth="1.5" />
      <text x="260" y="92" textAnchor="middle" fill={textColor} fontSize="12" fontWeight="700">Group Sampling</text>
      <rect x="220" y="108" width="80" height="22" rx="6" fill={isDark ? 'rgba(77,195,255,0.1)' : 'rgba(42,77,255,0.06)'} stroke={accentColor} strokeWidth="0.8" />
      <text x="260" y="123" textAnchor="middle" fill={accentColor} fontSize="11" fontWeight="700" fontFamily="monospace">k = 8</text>

      {/* Arrow */}
      <line x1="335" y1="100" x2="365" y2="100" stroke={arrowColor} strokeWidth="2" markerEnd="url(#grpo-arrow)" />

      {/* Box 3: Reward Scoring */}
      <rect x="370" y="60" width="130" height="80" rx="12" fill={boxFill} stroke={boxStroke} strokeWidth="1.5" />
      <text x="435" y="92" textAnchor="middle" fill={textColor} fontSize="12" fontWeight="700">Reward</text>
      <text x="435" y="109" textAnchor="middle" fill={textColor} fontSize="12" fontWeight="700">Scoring</text>
      <text x="435" y="132" textAnchor="middle" fill={subTextColor} fontSize="9">5 reward functions</text>

      {/* Arrow */}
      <line x1="505" y1="100" x2="535" y2="100" stroke={arrowColor} strokeWidth="2" markerEnd="url(#grpo-arrow)" />

      {/* Box 4: Advantage Computation */}
      <rect x="540" y="60" width="130" height="80" rx="12" fill={boxFill} stroke={boxStroke} strokeWidth="1.5" />
      <text x="605" y="92" textAnchor="middle" fill={textColor} fontSize="12" fontWeight="700">Advantage</text>
      <text x="605" y="109" textAnchor="middle" fill={textColor} fontSize="12" fontWeight="700">Computation</text>
      <text x="605" y="132" textAnchor="middle" fill={subTextColor} fontSize="9">Group-relative</text>

      {/* Arrow */}
      <line x1="675" y1="100" x2="705" y2="100" stroke={arrowColor} strokeWidth="2" markerEnd="url(#grpo-arrow)" />

      {/* Box 5: Policy Update */}
      <rect x="710" y="60" width="95" height="80" rx="12" fill={boxFill} stroke={accentColor} strokeWidth="2" />
      <text x="757" y="95" textAnchor="middle" fill={textColor} fontSize="12" fontWeight="700">Policy</text>
      <text x="757" y="112" textAnchor="middle" fill={textColor} fontSize="12" fontWeight="700">Update</text>
      <text x="757" y="132" textAnchor="middle" fill={subTextColor} fontSize="9">LoRA weights</text>

      {/* KL Penalty feedback arrow */}
      <path d="M 757 150 Q 757 210, 400 215 Q 85 215, 85 148" fill="none" stroke={warnColor} strokeWidth="1.5" strokeDasharray="6 4" markerEnd="url(#grpo-arrow-warn)" />
      <rect x="340" y="200" width="120" height="24" rx="12" fill={isDark ? 'rgba(251,191,36,0.1)' : 'rgba(217,119,6,0.06)'} stroke={warnColor} strokeWidth="0.8" />
      <text x="400" y="216" textAnchor="middle" fill={warnColor} fontSize="10" fontWeight="700">KL Penalty</text>
      <text x="400" y="248" textAnchor="middle" fill={subTextColor} fontSize="10">Prevents catastrophic deviation from reference policy</text>
    </svg>
  )
}

/* ============ EVOLUTION LOOP DIAGRAM (SVG) ============ */
function EvolutionLoopDiagram({ isDark }: { isDark: boolean }) {
  const boxFill = isDark ? 'rgba(255,255,255,0.04)' : 'rgba(0,0,0,0.02)'
  const boxStroke = isDark ? 'rgba(77,195,255,0.3)' : 'rgba(42,77,255,0.25)'
  const textColor = isDark ? '#e5e7eb' : '#1e293b'
  const subTextColor = isDark ? '#9ca3af' : '#64748b'
  const arrowColor = isDark ? 'rgba(77,195,255,0.5)' : 'rgba(42,77,255,0.4)'
  const accentColor = isDark ? '#4dc3ff' : '#2a4dff'
  const greenColor = isDark ? '#34d399' : '#059669'
  const diamondColor = isDark ? 'rgba(251,191,36,0.3)' : 'rgba(217,119,6,0.15)'
  const diamondStroke = isDark ? '#fbbf24' : '#d97706'

  return (
    <svg viewBox="0 0 820 220" className="w-full" style={{ maxWidth: 820 }}>
      <defs>
        <marker id="evo-arrow" markerWidth="10" markerHeight="7" refX="10" refY="3.5" orient="auto">
          <polygon points="0 0, 10 3.5, 0 7" fill={arrowColor} />
        </marker>
        <marker id="evo-arrow-green" markerWidth="10" markerHeight="7" refX="10" refY="3.5" orient="auto">
          <polygon points="0 0, 10 3.5, 0 7" fill={greenColor} />
        </marker>
      </defs>

      {/* Round 0 */}
      <rect x="10" y="50" width="90" height="50" rx="10" fill={boxFill} stroke={accentColor} strokeWidth="1.5" />
      <text x="55" y="72" textAnchor="middle" fill={accentColor} fontSize="9" fontWeight="700" fontFamily="monospace">ROUND 0</text>
      <text x="55" y="88" textAnchor="middle" fill={textColor} fontSize="11" fontWeight="600">Base Model</text>

      <line x1="105" y1="75" x2="125" y2="75" stroke={arrowColor} strokeWidth="1.5" markerEnd="url(#evo-arrow)" />

      {/* Debate box */}
      <rect x="130" y="55" width="70" height="40" rx="8" fill={boxFill} stroke={boxStroke} strokeWidth="1" />
      <text x="165" y="79" textAnchor="middle" fill={subTextColor} fontSize="10" fontWeight="600">Debate</text>

      <line x1="205" y1="75" x2="225" y2="75" stroke={arrowColor} strokeWidth="1.5" markerEnd="url(#evo-arrow)" />

      {/* Train box */}
      <rect x="230" y="55" width="70" height="40" rx="8" fill={boxFill} stroke={boxStroke} strokeWidth="1" />
      <text x="265" y="79" textAnchor="middle" fill={subTextColor} fontSize="10" fontWeight="600">Train</text>

      <line x1="305" y1="75" x2="325" y2="75" stroke={arrowColor} strokeWidth="1.5" markerEnd="url(#evo-arrow)" />

      {/* Convergence diamond 1 */}
      <polygon points="355,55 375,75 355,95 335,75" fill={diamondColor} stroke={diamondStroke} strokeWidth="1" />
      <text x="355" y="78" textAnchor="middle" fill={diamondStroke} fontSize="7" fontWeight="700">CHK</text>

      <line x1="380" y1="75" x2="400" y2="75" stroke={arrowColor} strokeWidth="1.5" markerEnd="url(#evo-arrow)" />

      {/* Round 1 */}
      <rect x="405" y="50" width="90" height="50" rx="10" fill={boxFill} stroke={accentColor} strokeWidth="1.5" />
      <text x="450" y="72" textAnchor="middle" fill={accentColor} fontSize="9" fontWeight="700" fontFamily="monospace">ROUND 1</text>
      <text x="450" y="88" textAnchor="middle" fill={textColor} fontSize="10" fontWeight="600">Evolved v1</text>

      <line x1="500" y1="75" x2="520" y2="75" stroke={arrowColor} strokeWidth="1.5" markerEnd="url(#evo-arrow)" />

      {/* Debate box 2 */}
      <rect x="525" y="55" width="70" height="40" rx="8" fill={boxFill} stroke={boxStroke} strokeWidth="1" />
      <text x="560" y="79" textAnchor="middle" fill={subTextColor} fontSize="10" fontWeight="600">Debate</text>

      <line x1="600" y1="75" x2="620" y2="75" stroke={arrowColor} strokeWidth="1.5" markerEnd="url(#evo-arrow)" />

      {/* Train box 2 */}
      <rect x="625" y="55" width="70" height="40" rx="8" fill={boxFill} stroke={boxStroke} strokeWidth="1" />
      <text x="660" y="79" textAnchor="middle" fill={subTextColor} fontSize="10" fontWeight="600">Train</text>

      <line x1="700" y1="75" x2="720" y2="75" stroke={arrowColor} strokeWidth="1.5" markerEnd="url(#evo-arrow)" />

      {/* Convergence diamond 2 */}
      <polygon points="750,55 770,75 750,95 730,75" fill={diamondColor} stroke={diamondStroke} strokeWidth="1" />
      <text x="750" y="78" textAnchor="middle" fill={diamondStroke} fontSize="7" fontWeight="700">CHK</text>

      {/* Converged output */}
      <line x1="750" y1="100" x2="750" y2="140" stroke={greenColor} strokeWidth="1.5" markerEnd="url(#evo-arrow-green)" />
      <rect x="700" y="145" width="100" height="40" rx="10" fill={isDark ? 'rgba(52,211,153,0.1)' : 'rgba(5,150,105,0.06)'} stroke={greenColor} strokeWidth="1.5" />
      <text x="750" y="163" textAnchor="middle" fill={greenColor} fontSize="10" fontWeight="700">Converged</text>
      <text x="750" y="177" textAnchor="middle" fill={greenColor} fontSize="9">Final Model</text>

      {/* Legend at bottom */}
      <rect x="20" y="130" width="12" height="12" rx="3" fill={diamondColor} stroke={diamondStroke} strokeWidth="0.8" />
      <text x="38" y="141" fill={subTextColor} fontSize="10">= Convergence Check (accuracy plateau or max rounds)</text>

      {/* Not converged label on diamonds */}
      <text x="355" y="115" textAnchor="middle" fill={subTextColor} fontSize="8">not converged</text>
      <text x="750" y="115" textAnchor="middle" fill={subTextColor} fontSize="8">converged?</text>

      {/* Dotted continuation hint */}
      <text x="795" y="78" fill={subTextColor} fontSize="14">...</text>
    </svg>
  )
}

/* ============ DOCS PAGE ============ */
export default function Docs() {
  const { isDark } = useTheme()
  const [activeSection, setActiveSection] = useState('installation')
  const [searchQuery, setSearchQuery] = useState('')

  const cardCls = `p-6 sm:p-8 rounded-2xl ${isDark ? 'glass-card' : 'glass-card-light'}`
  const headCls = `text-xl sm:text-2xl font-display font-bold mb-4 ${isDark ? 'text-white' : 'text-navy-900'}`
  const subheadCls = `text-lg font-display font-semibold mb-3 ${isDark ? 'text-white' : 'text-navy-900'}`
  const textCls = `text-sm leading-relaxed ${isDark ? 'text-gray-400' : 'text-gray-600'}`
  const labelCls = `text-xs font-mono uppercase tracking-wider mb-2 ${isDark ? 'text-electric-400' : 'text-navy-500'}`
  const inlineCodeCls = `font-mono text-xs px-1.5 py-0.5 rounded-md ${isDark ? 'bg-white/[0.06] text-electric-300 border border-white/[0.04]' : 'bg-navy-50 text-navy-600 border border-navy-200/20'}`

  /* ---- Section Definitions ---- */
  const sections: Section[] = useMemo(() => [
    /* ====== INSTALLATION ====== */
    {
      id: 'installation',
      title: 'Installation',
      icon: 'M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4',
      content: (
        <div className="space-y-6">
          <div className={cardCls}>
            <h2 className={headCls}>Installation</h2>
            <p className={textCls}>DTE requires Python 3.8+ and a CUDA-compatible GPU for training. Debate-only mode works on CPU with 8GB+ RAM.</p>
          </div>

          <div className={cardCls}>
            <h3 className={subheadCls}>Prerequisites</h3>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 mb-4">
              {[
                { label: 'Python', value: '3.8+' },
                { label: 'GPU', value: 'CUDA-compatible (for training)' },
                { label: 'RAM', value: '16GB+ (full pipeline)' },
                { label: 'Debate only', value: '8GB+ RAM, CPU OK' },
              ].map(p => (
                <div key={p.label} className={`flex items-center gap-3 px-4 py-2.5 rounded-lg ${isDark ? 'bg-white/3' : 'bg-navy-50/50'}`}>
                  <span className={`text-xs font-mono font-bold ${isDark ? 'text-electric-300' : 'text-navy-600'}`}>{p.label}</span>
                  <span className={`text-sm ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>{p.value}</span>
                </div>
              ))}
            </div>
          </div>

          <div className={cardCls}>
            <h3 className={subheadCls}>Install from Source (Recommended)</h3>
            <CodeBlock language="bash" title="terminal" code={`# Clone the repository
git clone https://github.com/ctrl-gaurav/Debate-Train-Evolve.git
cd Debate-Train-Evolve

# Create a virtual environment (recommended)
python -m venv dte_env
source dte_env/bin/activate  # On Windows: dte_env\\Scripts\\activate

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .`} />
          </div>

          <div className={cardCls}>
            <h3 className={subheadCls}>Install with pip</h3>
            <CodeBlock language="bash" title="terminal" code={`pip install dte-framework`} />
          </div>

          <div className={cardCls}>
            <h3 className={subheadCls}>GPU Support</h3>
            <p className={`${textCls} mb-3`}>For GPU-accelerated training, ensure you have CUDA installed and install PyTorch with CUDA support:</p>
            <CodeBlock language="bash" title="terminal" code={`# Install PyTorch with CUDA 12.1 support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Verify GPU availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"

# Install DTE with all dependencies
pip install -e ".[dev]"`} />
          </div>

          <div className={cardCls}>
            <h3 className={subheadCls}>Verify Installation</h3>
            <CodeBlock language="bash" title="terminal" code={`# Check system information and GPU status
python main.py info

# Initialize default configuration
python main.py init

# Validate the generated configuration
python main.py validate config.yaml`} />
            <Note type="tip" isDark={isDark}>
              If <code className={inlineCodeCls}>python main.py info</code> shows CUDA as available and lists your GPU, you are ready for training. For debate-only usage, no GPU is needed.
            </Note>
          </div>

          <div className={cardCls}>
            <h3 className={subheadCls}>Development Setup</h3>
            <p className={`${textCls} mb-3`}>For contributing or running the test suite:</p>
            <CodeBlock language="bash" title="terminal" code={`# Install development dependencies
pip install -e ".[dev]"

# Run unit tests (no GPU required, 124 tests)
pytest -m "not gpu" -v

# Run GPU integration tests
CUDA_VISIBLE_DEVICES=0 pytest tests/test_debate_integration.py -v
CUDA_VISIBLE_DEVICES=1 pytest tests/test_training_integration.py -v

# Lint and format
ruff check dte/ tests/
ruff format dte/ tests/

# Type checking
mypy dte/ --ignore-missing-imports`} />
          </div>
        </div>
      ),
    },

    /* ====== QUICK START ====== */
    {
      id: 'quickstart',
      title: 'Quick Start',
      icon: 'M13 10V3L4 14h7v7l9-11h-7z',
      content: (
        <div className="space-y-6">
          <div className={cardCls}>
            <h2 className={headCls}>Quick Start Guide</h2>
            <p className={textCls}>Get up and running with DTE in under 5 minutes. Choose your preferred interface -- Python API or CLI.</p>
          </div>

          <div className={cardCls}>
            <div className={labelCls}>Core Mechanism</div>
            <h3 className={subheadCls}>Reflect-Critique-Refine (RCR) Flow</h3>
            <p className={`${textCls} mb-4`}>Each debate round uses a three-step RCR prompting protocol to reduce sycophancy and improve reasoning quality.</p>
            <RCRFlowDiagram isDark={isDark} />
          </div>

          <div className={cardCls}>
            <h3 className={subheadCls}>1. One-Liner Debate (Python API)</h3>
            <p className={`${textCls} mb-3`}>The simplest way to run a multi-agent debate:</p>
            <CodeBlock language="python" title="examples/quick_start.py" code={`import dte

# Run a 3-agent debate on a math problem
result = dte.debate(
    query="What is 15 * 24?",
    model="Qwen/Qwen2.5-0.5B-Instruct",
    num_agents=3,
    max_rounds=3,
    task_type="math",
)

print(f"Final answer : {result.final_answer}")      # "360"
print(f"Consensus    : {result.consensus_reached}")  # True
print(f"Total rounds : {result.total_rounds}")
print(f"Time         : {result.metrics.get('total_time', 0):.2f}s")

# Show answer progression per round
for round_idx, answers in enumerate(result.extracted_answers):
    print(f"  Round {round_idx}: {answers}")`} />
          </div>

          <div className={cardCls}>
            <h3 className={subheadCls}>2. Component-Level API</h3>
            <p className={`${textCls} mb-3`}>For fine-grained control over model and debate configuration:</p>
            <CodeBlock language="python" title="examples/custom_debate.py" code={`from dte.core.config import ModelConfig, DebateConfig
from dte.debate.manager import DebateManager

# Configure the model
model_config = ModelConfig(
    base_model_name="Qwen/Qwen2.5-0.5B-Instruct",
    device="auto",
    max_length=1024,
    temperature=0.5,
    top_p=0.95,
    top_k=40,
)

# Configure the debate
debate_config = DebateConfig(
    num_agents=3,
    max_rounds=4,
)

# Create manager and run debates
manager = DebateManager(debate_config, model_config)

queries = [
    ("What is 123 * 456?", "math"),
    ("What is the square root of 144?", "math"),
]

for query, task_type in queries:
    result = manager.conduct_debate(query, task_type)
    print(f"Query: {query}")
    print(f"  Answer    : {result.final_answer}")
    print(f"  Consensus : {result.consensus_reached}")
    print(f"  Rounds    : {result.total_rounds}")
    print(f"  Sycophancy: {result.metrics.get('sycophancy_rate', 0):.2%}")

# View aggregate statistics
stats = manager.get_debate_statistics()
for key, value in stats.items():
    print(f"  {key}: {value}")

manager.cleanup()  # Release GPU memory`} />
          </div>

          <div className={cardCls}>
            <h3 className={subheadCls}>3. CLI: Single Query Debate</h3>
            <CodeBlock language="bash" title="terminal" code={`# Single query debate
python main.py debate --query "What is 15 * 24?" --agents 3 --rounds 3

# With verbose output showing answer progression
python main.py debate --query "Solve: 3x + 5 = 14" \\
  --agents 3 --rounds 2 --task-type math --verbose

# Save results to file
python main.py debate --query "What causes seasons?" \\
  --output debate_result.json`} />
          </div>

          <div className={cardCls}>
            <h3 className={subheadCls}>4. CLI: Dataset Evaluation</h3>
            <CodeBlock language="bash" title="terminal" code={`# Evaluate on GSM8K samples
python main.py debate --dataset gsm8k --samples 20 --verbose

# Evaluate on ARC-Challenge
python main.py debate --dataset arc_challenge --samples 10 --output results.json

# Use specific models for diverse-agent debate
python main.py debate --query "Solve: 3x + 5 = 14" \\
  --models "Qwen/Qwen2.5-1.5B-Instruct,meta-llama/Llama-3.2-3B-Instruct,microsoft/Phi-3.5-mini-instruct"`} />
            <Note type="info" isDark={isDark}>
              When using <code className={inlineCodeCls}>--models</code>, the number of models must match <code className={inlineCodeCls}>--agents</code>. Each model acts as a separate debate agent.
            </Note>
          </div>

          <div className={cardCls}>
            <h3 className={subheadCls}>5. Full Pipeline from Config</h3>
            <CodeBlock language="python" title="examples/full_pipeline.py" code={`import dte
from dte.core.config import DTEConfig

# Load and validate configuration
config = DTEConfig.from_yaml("config.yaml")
errors = config.validate()
if errors:
    for err in errors:
        print(f"Config error: {err}")

# Create and run the complete DTE pipeline
pipeline = dte.DTEPipeline(config)
results = pipeline.run_complete_pipeline()

print(f"Total time       : {results['total_time_hours']:.2f} hours")
print(f"Evolution rounds : {results['total_evolution_rounds']}")
print(f"Best performance : {results['best_performance']:.4f}")
print(f"Total improvement: {results['total_improvement']:.4f}")
print(f"Converged        : {results['convergence_achieved']}")`} />
            <div className="mt-4">
              <CodeBlock language="bash" title="terminal" code={`# Or via CLI
python main.py run --config config.yaml

# Resume from checkpoint
python main.py run --resume checkpoint.json --save-checkpoint new_checkpoint.json`} />
            </div>
          </div>
        </div>
      ),
    },

    /* ====== API REFERENCE ====== */
    {
      id: 'api',
      title: 'API Reference',
      icon: 'M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4',
      content: (
        <div className="space-y-6">
          <div className={cardCls}>
            <h2 className={headCls}>API Reference</h2>
            <p className={textCls}>Complete reference for all public functions and classes in the DTE framework.</p>
          </div>

          <div className={cardCls}>
            <div className={labelCls}>Top-Level API</div>
            <h3 className={subheadCls}>dte.debate()</h3>
            <p className={`${textCls} mb-3`}>Run a multi-agent debate on a single query with minimal configuration.</p>
            <CodeBlock language="python" title="dte/__init__.py" code={`def debate(
    query: str,
    model: str = "Qwen/Qwen2.5-0.5B-Instruct",
    num_agents: int = 3,
    max_rounds: int = 3,
    task_type: str = "math",
    temperature: float = 0.7,
    verbose: bool = False,
) -> DebateResult:
    """Run a multi-agent debate on a single query.

    Args:
        query: The question or problem to debate.
        model: HuggingFace model identifier.
        num_agents: Number of debate agents (default: 3).
        max_rounds: Maximum debate rounds (default: 3).
        task_type: "math", "arc", "reasoning", or "general".
        temperature: Sampling temperature (default: 0.7).
        verbose: Print debate progress.

    Returns:
        DebateResult with final_answer, consensus_reached,
        total_rounds, extracted_answers, sycophancy_history,
        all_responses, and metrics dict.
    """`} />
          </div>

          <div className={cardCls}>
            <div className={labelCls}>Top-Level API</div>
            <h3 className={subheadCls}>dte.from_config()</h3>
            <p className={`${textCls} mb-3`}>Create a full pipeline instance from a YAML configuration file.</p>
            <CodeBlock language="python" title="dte/__init__.py" code={`def from_config(config_path: str) -> Pipeline:
    """Create a Pipeline from a YAML configuration file.

    Args:
        config_path: Path to YAML configuration file.

    Returns:
        Pipeline instance ready to run.
    """

# Usage:
pipeline = dte.from_config("config.yaml")
results = pipeline.run_complete_pipeline()`} />
          </div>

          <div className={cardCls}>
            <div className={labelCls}>Core Classes</div>
            <h3 className={subheadCls}>DTEConfig</h3>
            <p className={`${textCls} mb-3`}>Top-level configuration that aggregates all sub-configs. Load from YAML and validate before use.</p>
            <CodeBlock language="python" title="dte/core/config.py" code={`from dte.core.config import DTEConfig

# Load from YAML
config = DTEConfig.from_yaml("config.yaml")

# Validate (returns list of error strings, empty = valid)
errors = config.validate()

# Access sub-configs
print(config.model.base_model_name)     # "Qwen/Qwen2.5-1.5B-Instruct"
print(config.debate.num_agents)         # 3
print(config.training.learning_rate)    # 5e-6
print(config.evolution.max_rounds)      # 3
print(config.experiment.name)           # "dte_pipeline_v1"

# Save to YAML
config.save_yaml("my_config.yaml")`} />
          </div>

          <div className={cardCls}>
            <div className={labelCls}>Core Classes</div>
            <h3 className={subheadCls}>ModelConfig</h3>
            <CodeBlock language="python" title="dte/core/config.py" code={`@dataclass
class ModelConfig:
    base_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    base_model_path: str | None = None     # local model path
    max_length: int = 2048                 # max token context
    temperature: float = 0.7               # sampling temperature
    top_p: float = 0.9                     # nucleus sampling
    top_k: int = 50                        # top-k sampling
    device: str = "auto"                   # "auto", "cpu", "cuda", "mps"`} />
          </div>

          <div className={cardCls}>
            <div className={labelCls}>Core Classes</div>
            <h3 className={subheadCls}>DebateConfig</h3>
            <CodeBlock language="python" title="dte/core/config.py" code={`@dataclass
class DebateConfig:
    num_agents: int = 3                    # debate agents (2-7)
    max_rounds: int = 3                    # maximum rounds
    consensus_threshold: float = 1.0       # 1.0 = full consensus
    rcr_prompting: bool = True             # enable RCR
    require_novel_reasoning: bool = True   # novel reasoning on answer change
    critique_pairs: int = 2                # peer critiques per agent
    use_diverse_agents: bool = False       # different models per agent
    agent_models: list[str] = field(default_factory=list)
    temperature_annealing: bool = True     # temp decay for small models
    start_temp: float = 0.7
    end_temp: float = 0.3
    min_model_size: str = "3B"             # anneal only below this`} />
          </div>

          <div className={cardCls}>
            <div className={labelCls}>Debate</div>
            <h3 className={subheadCls}>DebateManager</h3>
            <p className={`${textCls} mb-3`}>Orchestrates multi-agent debates with RCR prompting, consensus detection, and sycophancy tracking.</p>
            <CodeBlock language="python" title="dte/debate/manager.py" code={`from dte.debate.manager import DebateManager

manager = DebateManager(debate_cfg, model_cfg)

# Conduct a single debate
result = manager.conduct_debate(query, task_type="math")

# Access results
result.final_answer         # str: the consensus or majority answer
result.consensus_reached    # bool
result.total_rounds         # int
result.extracted_answers    # list[list[str]]: per-round per-agent
result.sycophancy_history   # list[dict]: per-round sycophancy flags
result.all_responses        # list[list]: full reasoning traces
result.metrics              # dict: timing, consensus_rate, etc.

# Aggregate stats across multiple debates
stats = manager.get_debate_statistics()
manager.cleanup()  # free GPU memory`} />
          </div>

          <div className={cardCls}>
            <div className={labelCls}>Training</div>
            <h3 className={subheadCls}>GRPOTrainer</h3>
            <p className={`${textCls} mb-3`}>GRPO training with LoRA, shaped rewards, and KL regularization.</p>
            <CodeBlock language="python" title="dte/training/grpo_trainer.py" code={`from dte.training.grpo_trainer import GRPOTrainer

trainer = GRPOTrainer(
    training_config=config.training,
    model_config=config.model,
    paths_config=config.paths,
    logger=logger,
)

# Train on debate-generated data
metrics = trainer.train(training_examples)
print(f"Final loss: {metrics['epoch_losses'][-1]:.4f}")

# The model is saved automatically to config.paths.models_dir`} />
          </div>

          <div className={cardCls}>
            <div className={labelCls}>Training</div>
            <h3 className={subheadCls}>DTERewardModel</h3>
            <p className={`${textCls} mb-3`}>Computes all 5 shaped reward functions used in GRPO training.</p>
            <CodeBlock language="python" title="examples/reward_functions.py" code={`from dte.training.reward_model import DTERewardModel

model = DTERewardModel()

# Calculate all 5 rewards for a batch of responses
rewards_dict = model.calculate_all_rewards(
    query="What is 6 * 7?",
    responses=[
        "<reasoning>\\n6 * 7 = 42\\n</reasoning>\\n<answer>\\n42\\n</answer>\\n",
        "The answer is 42.",
    ],
    ground_truth="42",
)

# Combine with DTE weights
weights = {
    "correctness": 2.0,
    "int": 0.5,
    "strict_format": 0.5,
    "soft_format": 0.5,
    "xmlcount": 0.5,
}
combined = model.combine_rewards(rewards_dict, weights)
# Maximum possible: 4.0 per response`} />
          </div>

          <div className={cardCls}>
            <div className={labelCls}>Evaluation</div>
            <h3 className={subheadCls}>DTEEvaluator</h3>
            <CodeBlock language="python" title="examples/evaluation_example.py" code={`from dte.core.evaluator import DTEEvaluator

evaluator = DTEEvaluator(datasets_config, debate_config, model_config, logger)
metrics = evaluator.evaluate_model(evolution_round=0, max_samples_per_dataset=100)

print(f"Overall accuracy    : {metrics.overall_accuracy:.2%}")
print(f"Consensus rate      : {metrics.consensus_rate:.2%}")
print(f"Debate helped rate  : {metrics.debate_helped_rate:.2%}")
print(f"Sycophancy rate     : {metrics.sycophancy_rate:.2%}")
print(f"Avg debate rounds   : {metrics.average_debate_rounds:.1f}")

# Per-dataset breakdown
for ds_name, ds_metrics in metrics.per_dataset_metrics.items():
    print(f"  {ds_name}: {ds_metrics}")

evaluator.cleanup()`} />
          </div>

          <div className={cardCls}>
            <div className={labelCls}>Project Structure</div>
            <h3 className={subheadCls}>Architecture</h3>
            <div className="mb-6">
              <ArchitectureDiagram isDark={isDark} />
            </div>
            <CodeBlock language="bash" title="Project Layout" code={`Debate-Train-Evolve/
├── dte/                          # Main package
│   ├── __init__.py              # Public API (dte.debate(), dte.from_config())
│   ├── cli.py                   # CLI entry point
│   ├── core/                    # Core components
│   │   ├── config.py            # Configuration dataclasses (DTEConfig)
│   │   ├── evaluator.py         # Model evaluation (DTEEvaluator)
│   │   ├── logger.py            # Rich-based logging (DTELogger)
│   │   └── pipeline.py          # Pipeline orchestrator (DTEPipeline)
│   ├── debate/                  # Multi-agent debate
│   │   ├── agent.py             # Individual debate agents
│   │   ├── manager.py           # Debate orchestration & consensus
│   │   └── prompts.py           # RCR prompting system
│   ├── training/                # GRPO training
│   │   ├── grpo_trainer.py      # GRPO implementation (GRPOTrainer)
│   │   └── reward_model.py      # 5 reward functions (DTERewardModel)
│   ├── data/                    # Data processing
│   │   ├── generator.py         # Debate data generation
│   │   ├── processor.py         # Data preprocessing
│   │   └── dataset_manager.py   # Dataset loading & management
│   └── utils/                   # Utilities
│       ├── answer_extraction.py # Answer parsing & consensus
│       ├── helpers.py           # General utilities
│       └── data_utils.py        # Data I/O utilities
├── tests/                       # 124 unit + 11 GPU tests
├── examples/                    # 6 usage examples
│   ├── quick_start.py           # One-liner debate
│   ├── custom_debate.py         # Custom configuration
│   ├── full_pipeline.py         # End-to-end pipeline
│   ├── evaluation_example.py    # Benchmark evaluation
│   ├── reward_functions.py      # Reward function demo
│   └── multi_gpu_training.py    # Multi-GPU setup
├── config.yaml                  # Default configuration
├── main.py                      # CLI interface
└── pyproject.toml               # Package metadata`} />
          </div>
        </div>
      ),
    },

    /* ====== CONFIGURATION ====== */
    {
      id: 'config',
      title: 'Configuration',
      icon: 'M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.066 2.573c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.573 1.066c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.066-2.573c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z M15 12a3 3 0 11-6 0 3 3 0 016 0z',
      content: (
        <div className="space-y-6">
          <div className={cardCls}>
            <h2 className={headCls}>Configuration Reference</h2>
            <p className={textCls}>DTE uses YAML configuration files. Every parameter is documented below. Generate a default config with <code className={inlineCodeCls}>python main.py init</code>.</p>
          </div>

          <Collapsible title="Model Configuration" defaultOpen isDark={isDark}>
            <CodeBlock language="yaml" title="config.yaml -- model" code={`model:
  base_model_name: "Qwen/Qwen2.5-1.5B-Instruct"  # HuggingFace model ID
  base_model_path: null          # Optional: path to local model weights
  max_length: 2048               # Maximum token context length
  temperature: 0.7               # Sampling temperature (0.0 = greedy)
  top_p: 0.9                     # Nucleus sampling threshold
  top_k: 50                      # Top-k sampling`} />
            <div className="mt-3">
              <Note type="tip" isDark={isDark}>
                Set <code className={inlineCodeCls}>base_model_path</code> to a local directory to skip downloading from HuggingFace. This is useful for gated models or offline environments.
              </Note>
            </div>
          </Collapsible>

          <Collapsible title="Debate Configuration" defaultOpen isDark={isDark}>
            <CodeBlock language="yaml" title="config.yaml -- debate" code={`debate:
  num_agents: 3                  # Number of agents in debate (2-7)
  max_rounds: 3                  # Maximum debate rounds
  consensus_threshold: 1.0       # 1.0 = all agents must agree

  rcr_prompting:
    enabled: true                # Enable Reflect-Critique-Refine
    require_novel_reasoning: true # Must provide new reasoning when switching
    critique_pairs: 2            # Number of peer critiques per agent

  use_diverse_agents: false      # Use different models for agents
  agent_models: []               # List of model IDs if diverse

  temperature_annealing:
    enabled: true                # Anneal temperature across rounds
    start_temp: 0.7              # Starting temperature
    end_temp: 0.3                # Ending temperature
    min_model_size: "3B"         # Only anneal for models < this size`} />
            <div className="mt-3">
              <Note type="info" isDark={isDark}>
                Temperature annealing is critical for models under 3B parameters. Without it, small models suffer catastrophic forgetting after round 2. Lowering temperature from 0.7 to 0.3 reduces KL drift by ~33%.
              </Note>
            </div>
          </Collapsible>

          <Collapsible title="GRPO Training Configuration" isDark={isDark}>
            <CodeBlock language="yaml" title="config.yaml -- training" code={`training:
  learning_rate: 5e-6            # AdamW learning rate
  weight_decay: 0.1              # L2 regularization
  warmup_steps: 50               # Linear warmup steps
  max_epochs: 3                  # Training epochs per round
  batch_size: 8                  # Batch size
  gradient_accumulation_steps: 4 # Effective batch = batch * grad_accum

  grpo:
    group_size: 8                # Responses per query for comparison
    advantage_normalization: true # Normalize advantages
    clip_ratio: 0.2              # PPO clipping threshold
    kl_penalty: 0.02             # KL regularization coefficient

  rewards:
    correctness_weight: 2.0      # Answer correctness (main signal)
    int_weight: 0.5              # Numeric format reward
    strict_format_weight: 0.5    # Exact XML format compliance
    soft_format_weight: 0.5      # Flexible XML format check
    xmlcount_weight: 0.5         # Granular XML tag counting

  lora:
    enabled: true                # Use LoRA / QLoRA
    rank: 128                    # LoRA rank (higher = more params)
    alpha: 256                   # LoRA alpha (usually 2x rank)
    dropout: 0.05                # LoRA dropout
    target_modules:              # Which layers to adapt
      - "q_proj"
      - "k_proj"
      - "v_proj"
      - "o_proj"
      - "gate_proj"
      - "up_proj"
      - "down_proj"`} />
          </Collapsible>

          <Collapsible title="Evolution Configuration" isDark={isDark}>
            <CodeBlock language="yaml" title="config.yaml -- evolution" code={`evolution:
  max_rounds: 3                  # Maximum evolution rounds
  convergence_threshold: 0.01    # Stop if improvement < 1%
  patience: 2                    # Stop after N rounds without improvement
  samples_per_round: 500         # Debate samples per round
  validation_split: 0.2          # Fraction for validation
  validation_freq: 1             # Validate every N epochs`} />
          </Collapsible>

          <Collapsible title="Dataset Configuration" isDark={isDark}>
            <CodeBlock language="yaml" title="config.yaml -- datasets" code={`datasets:
  train_datasets:
    - name: "gsm8k"
      path: "openai/gsm8k"      # HuggingFace dataset or local path
      split: "train"
      max_samples: 1000

    - name: "gsm_plus"
      path: "qintongli/GSM-Plus"
      split: "train"
      max_samples: 500

  eval_datasets:
    - name: "gsm8k_test"
      path: "openai/gsm8k"
      split: "test"
      max_samples: 500

    - name: "math"
      path: "hendrycks/competition_math"
      split: "test"
      max_samples: 200

    - name: "arc_challenge"
      path: "allenai/ai2_arc"
      split: "ARC-Challenge"
      max_samples: 200`} />
          </Collapsible>

          <Collapsible title="Hardware Configuration" isDark={isDark}>
            <CodeBlock language="yaml" title="config.yaml -- hardware" code={`hardware:
  device: "auto"                 # "auto", "cpu", "cuda", "mps"
  mixed_precision: true          # FP16/BF16 training
  max_memory_per_gpu: "20GB"     # GPU memory limit
  gradient_checkpointing: true   # Trade compute for memory
  num_workers: 4                 # DataLoader workers
  dataloader_pin_memory: true    # Pin memory for GPU transfer`} />
          </Collapsible>

          <Collapsible title="Logging, Paths, & Experiment Tracking" isDark={isDark}>
            <CodeBlock language="yaml" title="config.yaml -- logging, paths, experiment" code={`logging:
  level: "INFO"                  # DEBUG, INFO, WARNING, ERROR
  log_dir: "./logs"
  experiment_name: "dte_experiment"
  save_checkpoints: true
  checkpoint_freq: 100           # Save every N steps
  track_metrics:
    - "accuracy"
    - "debate_consensus_rate"
    - "sycophancy_rate"
    - "training_loss"
    - "kl_divergence"

paths:
  output_dir: "./outputs"
  models_dir: "./models"
  data_dir: "./data"
  cache_dir: "./cache"
  temp_dir: "./tmp"

experiment:
  name: "dte_pipeline_v1"
  seed: 42
  deterministic: true
  wandb:
    enabled: false
    project: "dte-framework"
    entity: null

safety:
  filter_toxic_content: true
  max_reasoning_length: 1000
  validate_model_outputs: true
  check_format_compliance: true
  auto_backup: true
  backup_frequency: "1h"`} />
          </Collapsible>

          <div className={cardCls}>
            <h3 className={subheadCls}>Minimal Configuration</h3>
            <p className={`${textCls} mb-3`}>The smallest config that works -- all other values use defaults:</p>
            <CodeBlock language="yaml" title="minimal_config.yaml" code={`model:
  base_model_name: "Qwen/Qwen2.5-1.5B-Instruct"
  temperature: 0.7

debate:
  num_agents: 3
  max_rounds: 3
  rcr_prompting:
    enabled: true

training:
  learning_rate: 5e-6
  max_epochs: 3
  lora:
    enabled: true
    rank: 128

evolution:
  max_rounds: 3
  samples_per_round: 500`} />
          </div>
        </div>
      ),
    },

    /* ====== TRAINING GUIDE ====== */
    {
      id: 'training',
      title: 'Training Guide',
      icon: 'M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z',
      content: (
        <div className="space-y-6">
          <div className={cardCls}>
            <h2 className={headCls}>Training Guide</h2>
            <p className={textCls}>Step-by-step guide to train models with the DTE framework, from configuration to multi-GPU deployment.</p>
          </div>

          <div className={cardCls}>
            <div className={labelCls}>GRPO Overview</div>
            <h3 className={subheadCls}>Training Loop</h3>
            <p className={`${textCls} mb-4`}>DTE uses Group Relative Policy Optimization (GRPO) to update model weights based on debate trace quality.</p>
            <GRPOTrainingDiagram isDark={isDark} />
          </div>

          <div className={cardCls}>
            <div className={labelCls}>Evolution Pipeline</div>
            <h3 className={subheadCls}>Multi-Round Evolution</h3>
            <p className={`${textCls} mb-4`}>The model iteratively evolves through repeated debate-train cycles, with convergence checks between rounds.</p>
            <EvolutionLoopDiagram isDark={isDark} />
          </div>

          <div className={cardCls}>
            <h3 className={subheadCls}>Step 1: Initialize Configuration</h3>
            <CodeBlock language="bash" title="terminal" code={`# Create a default configuration file
python main.py init

# Or create with custom name
python main.py init --output my_config.yaml --force

# Validate it
python main.py validate config.yaml`} />
          </div>

          <div className={cardCls}>
            <h3 className={subheadCls}>Step 2: Generate Debate Data</h3>
            <p className={`${textCls} mb-3`}>Generate training data from multi-agent debates. This runs N-agent RCR debates on your training dataset and produces JSONL traces.</p>
            <CodeBlock language="bash" title="terminal" code={`# Generate 500 debate traces for round 1
python main.py generate --samples 500 --output debate_data.jsonl --round 1

# With custom config and more samples
python main.py generate \\
  --config custom_config.yaml \\
  --samples 1000 \\
  --output data/round_1.jsonl`} />
            <Note type="info" isDark={isDark}>
              Each evolution round typically uses 500+ debate traces (~2M tokens for a 7B model). The <code className={inlineCodeCls}>--round</code> flag affects temperature annealing for small models.
            </Note>
          </div>

          <div className={cardCls}>
            <h3 className={subheadCls}>Step 3: Train with GRPO</h3>
            <CodeBlock language="bash" title="terminal" code={`# Train model using generated debate data
python main.py train \\
  --data debate_data.jsonl \\
  --epochs 3 \\
  --batch-size 8 \\
  --learning-rate 5e-6

# Specify output directory
python main.py train \\
  --data debate_data.jsonl \\
  --output-dir ./models/round_1`} />
          </div>

          <div className={cardCls}>
            <h3 className={subheadCls}>Step 4: Run Complete Pipeline</h3>
            <p className={`${textCls} mb-3`}>The full pipeline automates Steps 2-3 in a loop, with validation and early stopping:</p>
            <CodeBlock language="bash" title="terminal" code={`# Run the full Debate-Train-Evolve pipeline
python main.py run --config config.yaml

# Resume from checkpoint
python main.py run --resume checkpoint.json

# Save checkpoints after each round
python main.py run --save-checkpoint checkpoint.json`} />
          </div>

          <div className={cardCls}>
            <h3 className={subheadCls}>GRPO Hyperparameters</h3>
            <p className={`${textCls} mb-4`}>These are the exact hyperparameters used in the paper experiments:</p>
            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-3">
              {[
                { p: 'Learning Rate', v: '5e-6' }, { p: 'Weight Decay', v: '0.1' },
                { p: 'Batch Size', v: '8' }, { p: 'Group Size', v: '8' },
                { p: 'Clip Ratio', v: '0.2' }, { p: 'KL Coefficient', v: '0.02' },
                { p: 'LoRA Rank', v: '128' }, { p: 'LoRA Dropout', v: '0.05' },
                { p: 'Warmup', v: '10% cosine' }, { p: 'Steps', v: 'up to 10k' },
                { p: 'Optimizer', v: 'AdamW 8-bit' }, { p: 'Precision', v: 'BF16 / FP16' },
              ].map((h) => (
                <div key={h.p} className={`p-3 rounded-xl ${isDark ? 'bg-white/3' : 'bg-navy-50/50'}`}>
                  <div className={`text-xs mb-1 ${isDark ? 'text-gray-500' : 'text-gray-400'}`}>{h.p}</div>
                  <div className={`text-sm font-mono font-medium ${isDark ? 'text-white' : 'text-navy-900'}`}>{h.v}</div>
                </div>
              ))}
            </div>
          </div>

          <div className={cardCls}>
            <h3 className={subheadCls}>Multi-GPU Setup</h3>
            <p className={`${textCls} mb-3`}>DTE supports multi-GPU training via HuggingFace Accelerate:</p>
            <CodeBlock language="bash" title="terminal" code={`# Launch with accelerate for multi-GPU
accelerate launch --num_processes 2 main.py run --config config.yaml

# Set specific GPUs
CUDA_VISIBLE_DEVICES=0,1 python main.py run --config config.yaml

# Configure accelerate interactively
accelerate config`} />
            <div className="mt-3">
              <Note type="info" isDark={isDark}>
                GPU memory requirements: ~16GB for 1.5B models, ~24GB for 3B, ~48GB for 7B, ~80GB for 14B. Use gradient checkpointing and LoRA to reduce memory. Our paper experiments used A100 (80GB), H100, L40 (48GB), and A40 (48GB) GPUs.
              </Note>
            </div>
          </div>

          <div className={cardCls}>
            <h3 className={subheadCls}>Expected Training Time</h3>
            <div className="overflow-x-auto">
              <table className={`w-full text-sm ${isDark ? 'text-gray-300' : 'text-gray-700'}`}>
                <thead>
                  <tr className={`border-b ${isDark ? 'border-white/10' : 'border-navy-200/30'}`}>
                    <th className={`text-left py-2 pr-4 text-xs font-mono uppercase ${isDark ? 'text-gray-500' : 'text-gray-400'}`}>Model</th>
                    <th className={`text-left py-2 pr-4 text-xs font-mono uppercase ${isDark ? 'text-gray-500' : 'text-gray-400'}`}>GPU</th>
                    <th className={`text-left py-2 text-xs font-mono uppercase ${isDark ? 'text-gray-500' : 'text-gray-400'}`}>~Time / Round</th>
                  </tr>
                </thead>
                <tbody>
                  {[
                    { model: 'Qwen-2.5-1.5B', gpu: 'A100 80GB', time: '~12 hours' },
                    { model: 'Qwen-2.5-3B', gpu: 'A100 80GB', time: '~24 hours' },
                    { model: 'Qwen-2.5-7B', gpu: 'A100 80GB', time: '~68 hours' },
                    { model: 'Qwen-2.5-14B', gpu: 'A100 80GB', time: '~120 hours' },
                    { model: 'Llama-3.1-8B', gpu: 'A100 80GB', time: '~72 hours' },
                  ].map((r) => (
                    <tr key={r.model} className={`border-b ${isDark ? 'border-white/5' : 'border-navy-100/30'}`}>
                      <td className="py-2 pr-4 font-mono text-xs">{r.model}</td>
                      <td className="py-2 pr-4">{r.gpu}</td>
                      <td className={`py-2 font-mono ${isDark ? 'text-electric-300' : 'text-navy-600'}`}>{r.time}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      ),
    },

    /* ====== REWARD FUNCTIONS ====== */
    {
      id: 'rewards',
      title: 'Reward Functions',
      icon: 'M11.049 2.927c.3-.921 1.603-.921 1.902 0l1.519 4.674a1 1 0 00.95.69h4.915c.969 0 1.371 1.24.588 1.81l-3.976 2.888a1 1 0 00-.363 1.118l1.518 4.674c.3.922-.755 1.688-1.538 1.118l-3.976-2.888a1 1 0 00-1.176 0l-3.976 2.888c-.783.57-1.838-.197-1.538-1.118l1.518-4.674a1 1 0 00-.363-1.118l-3.976-2.888c-.784-.57-.38-1.81.588-1.81h4.914a1 1 0 00.951-.69l1.519-4.674z',
      content: (
        <div className="space-y-6">
          <div className={cardCls}>
            <h2 className={headCls}>Reward Functions</h2>
            <p className={textCls}>DTE uses 5 shaped reward functions for GRPO training, with a maximum total of 4.0 per response. Models are expected to output structured responses using <code className={inlineCodeCls}>{'<reasoning>'}</code> and <code className={inlineCodeCls}>{'<answer>'}</code> XML tags.</p>
          </div>

          <div className={cardCls}>
            <div className={labelCls}>Visual Overview</div>
            <h3 className={subheadCls}>Reward Weight Distribution</h3>
            <RewardStackDiagram isDark={isDark} />
          </div>

          {[
            { name: 'Correctness Reward', w: '2.0', desc: 'Checks if the extracted answer matches the consensus answer from multi-agent debate via exact string match after normalization. This is the primary training signal with the highest weight.', formula: 'r_correct = 2.0 * 1[extract(response) == consensus_answer]', color: isDark ? 'text-green-400' : 'text-green-600' },
            { name: 'Integer/Numeric Format', w: '0.5', desc: 'Verifies the answer is a properly formatted number. For math problems, encourages clean numeric outputs rather than text descriptions.', formula: 'r_int = 0.5 * 1[is_numeric(extract(response))]', color: isDark ? 'text-electric-300' : 'text-navy-600' },
            { name: 'Strict XML Format', w: '0.5', desc: 'Exact XML compliance: response must have properly nested <reasoning>...</reasoning> and <answer>...</answer> tags with no extraneous content outside tags.', formula: 'r_strict = 0.5 * 1[regex_match(response, strict_pattern)]', color: isDark ? 'text-cyan-300' : 'text-teal-600' },
            { name: 'Soft XML Format', w: '0.5', desc: 'Lenient check: awards credit if required XML tags are present, even if not perfectly structured. Helps early training stages.', formula: 'r_soft = 0.5 * 1[contains("<reasoning>") and contains("<answer>")]', color: isDark ? 'text-purple-400' : 'text-purple-600' },
            { name: 'XML Count Reward', w: '0.5', desc: 'Graduated scoring based on correct XML tag pairs found. Provides partial credit rather than all-or-nothing.', formula: 'r_count = 0.5 * (num_correct_tag_pairs / num_expected_tag_pairs)', color: isDark ? 'text-amber-400' : 'text-amber-600' },
          ].map((r, i) => (
            <div key={r.name} className={cardCls}>
              <div className="flex items-start justify-between mb-3 flex-wrap gap-2">
                <div>
                  <div className={labelCls}>Reward {i + 1}</div>
                  <h3 className={`text-lg font-display font-bold ${isDark ? 'text-white' : 'text-navy-900'}`}>{r.name}</h3>
                </div>
                <span className={`font-mono text-sm font-bold px-3 py-1 rounded-lg ${isDark ? 'bg-white/5' : 'bg-navy-50'} ${r.color}`}>weight = {r.w}</span>
              </div>
              <p className={`${textCls} mb-3`}>{r.desc}</p>
              <CodeBlock language="python" code={r.formula} />
            </div>
          ))}

          <div className={`${cardCls} border-2 ${isDark ? 'border-electric-500/20' : 'border-navy-300/30'}`}>
            <h3 className={subheadCls}>Total Reward Formula</h3>
            <CodeBlock language="python" code={`total_reward = (
    2.0 * correctness_reward     # max 2.0
    + 0.5 * integer_format       # max 0.5
    + 0.5 * strict_xml           # max 0.5
    + 0.5 * soft_xml             # max 0.5
    + 0.5 * xml_count            # max 0.5
)
# Maximum possible: 4.0 per response`} />
          </div>

          <div className={cardCls}>
            <h3 className={subheadCls}>Using Rewards Programmatically</h3>
            <CodeBlock language="python" title="examples/reward_functions.py" code={`from dte.training.reward_model import DTERewardModel

model = DTERewardModel()

responses = [
    # Perfect: correct answer + strict XML format
    "<reasoning>\\n6 * 7 = 42\\n</reasoning>\\n<answer>\\n42\\n</answer>\\n",
    # Correct but sloppy format
    "<reasoning>6 * 7 = 42</reasoning><answer>42</answer>",
    # Incorrect answer but good format
    "<reasoning>\\n6 * 7 = 48\\n</reasoning>\\n<answer>\\n48\\n</answer>\\n",
    # No XML format at all
    "The answer is 42.",
]

rewards = model.calculate_all_rewards(
    query="What is 6 * 7?",
    responses=responses,
    ground_truth="42",
)

# Combine with DTE weights
weights = {"correctness": 2.0, "int": 0.5, "strict_format": 0.5,
           "soft_format": 0.5, "xmlcount": 0.5}
combined = model.combine_rewards(rewards, weights)
# [4.0, 3.5, 2.0, 2.0]  (approx)`} />
          </div>
        </div>
      ),
    },

    /* ====== DATASETS ====== */
    {
      id: 'datasets',
      title: 'Datasets',
      icon: 'M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4m0 5c0 2.21-3.582 4-8 4s-8-1.79-8-4',
      content: (
        <div className="space-y-6">
          <div className={cardCls}>
            <h2 className={headCls}>Dataset Reference</h2>
            <p className={textCls}>DTE is evaluated on 7 diverse reasoning benchmarks spanning math, science, STEM, and commonsense reasoning.</p>
          </div>

          {[
            { name: 'GSM8K', path: 'openai/gsm8k', train: '7,473', test: '1,319', type: 'Math', desc: 'Grade school math word problems requiring 2-8 reasoning steps. Primary training benchmark.' },
            { name: 'GSM-Plus', path: 'qintongli/GSM-Plus', train: '--', test: '2,400', type: 'Math', desc: 'Adversarial variations of GSM8K with more complex values and extra reasoning steps. DTE shows +8.92% average gain here.' },
            { name: 'MATH', path: 'hendrycks/competition_math', train: '7,500', test: '5,000', type: 'Math', desc: 'Competition-level mathematics from AMC, AIME. Levels 1-5, from algebra to number theory.' },
            { name: 'ARC-Easy', path: 'allenai/ai2_arc', train: '2,251', test: '2,376', type: 'Science', desc: 'Elementary-level science questions answerable by middle school students.' },
            { name: 'ARC-Challenge', path: 'allenai/ai2_arc', train: '1,119', test: '1,172', type: 'Science', desc: 'Science questions that challenge retrieval-based methods, requiring deeper understanding. DTE shows +8.88% for Llama-8B.' },
            { name: 'GPQA Main', path: 'Idavidrein/gpqa', train: '--', test: '448', type: 'STEM', desc: 'Graduate-level STEM (biology, physics, chemistry). "Google-proof": non-experts achieve ~34% accuracy.' },
            { name: 'CommonsenseQA', path: 'tau/commonsense_qa', train: '9,741', test: '1,140', type: 'Commonsense', desc: 'Questions requiring commonsense knowledge beyond factual recall. Tests cross-domain generalization.' },
          ].map((d) => (
            <div key={d.name} className={cardCls}>
              <div className="flex items-start justify-between mb-2 flex-wrap gap-2">
                <h3 className={subheadCls}>{d.name}</h3>
                <span className={`text-xs font-mono px-2 py-0.5 rounded ${
                  d.type === 'Math' ? isDark ? 'bg-electric-500/10 text-electric-300' : 'bg-blue-50 text-blue-600'
                  : d.type === 'Science' ? isDark ? 'bg-cyan-500/10 text-cyan-300' : 'bg-teal-50 text-teal-600'
                  : d.type === 'STEM' ? isDark ? 'bg-purple-500/10 text-purple-300' : 'bg-purple-50 text-purple-600'
                  : isDark ? 'bg-amber-500/10 text-amber-300' : 'bg-amber-50 text-amber-600'
                }`}>{d.type}</span>
              </div>
              <p className={`${textCls} mb-3`}>{d.desc}</p>
              <div className="flex flex-wrap gap-4 text-xs font-mono">
                <span className={isDark ? 'text-gray-500' : 'text-gray-400'}>Train: {d.train}</span>
                <span className={isDark ? 'text-gray-500' : 'text-gray-400'}>Test: {d.test}</span>
                <span className={isDark ? 'text-gray-500' : 'text-gray-400'}>HF: {d.path}</span>
              </div>
            </div>
          ))}

          <div className={cardCls}>
            <h3 className={subheadCls}>Loading Datasets in Code</h3>
            <CodeBlock language="python" title="dte/data/dataset_manager.py" code={`from dte.data.dataset_manager import DatasetManager

dm = DatasetManager()

# Load by name (auto-resolves HuggingFace path)
dataset = dm.load_dataset_by_name("gsm8k", split="test", max_samples=100)
processed = dm.preprocess_dataset(dataset, "gsm8k")

# Each sample has:
# sample["formatted_query"]  - prompt ready for the model
# sample["ground_truth"]     - expected answer for evaluation`} />
          </div>
        </div>
      ),
    },

    /* ====== CLI REFERENCE ====== */
    {
      id: 'cli',
      title: 'CLI Reference',
      icon: 'M8 9l3 3-3 3m5 0h3M5 20h14a2 2 0 002-2V6a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z',
      content: (
        <div className="space-y-6">
          <div className={cardCls}>
            <h2 className={headCls}>CLI Reference</h2>
            <p className={textCls}>All commands available through the DTE command-line interface. The CLI is built with Click and supports global options <code className={inlineCodeCls}>--config</code> and <code className={inlineCodeCls}>--verbose</code>.</p>
          </div>

          <div className={cardCls}>
            <h3 className={subheadCls}>Global Options</h3>
            <CodeBlock language="bash" title="terminal" code={`# All commands accept these global options
python main.py --config custom.yaml <command>  # Use custom config
python main.py --verbose <command>              # Enable debug logging
python main.py --version                        # Show version
python main.py --help                           # Show help`} />
          </div>

          {[
            { cmd: 'run', desc: 'Run the complete DTE pipeline (debate -> train -> evolve loop).', flags: '--resume PATH, --save-checkpoint PATH', ex: 'python main.py run --config config.yaml\npython main.py run --resume checkpoint.json\npython main.py run --save-checkpoint checkpoint.json' },
            { cmd: 'debate', desc: 'Run standalone multi-agent debate on a query or dataset.', flags: '-q/--query, --dataset, -s/--samples, -a/--agents, -r/--rounds, --task-type, -o/--output, -v/--verbose, --models', ex: 'python main.py debate --query "What is 15 * 24?" --agents 3 --rounds 3\npython main.py debate --dataset gsm8k --samples 20 --verbose\npython main.py debate --query "Solve: 3x+5=14" \\n  --models "Qwen/Qwen2.5-1.5B-Instruct,meta-llama/Llama-3.2-3B-Instruct,microsoft/Phi-3.5-mini-instruct" \\n  --output result.json' },
            { cmd: 'generate', desc: 'Generate training data from multi-agent debates.', flags: '-n/--samples, -o/--output (required), -r/--round', ex: 'python main.py generate --samples 500 --output debate_data.jsonl --round 1\npython main.py generate --config custom.yaml --samples 1000 --output data/r1.jsonl' },
            { cmd: 'train', desc: 'Train model with GRPO on debate data.', flags: '-d/--data (required), -e/--epochs, -b/--batch-size, -lr/--learning-rate, -o/--output-dir', ex: 'python main.py train --data debate_data.jsonl --epochs 3 --batch-size 8\npython main.py train --data data.jsonl --output-dir ./models/round_1 --learning-rate 5e-6' },
            { cmd: 'validate', desc: 'Validate a configuration file and report any errors.', flags: 'CONFIG_FILE (positional argument)', ex: 'python main.py validate config.yaml' },
            { cmd: 'init', desc: 'Generate a default configuration file.', flags: '-o/--output, -f/--force', ex: 'python main.py init\npython main.py init --output my_config.yaml --force' },
            { cmd: 'info', desc: 'Show configuration, GPU, and system information.', flags: '(none)', ex: 'python main.py info' },
          ].map((c) => (
            <div key={c.cmd} className={cardCls}>
              <h3 className={subheadCls}>
                <code className={`font-mono ${isDark ? 'text-electric-300' : 'text-navy-600'}`}>main.py {c.cmd}</code>
              </h3>
              <p className={`${textCls} mb-2`}>{c.desc}</p>
              <p className={`text-xs font-mono mb-3 ${isDark ? 'text-gray-500' : 'text-gray-400'}`}>Flags: {c.flags}</p>
              <CodeBlock language="bash" title="terminal" code={c.ex} />
            </div>
          ))}

          <div className={cardCls}>
            <h3 className={subheadCls}>Available Dataset Choices for debate</h3>
            <p className={`${textCls} mb-2`}>The <code className={inlineCodeCls}>--dataset</code> flag accepts these values:</p>
            <div className="flex flex-wrap gap-2">
              {['gsm8k', 'gsm_plus', 'math', 'arc_challenge', 'arc_easy'].map(d => (
                <span key={d} className={`text-xs font-mono px-3 py-1 rounded-lg ${isDark ? 'bg-white/5 text-electric-300' : 'bg-navy-50 text-navy-600'}`}>{d}</span>
              ))}
            </div>
          </div>

          <div className={cardCls}>
            <h3 className={subheadCls}>Task Type Choices for debate</h3>
            <p className={`${textCls} mb-2`}>The <code className={inlineCodeCls}>--task-type</code> flag accepts:</p>
            <div className="flex flex-wrap gap-2">
              {['math', 'arc', 'reasoning', 'general', 'auto'].map(d => (
                <span key={d} className={`text-xs font-mono px-3 py-1 rounded-lg ${isDark ? 'bg-white/5 text-electric-300' : 'bg-navy-50 text-navy-600'}`}>{d}</span>
              ))}
            </div>
            <p className={`text-xs mt-2 ${isDark ? 'text-gray-500' : 'text-gray-400'}`}>When set to <code className={inlineCodeCls}>auto</code>, the task type is detected from the query content or dataset name.</p>
          </div>
        </div>
      ),
    },

    /* ====== TROUBLESHOOTING ====== */
    {
      id: 'troubleshooting',
      title: 'Troubleshooting',
      icon: 'M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z',
      content: (
        <div className="space-y-6">
          <div className={cardCls}>
            <h2 className={headCls}>Troubleshooting</h2>
            <p className={textCls}>Common issues and their solutions.</p>
          </div>

          <Collapsible title="GPU Out of Memory (OOM)" defaultOpen isDark={isDark}>
            <div className="space-y-3">
              <p className={textCls}>Reduce memory usage with these config changes:</p>
              <CodeBlock language="yaml" title="config.yaml" code={`training:
  batch_size: 2              # Reduce batch size
  gradient_accumulation_steps: 8  # Keep effective batch size
  lora:
    enabled: true            # Use LoRA (mandatory for large models)
    rank: 64                 # Lower rank = less memory
hardware:
  gradient_checkpointing: true
  mixed_precision: true
  max_memory_per_gpu: "20GB"
model:
  max_length: 1024           # Reduce context if possible`} />
              <Note type="tip" isDark={isDark}>
                For 7B+ models on 24GB GPUs, use LoRA rank 64, batch size 2, and gradient checkpointing. For 14B+ models, you need 80GB GPUs (A100/H100).
              </Note>
            </div>
          </Collapsible>

          <Collapsible title="Model Loading Errors" isDark={isDark}>
            <div className="space-y-3">
              <p className={textCls}>Check model availability and authentication:</p>
              <CodeBlock language="bash" title="terminal" code={`# Check model name and availability
python main.py info

# Login to HuggingFace for gated models (e.g., Llama)
huggingface-cli login

# Use local model path instead
# Edit config.yaml: model.base_model_path: "/path/to/model"

# Clear HuggingFace cache if corrupted
rm -rf ~/.cache/huggingface/hub/<model_id>`} />
            </div>
          </Collapsible>

          <Collapsible title="Configuration Validation Errors" isDark={isDark}>
            <div className="space-y-3">
              <p className={textCls}>Always validate your config before running:</p>
              <CodeBlock language="bash" title="terminal" code={`# Validate configuration
python main.py validate config.yaml

# Regenerate default config if yours is broken
python main.py init --output fresh_config.yaml --force

# Check logs for runtime errors
tail -f logs/dte_experiment.jsonl`} />
            </div>
          </Collapsible>

          <Collapsible title="Catastrophic Forgetting in Small Models" isDark={isDark}>
            <div className="space-y-3">
              <p className={textCls}>Models under 3B parameters may lose performance after round 2. Enable temperature annealing to mitigate this:</p>
              <CodeBlock language="yaml" title="config.yaml" code={`debate:
  temperature_annealing:
    enabled: true
    start_temp: 0.7
    end_temp: 0.3
    min_model_size: "3B"    # Applies only to models < 3B

training:
  grpo:
    kl_penalty: 0.02        # Increase if policy drifts too far`} />
              <Note type="info" isDark={isDark}>
                Temperature annealing reduces KL divergence by ~33% and recovers up to 76% of lost performance in sub-3B models.
              </Note>
            </div>
          </Collapsible>

          <Collapsible title="Slow Training Speed" isDark={isDark}>
            <div className="space-y-3">
              <CodeBlock language="yaml" title="config.yaml" code={`training:
  batch_size: 8                    # Increase if GPU allows
  gradient_accumulation_steps: 4
hardware:
  mixed_precision: true            # Use FP16/BF16
  gradient_checkpointing: true     # Trades speed for memory
  num_workers: 4                   # DataLoader workers
  dataloader_pin_memory: true`} />
            </div>
          </Collapsible>

          <Collapsible title="Debate Not Reaching Consensus" isDark={isDark}>
            <div className="space-y-3">
              <p className={textCls}>If debates rarely reach consensus:</p>
              <CodeBlock language="yaml" title="config.yaml" code={`debate:
  max_rounds: 5              # Increase from default 3
  num_agents: 3              # Odd number helps majority voting
  consensus_threshold: 0.67  # Lower threshold (2/3 majority)
model:
  temperature: 0.5           # Lower temp for more focused answers`} />
              <Note type="tip" isDark={isDark}>
                When consensus is not reached within max rounds, DTE falls back to majority voting among the final round answers.
              </Note>
            </div>
          </Collapsible>

          <Collapsible title="Import Errors / Missing Dependencies" isDark={isDark}>
            <div className="space-y-3">
              <CodeBlock language="bash" title="terminal" code={`# Reinstall all dependencies
pip install -r requirements.txt

# Or install in development mode
pip install -e ".[dev]"

# Check Python version (must be 3.8+)
python --version

# Verify torch installation
python -c "import torch; print(torch.__version__)"
python -c "import transformers; print(transformers.__version__)"
python -c "import dte; print('DTE imported successfully')"

# If torch/CUDA mismatch:
pip install torch --index-url https://download.pytorch.org/whl/cu121`} />
            </div>
          </Collapsible>
        </div>
      ),
    },

    /* ====== FAQ ====== */
    {
      id: 'faq',
      title: 'FAQ',
      icon: 'M8.228 9c.549-1.165 2.03-2 3.772-2 2.21 0 4 1.343 4 3 0 1.4-1.278 2.575-3.006 2.907-.542.104-.994.54-.994 1.093m0 3h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z',
      content: (
        <div className="space-y-6">
          <div className={cardCls}>
            <h2 className={headCls}>Frequently Asked Questions</h2>
          </div>

          {[
            { q: 'Does DTE require ground truth labels?', a: 'No. DTE is entirely ground-truth-free. The consensus answer from multi-agent debate serves as the training signal. No external supervision or labeled data is needed.' },
            { q: 'What models are supported?', a: 'Any HuggingFace-compatible causal language model. We evaluated Qwen-2.5 (0.5B, 1.5B, 3B, 7B, 14B), Llama-3.2-3B, Llama-3.1-8B, Phi-3.5-mini, and Mistral-7B. The framework also supports GPT-4o and GPT-4o-mini for debate (but not training).' },
            { q: 'How long does one evolution round take?', a: 'A single evolution round for a 7B model takes approximately 68 hours on one A100 (80GB) GPU. This includes debate generation (~8k traces), GRPO training (up to 10k steps), and evaluation. Smaller models are proportionally faster.' },
            { q: 'Why GRPO instead of SFT or DPO?', a: 'GRPO consistently outperforms both SFT and DPO in our experiments. It eliminates the need for a separate value network (unlike PPO), provides more stable training through group-relative advantages, and better utilizes the diverse reasoning traces from debate.' },
            { q: 'Can I run debates without training?', a: 'Yes! The standalone debate mode works independently: python main.py debate --query "Your question" --agents 3. This requires only inference (no GPU training needed, CPU works fine).' },
            { q: 'How many debate agents should I use?', a: 'We found 3 agents to be optimal for cost/performance. Our scaling experiments (1-7 agents) show diminishing returns beyond 5 agents, though harder problems can benefit from up to 7.' },
            { q: 'Does DTE work on non-English tasks?', a: 'The framework is language-agnostic, but our experiments were conducted on English benchmarks only. The base model needs to support the target language.' },
            { q: 'What hardware do I need?', a: 'For debate only: 8GB+ RAM, CPU is fine. For training: CUDA GPU with 16GB+ VRAM recommended. We used A100 (80GB), H100 (80GB), L40 (48GB), and A40 (48GB) GPUs. LoRA + gradient checkpointing helps fit on smaller GPUs.' },
            { q: 'What is the expected improvement?', a: 'On average, +8.92% on GSM-Plus and +5.8% cross-domain. The best single improvement was +13.92% (Qwen-1.5B on GSM-Plus). Larger models see smaller but consistent gains since they start from a higher baseline.' },
            { q: 'Can I use DTE with a private/custom dataset?', a: 'Yes. Add your dataset path to the YAML config under datasets.train_datasets. Your dataset needs a "question" and optionally a "ground_truth" field. For debate-only, just use --query with any text.' },
            { q: 'How does RCR reduce sycophancy?', a: 'RCR forces each agent to: (1) self-critique before seeing peers, (2) critique exactly 2 peers explicitly, and (3) provide novel reasoning if changing answers. This prevents blind agreement with confident-sounding wrong answers, reducing sycophancy by ~50%.' },
          ].map((item) => (
            <Collapsible key={item.q} title={item.q} isDark={isDark}>
              <p className={textCls}>{item.a}</p>
            </Collapsible>
          ))}
        </div>
      ),
    },
  ], [isDark, cardCls, headCls, subheadCls, textCls, labelCls, inlineCodeCls])

  /* ---- Search filter ---- */
  const filteredSections = useMemo(() => {
    if (!searchQuery.trim()) return sections
    const q = searchQuery.toLowerCase()
    return sections.filter(s => s.title.toLowerCase().includes(q) || s.id.toLowerCase().includes(q))
  }, [searchQuery, sections])

  const currentSection = sections.find(s => s.id === activeSection) || sections[0]

  return (
    <PageWrapper>
      <section className="pt-24 pb-20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          {/* Header */}
          <div className="text-center mb-8">
            <span className={`inline-block px-4 py-1.5 rounded-full text-xs font-semibold tracking-wider uppercase mb-4 ${
              isDark ? 'bg-electric-500/10 text-electric-300 border border-electric-500/20' : 'bg-navy-500/10 text-navy-600 border border-navy-500/20'
            }`}>Documentation</span>
            <h1 className={`text-3xl sm:text-4xl lg:text-5xl font-display font-bold mb-3 ${isDark ? 'text-white' : 'text-navy-900'}`}>
              DTE Framework Docs
            </h1>
            <p className={`text-lg max-w-2xl mx-auto ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>
              Everything you need to get started with Debate, Train, Evolve.
            </p>
          </div>

          {/* Search */}
          <div className="max-w-md mx-auto mb-10">
            <div className={`relative rounded-xl transition-all duration-300 ${isDark ? 'bg-white/[0.04] border border-white/[0.08] focus-within:border-electric-500/25' : 'bg-white border border-navy-200 focus-within:border-navy-400'}`}>
              <svg className={`absolute left-3.5 top-1/2 -translate-y-1/2 w-4 h-4 ${isDark ? 'text-gray-500' : 'text-gray-400'}`} fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
                <path d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
              </svg>
              <input
                type="text"
                placeholder="Search documentation..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className={`w-full pl-10 pr-4 py-2.5 rounded-xl text-sm bg-transparent outline-none ${
                  isDark ? 'text-white placeholder:text-gray-600' : 'text-navy-900 placeholder:text-gray-400'
                }`}
              />
            </div>
          </div>

          {/* Mobile section selector */}
          <div className="lg:hidden mb-6">
            <select
              value={activeSection}
              onChange={(e) => setActiveSection(e.target.value)}
              className={`w-full px-4 py-2.5 rounded-xl text-sm font-medium ${
                isDark
                  ? 'bg-white/5 text-white border border-white/10'
                  : 'bg-white text-navy-900 border border-navy-200'
              }`}
            >
              {sections.map(s => (
                <option key={s.id} value={s.id}>{s.title}</option>
              ))}
            </select>
          </div>

          {/* Layout: sidebar + content */}
          <div className="flex gap-8">
            {/* Sidebar */}
            <nav className={`hidden lg:block w-56 shrink-0 sticky top-24 self-start max-h-[calc(100vh-8rem)] overflow-y-auto rounded-xl p-2 ${
              isDark ? 'bg-white/[0.02] border border-white/[0.04]' : 'bg-white/60 border border-navy-200/20'
            }`}>
              <div className="space-y-0.5">
                {filteredSections.map((s) => (
                  <button
                    key={s.id}
                    onClick={() => { setActiveSection(s.id); window.scrollTo({ top: 0, behavior: 'smooth' }) }}
                    className={`w-full text-left px-3 py-2.5 rounded-lg text-sm transition-all duration-200 flex items-center gap-2.5 ${
                      activeSection === s.id
                        ? isDark
                          ? 'bg-electric-500/10 text-electric-300 font-semibold sidebar-active'
                          : 'bg-navy-500/[0.08] text-navy-700 font-semibold sidebar-active'
                        : isDark
                          ? 'text-gray-500 hover:text-gray-300 hover:bg-white/[0.04]'
                          : 'text-gray-500 hover:text-navy-700 hover:bg-navy-50'
                    }`}
                  >
                    <svg className="w-4 h-4 shrink-0" fill="none" stroke="currentColor" strokeWidth="1.5" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" d={s.icon} />
                    </svg>
                    {s.title}
                  </button>
                ))}
              </div>
            </nav>

            {/* Content */}
            <div className="flex-1 min-w-0">
              <AnimatePresence mode="wait">
                <motion.div
                  key={currentSection.id}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -10 }}
                  transition={{ duration: 0.3 }}
                >
                  {currentSection.content}
                </motion.div>
              </AnimatePresence>
            </div>
          </div>
        </div>
      </section>
    </PageWrapper>
  )
}

import { useState, useMemo } from 'react'
import { motion } from 'framer-motion'
import { useTheme } from '../context/ThemeContext'
import PageWrapper from '../components/PageWrapper'
import SectionHeading from '../components/SectionHeading'
import { mainResults, crossDomainResults, benchmarkInfo, type ModelResult } from '../data/results'

type SortKey = 'model' | 'gsm8k' | 'gsmPlus' | 'math' | 'arcC' | 'gpqa'
type SortDir = 'asc' | 'desc'

function DeltaCell({ val, isDark }: { val: number; isDark: boolean }) {
  const positive = val > 0
  const color = positive
    ? isDark ? 'text-green-400' : 'text-green-600'
    : val < 0
      ? isDark ? 'text-red-400' : 'text-red-500'
      : isDark ? 'text-gray-500' : 'text-gray-400'
  return (
    <span className={`font-mono text-xs font-medium px-1.5 py-0.5 rounded ${color} ${
      positive ? isDark ? 'bg-green-500/10' : 'bg-green-50' : val < 0 ? isDark ? 'bg-red-500/10' : 'bg-red-50' : ''
    }`}>
      {positive ? '+' : ''}{val.toFixed(2)}
    </span>
  )
}

function ResultBar({ original, evolved, max, isDark }: { original: number; evolved: number; max: number; isDark: boolean }) {
  const origW = (original / max) * 100
  const evolW = (evolved / max) * 100
  return (
    <div className="space-y-1">
      <div className={`h-1.5 rounded-full overflow-hidden ${isDark ? 'bg-white/[0.04]' : 'bg-navy-100'}`}>
        <motion.div
          initial={{ width: 0 }}
          whileInView={{ width: `${origW}%` }}
          viewport={{ once: true }}
          transition={{ duration: 0.8, ease: 'easeOut' }}
          className={`h-full rounded-full ${isDark ? 'bg-gray-600' : 'bg-gray-300'}`}
        />
      </div>
      <div className={`h-1.5 rounded-full overflow-hidden ${isDark ? 'bg-white/[0.04]' : 'bg-navy-100'}`}>
        <motion.div
          initial={{ width: 0 }}
          whileInView={{ width: `${evolW}%` }}
          viewport={{ once: true }}
          transition={{ duration: 0.8, ease: 'easeOut', delay: 0.1 }}
          className={`h-full rounded-full ${
            evolved >= original
              ? isDark ? 'bg-gradient-to-r from-electric-500 to-electric-400' : 'bg-gradient-to-r from-navy-600 to-navy-400'
              : isDark ? 'bg-red-400/60' : 'bg-red-400'
          }`}
        />
      </div>
    </div>
  )
}

export default function Results() {
  const { isDark } = useTheme()
  const [sortKey, setSortKey] = useState<SortKey>('gsmPlus')
  const [sortDir, setSortDir] = useState<SortDir>('desc')
  const [familyFilter, setFamilyFilter] = useState<string>('all')
  const [activeTab, setActiveTab] = useState(0)

  const handleSort = (key: SortKey) => {
    if (sortKey === key) setSortDir(sortDir === 'asc' ? 'desc' : 'asc')
    else { setSortKey(key); setSortDir('desc') }
  }

  const sorted = useMemo(() => {
    let data = [...mainResults]
    if (familyFilter !== 'all') data = data.filter(r => r.modelFamily === familyFilter)
    data.sort((a, b) => {
      const av = sortKey === 'model' ? a.model : a[sortKey].delta
      const bv = sortKey === 'model' ? b.model : b[sortKey].delta
      if (typeof av === 'string' && typeof bv === 'string') return sortDir === 'asc' ? av.localeCompare(bv) : bv.localeCompare(av)
      return sortDir === 'asc' ? (av as number) - (bv as number) : (bv as number) - (av as number)
    })
    return data
  }, [sortKey, sortDir, familyFilter])

  const cardCls = `rounded-2xl ${isDark ? 'glass-card' : 'glass-card-light'}`
  const thCls = `px-3 py-3 text-left text-xs font-mono uppercase tracking-wider cursor-pointer select-none transition-colors duration-200 ${
    isDark ? 'text-gray-500 hover:text-electric-300' : 'text-gray-400 hover:text-navy-600'
  }`
  const tdCls = `px-3 py-3 text-sm font-mono ${isDark ? 'text-gray-300' : 'text-gray-700'}`

  const SortIcon = ({ k }: { k: SortKey }) => {
    if (sortKey !== k) return <span className="opacity-30 ml-1">--</span>
    return <span className="ml-1">{sortDir === 'asc' ? ' ^' : ' v'}</span>
  }

  const tabs = ['Main Results', 'Cross-Domain', 'Benchmarks']

  return (
    <PageWrapper>
      <section className="pt-24 pb-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <SectionHeading
            badge="Performance"
            title="Experimental Results"
            subtitle="Comprehensive evaluation across 6 models and 7 reasoning benchmarks after one DTE evolution round."
          />

          {/* Tabs */}
          <div className={`flex flex-wrap gap-1 p-1 rounded-xl w-fit mx-auto mb-8 ${
            isDark ? 'bg-white/[0.04] border border-white/[0.06]' : 'bg-navy-100/50 border border-navy-200/30'
          }`}>
            {tabs.map((t, i) => (
              <button key={t} onClick={() => setActiveTab(i)} className={`relative px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200 ${
                activeTab === i ? isDark ? 'text-white' : 'text-navy-900' : isDark ? 'text-gray-500 hover:text-gray-300' : 'text-gray-500 hover:text-navy-700'
              }`}>
                {activeTab === i && (
                  <motion.div layoutId="resTab" className={`absolute inset-0 rounded-lg -z-10 ${isDark ? 'bg-white/[0.08] border border-electric-500/15' : 'bg-white shadow-sm border border-navy-200/30'}`} transition={{ type: 'spring', stiffness: 300, damping: 30 }} />
                )}
                {t}
              </button>
            ))}
          </div>
        </div>
      </section>

      <section className="pb-20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">

          {/* TAB 0: Main Results */}
          {activeTab === 0 && (
            <motion.div key="main" initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-8">
              {/* Filters */}
              <div className="flex flex-wrap items-center gap-3">
                <span className={`text-sm ${isDark ? 'text-gray-500' : 'text-gray-400'}`}>Filter:</span>
                {['all', 'Qwen', 'Llama'].map((f) => (
                  <button key={f} onClick={() => setFamilyFilter(f)} className={`px-3 py-1 rounded-lg text-xs font-medium transition-all duration-200 ${
                    familyFilter === f
                      ? isDark ? 'bg-electric-500/15 text-electric-300 border border-electric-500/25' : 'bg-navy-500/10 text-navy-700 border border-navy-300/40'
                      : isDark ? 'bg-white/[0.03] text-gray-500 border border-white/[0.06] hover:text-gray-300' : 'bg-navy-50 text-gray-500 border border-navy-200/40 hover:text-navy-700'
                  }`}>
                    {f === 'all' ? 'All Models' : f}
                  </button>
                ))}
              </div>

              {/* Table */}
              <div className={`${cardCls} overflow-hidden`}>
                <div className="overflow-x-auto">
                  <table className="w-full">
                    <thead>
                      <tr className={`border-b ${isDark ? 'border-white/[0.06]' : 'border-navy-200/30'}`}>
                        <th className={thCls} onClick={() => handleSort('model')}>Model<SortIcon k="model" /></th>
                        {(['gsm8k', 'gsmPlus', 'math', 'arcC', 'gpqa'] as SortKey[]).map((k) => (
                          <th key={k} className={thCls} onClick={() => handleSort(k)}>
                            {k === 'gsm8k' ? 'GSM8K' : k === 'gsmPlus' ? 'GSM-Plus' : k === 'math' ? 'MATH' : k === 'arcC' ? 'ARC-C' : 'GPQA'}
                            <SortIcon k={k} />
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {sorted.map((r: ModelResult, i: number) => (
                        <motion.tr
                          key={r.model}
                          initial={{ opacity: 0, x: -10 }}
                          animate={{ opacity: 1, x: 0 }}
                          transition={{ duration: 0.3, delay: i * 0.05 }}
                          className={`border-b last:border-0 transition-colors duration-200 ${
                            isDark ? 'border-white/[0.03] hover:bg-white/[0.02]' : 'border-navy-100/50 hover:bg-navy-50/40'
                          }`}
                        >
                          <td className={`${tdCls} font-semibold whitespace-nowrap`}>
                            <span className={isDark ? 'text-white' : 'text-navy-900'}>{r.model}</span>
                          </td>
                          {(['gsm8k', 'gsmPlus', 'math', 'arcC', 'gpqa'] as const).map((k) => (
                            <td key={k} className={tdCls}>
                              <div className="space-y-1.5">
                                <div className="flex items-center gap-2">
                                  <span className={`text-xs ${isDark ? 'text-gray-500' : 'text-gray-400'}`}>{r[k].original.toFixed(2)}</span>
                                  <span className={isDark ? 'text-gray-600' : 'text-gray-300'}>{'>'}</span>
                                  <span className={`font-semibold ${isDark ? 'text-white' : 'text-navy-900'}`}>{r[k].evolved.toFixed(2)}</span>
                                  <DeltaCell val={r[k].delta} isDark={isDark} />
                                </div>
                                <ResultBar original={r[k].original} evolved={r[k].evolved} max={100} isDark={isDark} />
                              </div>
                            </td>
                          ))}
                        </motion.tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>

              {/* Summary stats */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                {[
                  { label: 'Avg. GSM-Plus Gain', val: '+8.92%' },
                  { label: 'Best Single Gain', val: '+13.92%' },
                  { label: 'Cross-domain Avg.', val: '+5.8%' },
                  { label: 'GRPO > SFT/DPO', val: 'Consistent' },
                ].map((s) => (
                  <div key={s.label} className={`p-4 rounded-xl text-center ${isDark ? 'glass-card' : 'glass-card-light'}`}>
                    <div className={`text-xl font-bold font-display mb-1 ${isDark ? 'text-electric-300' : 'text-navy-600'}`}>{s.val}</div>
                    <div className={`text-xs ${isDark ? 'text-gray-500' : 'text-gray-400'}`}>{s.label}</div>
                  </div>
                ))}
              </div>
            </motion.div>
          )}

          {/* TAB 1: Cross-Domain */}
          {activeTab === 1 && (
            <motion.div key="cross" initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-8">
              <div className={`p-6 ${cardCls}`}>
                <h3 className={`text-lg font-display font-bold mb-2 ${isDark ? 'text-white' : 'text-navy-900'}`}>Cross-Domain Generalization</h3>
                <p className={`text-sm ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>
                  Models fine-tuned on math datasets (GSM8K / GSM-Plus) also improve on unseen science and commonsense tasks, showing that DTE captures general reasoning capabilities.
                </p>
              </div>

              <div className={`${cardCls} overflow-hidden`}>
                <div className="overflow-x-auto">
                  <table className="w-full">
                    <thead>
                      <tr className={`border-b ${isDark ? 'border-white/[0.06]' : 'border-navy-200/30'}`}>
                        <th className={thCls}>Model</th>
                        <th className={thCls}>Fine-tuned on</th>
                        <th className={thCls}>GSM8K</th>
                        <th className={thCls}>GSM-Plus</th>
                        <th className={thCls}>ARC-Easy</th>
                        <th className={thCls}>ARC-C</th>
                        <th className={thCls}>CSQA</th>
                      </tr>
                    </thead>
                    <tbody>
                      {crossDomainResults.map((r, i) => (
                        <tr key={`${r.model}-${r.fineTunedOn}`} className={`border-b last:border-0 transition-colors duration-200 ${isDark ? 'border-white/[0.03] hover:bg-white/[0.02]' : 'border-navy-100/50 hover:bg-navy-50/40'}`}>
                          <td className={`${tdCls} font-semibold ${isDark ? 'text-white' : 'text-navy-900'}`}>{i % 2 === 0 ? r.model : ''}</td>
                          <td className={tdCls}>
                            <span className={`text-xs px-2 py-0.5 rounded-md ${isDark ? 'bg-electric-500/10 text-electric-300' : 'bg-navy-100 text-navy-600'}`}>
                              {r.fineTunedOn}
                            </span>
                          </td>
                          <td className={tdCls}>{r.gsm8k !== null ? r.gsm8k.toFixed(2) : <span className={isDark ? 'text-gray-600' : 'text-gray-300'}>--</span>}</td>
                          <td className={tdCls}>{r.gsmPlus !== null ? r.gsmPlus.toFixed(2) : <span className={isDark ? 'text-gray-600' : 'text-gray-300'}>--</span>}</td>
                          <td className={tdCls}>{r.arcEasy.toFixed(2)}</td>
                          <td className={tdCls}>{r.arcChallenge.toFixed(2)}</td>
                          <td className={tdCls}>{r.commonsenseQA.toFixed(2)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </motion.div>
          )}

          {/* TAB 2: Benchmarks */}
          {activeTab === 2 && (
            <motion.div key="bench" initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-8">
              <div className={`p-6 ${cardCls}`}>
                <h3 className={`text-lg font-display font-bold mb-2 ${isDark ? 'text-white' : 'text-navy-900'}`}>Benchmark Overview</h3>
                <p className={`text-sm ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>
                  Seven diverse reasoning benchmarks testing mathematical, scientific, and commonsense reasoning.
                </p>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
                {benchmarkInfo.map((b, i) => (
                  <motion.div
                    key={b.name}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.4, delay: i * 0.06 }}
                    className={`relative p-5 rounded-xl overflow-hidden ${isDark ? 'glass-card' : 'glass-card-light'}`}
                  >
                    {/* Left accent */}
                    <div className={`absolute left-0 top-3 bottom-3 w-px ${
                      b.type === 'Math'
                        ? isDark ? 'bg-electric-500/40' : 'bg-blue-400/30'
                        : b.type === 'Science' || b.type === 'STEM'
                          ? isDark ? 'bg-cyan-500/40' : 'bg-teal-400/30'
                          : isDark ? 'bg-purple-500/40' : 'bg-purple-400/30'
                    }`} />
                    <div className="flex items-center justify-between mb-2">
                      <h4 className={`font-display font-semibold ${isDark ? 'text-white' : 'text-navy-900'}`}>{b.name}</h4>
                      <span className={`text-xs font-mono px-2 py-0.5 rounded-md ${
                        b.type === 'Math'
                          ? isDark ? 'bg-electric-500/10 text-electric-300' : 'bg-blue-50 text-blue-600'
                          : b.type === 'Science' || b.type === 'STEM'
                            ? isDark ? 'bg-cyan-500/10 text-cyan-300' : 'bg-teal-50 text-teal-600'
                            : isDark ? 'bg-purple-500/10 text-purple-300' : 'bg-purple-50 text-purple-600'
                      }`}>
                        {b.type}
                      </span>
                    </div>
                    <p className={`text-sm mb-2 ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>{b.description}</p>
                    <span className={`text-xs font-mono ${isDark ? 'text-gray-500' : 'text-gray-400'}`}>{b.samples}</span>
                  </motion.div>
                ))}
              </div>
            </motion.div>
          )}

        </div>
      </section>
    </PageWrapper>
  )
}

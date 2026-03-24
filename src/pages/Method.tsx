import { useState } from 'react'
import { motion } from 'framer-motion'
import { useTheme } from '../context/ThemeContext'
import PageWrapper from '../components/PageWrapper'
import SectionHeading from '../components/SectionHeading'

function Tabs({ tabs, active, onChange, isDark }: {
  tabs: string[]; active: number; onChange: (i: number) => void; isDark: boolean
}) {
  return (
    <div className={`flex flex-wrap gap-1 p-1 rounded-xl w-fit mx-auto mb-8 ${
      isDark ? 'bg-white/[0.04] border border-white/[0.06]' : 'bg-navy-100/50 border border-navy-200/30'
    }`}>
      {tabs.map((t, i) => (
        <button
          key={t}
          onClick={() => onChange(i)}
          className={`relative px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200 ${
            active === i
              ? isDark ? 'text-white' : 'text-navy-900'
              : isDark ? 'text-gray-500 hover:text-gray-300' : 'text-gray-500 hover:text-navy-700'
          }`}
        >
          {active === i && (
            <motion.div
              layoutId="methodTab"
              className={`absolute inset-0 rounded-lg -z-10 ${
                isDark ? 'bg-white/[0.08] border border-electric-500/15' : 'bg-white shadow-sm border border-navy-200/30'
              }`}
              transition={{ type: 'spring', stiffness: 300, damping: 30 }}
            />
          )}
          {t}
        </button>
      ))}
    </div>
  )
}

export default function Method() {
  const { isDark } = useTheme()
  const [activeTab, setActiveTab] = useState(0)

  const cardCls = `p-6 sm:p-8 rounded-2xl ${isDark ? 'glass-card' : 'glass-card-light'}`
  const headCls = `text-xl sm:text-2xl font-display font-bold mb-4 ${isDark ? 'text-white' : 'text-navy-900'}`
  const textCls = `text-sm sm:text-base leading-relaxed ${isDark ? 'text-gray-400' : 'text-gray-600'}`
  const mathCls = `font-mono text-sm ${isDark ? 'text-electric-300' : 'text-navy-600'}`
  const labelCls = `text-xs font-mono uppercase tracking-wider mb-3 ${isDark ? 'text-electric-400' : 'text-navy-500'}`

  return (
    <PageWrapper>
      <section className="pt-24 pb-12">
        <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8">
          <SectionHeading
            badge="Methodology"
            title="DTE Framework"
            subtitle="A three-phase pipeline combining multi-agent debate, GRPO training, and iterative evolution for autonomous reasoning improvement."
          />
          <Tabs
            tabs={['RCR Prompting', 'GRPO Training', 'Evolution', 'Reward Functions']}
            active={activeTab}
            onChange={setActiveTab}
            isDark={isDark}
          />
        </div>
      </section>

      <section className="pb-20">
        <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 space-y-8">

          {activeTab === 0 && (
            <motion.div key="rcr" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4 }} className="space-y-8">
              <div className={cardCls}>
                <h2 className={headCls}>Reflect-Critique-Refine (RCR) Prompting</h2>
                <p className={textCls}>
                  Standard multi-agent debate suffers from two critical failure modes: <strong>sycophancy</strong> (agents abandon correct answers for confidently-stated wrong ones) and <strong>verbosity bias</strong> (agents prefer longer rationales regardless of validity). RCR addresses both through a structured three-phase response protocol.
                </p>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                {[
                  { phase: 'Reflect', desc: 'Each agent identifies potential errors in its own current reasoning by generating a self-critique.', formula: 'c_i^self = critique(r_i^(t-1))' },
                  { phase: 'Critique', desc: 'The agent evaluates exactly 2 peer rationales, producing targeted critiques to expose flaws.', formula: '|P_i| = 2, P_i subset A \\ {a_i}' },
                  { phase: 'Refine', desc: 'The agent updates its response. If the answer changes, it must provide at least one novel reasoning step.', formula: 'y_i^(t) != y_i^(t-1) => novel(r_i^(t))' },
                ].map((p, i) => (
                  <motion.div key={p.phase} initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4, delay: i * 0.1 }} className={`${cardCls} relative overflow-hidden`}>
                    {/* Top accent bar */}
                    <div className={`absolute top-0 left-0 right-0 h-px ${
                      isDark
                        ? 'bg-gradient-to-r from-transparent via-electric-500/30 to-transparent'
                        : 'bg-gradient-to-r from-transparent via-navy-400/25 to-transparent'
                    }`} />
                    <div className={labelCls}>Phase {i + 1}</div>
                    <h3 className={`text-lg font-display font-bold mb-2 ${isDark ? 'text-white' : 'text-navy-900'}`}>{p.phase}</h3>
                    <p className={`text-sm mb-3 ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>{p.desc}</p>
                    <code className={`text-xs ${mathCls} block p-2.5 rounded-lg ${isDark ? 'bg-[#0a0f1e] border border-white/[0.06]' : 'bg-[#f7f8fc] border border-navy-200/20'}`}>{p.formula}</code>
                  </motion.div>
                ))}
              </div>

              <div className={cardCls}>
                <h3 className={`text-lg font-display font-bold mb-4 ${isDark ? 'text-white' : 'text-navy-900'}`}>Debate Protocol (Algorithm 1)</h3>
                <div className={`font-mono text-xs sm:text-sm leading-loose p-5 rounded-xl border ${isDark ? 'bg-[#0a0f1e] text-gray-300 border-white/[0.06]' : 'bg-[#f7f8fc] text-gray-700 border-navy-200/20'}`}>
                  <div><span className={mathCls}>Input:</span> query q, agents A = &#123;a_1, ..., a_N&#125;, max rounds T</div>
                  <div><span className={mathCls}>Output:</span> consensus answer y* and reasoning traces R</div>
                  <div className="mt-2"><span className={mathCls}>Round 0:</span> Each a_i generates (y_i^0, r_i^0) ~ pi(. | q)</div>
                  <div className={`pl-4 ${isDark ? 'text-gray-500' : 'text-gray-400'}`}>if all y_i^0 identical: return consensus</div>
                  <div className="mt-1"><span className={mathCls}>for</span> t = 1 to T:</div>
                  <div className="pl-4"><span className={mathCls}>for each</span> agent a_i:</div>
                  <div className="pl-8">Receive peer responses</div>
                  <div className="pl-8"><strong>Reflect:</strong> Generate self-critique c_i^self</div>
                  <div className="pl-8"><strong>Critique:</strong> Select 2 peers, generate critiques</div>
                  <div className="pl-8"><strong>Refine:</strong> Update (y_i^t, r_i^t) with novel reasoning</div>
                  <div className={`pl-4 ${isDark ? 'text-gray-500' : 'text-gray-400'}`}>if all y_i^t identical: return consensus</div>
                  <div><span className={mathCls}>return</span> majority_vote(&#123;y_i^T&#125;)</div>
                </div>
              </div>

              <div className={cardCls}>
                <h3 className={`text-lg font-display font-bold mb-4 ${isDark ? 'text-white' : 'text-navy-900'}`}>Impact of RCR</h3>
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                  {[
                    { val: '-50%', label: 'Sycophancy reduction', color: isDark ? 'text-green-400' : 'text-green-600' },
                    { val: '+8.92%', label: 'Avg. GSM-Plus gain', color: isDark ? 'text-electric-300' : 'text-navy-600' },
                    { val: '2', label: 'Fixed critique quota per agent', color: isDark ? 'text-cyan-300' : 'text-teal-600' },
                  ].map((s) => (
                    <div key={s.label} className={`p-4 rounded-xl text-center ${isDark ? 'bg-white/[0.02] border border-white/[0.06]' : 'bg-navy-50/50 border border-navy-200/20'}`}>
                      <div className={`text-2xl font-bold font-display mb-1 ${s.color}`}>{s.val}</div>
                      <div className={`text-xs ${isDark ? 'text-gray-500' : 'text-gray-500'}`}>{s.label}</div>
                    </div>
                  ))}
                </div>
              </div>
            </motion.div>
          )}

          {activeTab === 1 && (
            <motion.div key="grpo" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4 }} className="space-y-8">
              <div className={cardCls}>
                <h2 className={headCls}>Group Relative Policy Optimization (GRPO)</h2>
                <p className={textCls}>
                  GRPO eliminates the need for a separate value function by estimating advantages through group-wise comparisons. For each query, G responses are sampled from the current policy, and advantages are computed using group statistics rather than a learned baseline.
                </p>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {[
                  { label: 'Advantage Estimation', formula: 'A_hat(i,t) = (r_i - r_mean) / (sigma_r + epsilon)', note: 'r_mean = (1/G) * sum(r_j), sigma_r = std(r_j), eps = 1e-8' },
                  { label: 'Clipped Policy Loss', formula: 'L_clip(i,t) = -min(rho * A, clip(rho, 1-eps, 1+eps) * A)', note: 'rho = pi_theta / pi_old, eps = 0.2' },
                  { label: 'GRPO Objective', formula: 'L_GRPO = (1/G) * sum_i (1/|o_i|) * sum_t [L_clip - beta * D_KL]', note: 'beta = 0.02 (KL regularization)' },
                  { label: 'KL Divergence Penalty', formula: 'D_KL(i,t) = log(pi_theta / pi_ref)', note: 'Anchors policy to prevent catastrophic forgetting' },
                ].map((eq) => (
                  <div key={eq.label} className={`${cardCls} relative overflow-hidden`}>
                    <div className={`absolute top-0 left-0 right-0 h-px ${
                      isDark
                        ? 'bg-gradient-to-r from-transparent via-electric-500/20 to-transparent'
                        : 'bg-gradient-to-r from-transparent via-navy-400/15 to-transparent'
                    }`} />
                    <div className={labelCls}>{eq.label}</div>
                    <div className={`p-4 rounded-xl font-mono text-sm border ${isDark ? 'bg-[#0a0f1e] border-white/[0.06]' : 'bg-[#f7f8fc] border-navy-200/20'}`}>
                      <div className={mathCls}>{eq.formula}</div>
                      <div className={`mt-2 text-xs ${isDark ? 'text-gray-500' : 'text-gray-400'}`}>{eq.note}</div>
                    </div>
                  </div>
                ))}
              </div>

              <div className={cardCls}>
                <h3 className={`text-lg font-display font-bold mb-4 ${isDark ? 'text-white' : 'text-navy-900'}`}>Training Configuration</h3>
                <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-3">
                  {[
                    { p: 'Learning Rate', v: '5e-6' }, { p: 'Weight Decay', v: '0.1' },
                    { p: 'Batch Size', v: '8' }, { p: 'Group Size (G)', v: '8' },
                    { p: 'Clip Threshold', v: '0.2' }, { p: 'KL Coeff', v: '0.02' },
                    { p: 'LoRA Rank', v: '128' }, { p: 'LoRA Dropout', v: '0.05' },
                    { p: 'Warmup', v: '10% cosine' }, { p: 'Steps', v: '10,000' },
                    { p: 'Optimizer', v: 'AdamW 8-bit' }, { p: 'QLoRA', v: 'Enabled' },
                  ].map((h) => (
                    <div key={h.p} className={`p-3 rounded-xl transition-colors duration-200 ${isDark ? 'bg-white/[0.02] hover:bg-white/[0.04] border border-white/[0.04]' : 'bg-navy-50/40 hover:bg-navy-50/60 border border-navy-200/15'}`}>
                      <div className={`text-xs mb-1 ${isDark ? 'text-gray-500' : 'text-gray-400'}`}>{h.p}</div>
                      <div className={`text-sm font-mono font-medium ${isDark ? 'text-white' : 'text-navy-900'}`}>{h.v}</div>
                    </div>
                  ))}
                </div>
              </div>
            </motion.div>
          )}

          {activeTab === 2 && (
            <motion.div key="evo" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4 }} className="space-y-8">
              <div className={cardCls}>
                <h2 className={headCls}>Iterative Evolution</h2>
                <p className={textCls}>
                  Starting with a base policy, each evolution round generates debate traces, fine-tunes the model with GRPO, then replaces the previous agent version in the debate ensemble. The process continues until validation performance plateaus.
                </p>
              </div>

              <div className={cardCls}>
                <h3 className={`text-lg font-display font-bold mb-4 ${isDark ? 'text-white' : 'text-navy-900'}`}>DTE Evolution Algorithm</h3>
                <div className={`font-mono text-xs sm:text-sm leading-loose p-5 rounded-xl border ${isDark ? 'bg-[#0a0f1e] text-gray-300 border-white/[0.06]' : 'bg-[#f7f8fc] text-gray-700 border-navy-200/20'}`}>
                  <div><span className={mathCls}>Input:</span> base policy pi_0, query dataset Q, max iterations K</div>
                  <div><span className={mathCls}>Output:</span> evolved policy pi_K</div>
                  <div className="mt-2">Initialize: theta = theta_0</div>
                  <div><span className={mathCls}>for</span> k = 1 to K:</div>
                  <div className="pl-4">Sample batch Q_k from Q</div>
                  <div className="pl-4"><span className={mathCls}>for each</span> query q in Q_k:</div>
                  <div className="pl-8">(y*, R) = RCR_Debate(agents, q)</div>
                  <div className="pl-8">D_k = D_k U &#123;(q, y*, R)&#125;</div>
                  <div className="pl-4"><span className={mathCls}>for</span> epoch e = 1 to E:</div>
                  <div className="pl-8">Sample G responses from pi_theta</div>
                  <div className="pl-8">Compute rewards: r_i = r(q, o_i)</div>
                  <div className="pl-8">Compute advantages: A_hat</div>
                  <div className="pl-8">Update theta via L_GRPO</div>
                  <div className="pl-4">Replace agent in ensemble</div>
                  <div className={`pl-4 ${isDark ? 'text-gray-500' : 'text-gray-400'}`}>if improvement {'<'} delta: break</div>
                  <div><span className={mathCls}>return</span> pi_theta</div>
                </div>
              </div>

              <div className={cardCls}>
                <h3 className={`text-lg font-display font-bold mb-4 ${isDark ? 'text-white' : 'text-navy-900'}`}>Temperature Annealing for Smaller Models</h3>
                <p className={`${textCls} mb-4`}>
                  Models with {'<'}3B parameters suffer accuracy loss after the second evolution round due to temperature-induced KL divergence. Lowering the sampling temperature from 0.7 to 0.3 cuts KL drift by ~33% and recovers up to 76% of lost performance.
                </p>
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                  {[
                    { label: 'Start Temp', val: '0.7' },
                    { label: 'End Temp', val: '0.3' },
                    { label: 'KL Reduction', val: '~33%' },
                  ].map((s) => (
                    <div key={s.label} className={`p-4 rounded-xl text-center ${isDark ? 'bg-white/[0.02] border border-white/[0.06]' : 'bg-navy-50/50 border border-navy-200/20'}`}>
                      <div className={`font-mono text-sm mb-1 ${isDark ? 'text-gray-400' : 'text-gray-500'}`}>{s.label}</div>
                      <div className={`text-xl font-bold ${isDark ? 'text-white' : 'text-navy-900'}`}>{s.val}</div>
                    </div>
                  ))}
                </div>
              </div>
            </motion.div>
          )}

          {activeTab === 3 && (
            <motion.div key="rewards" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4 }} className="space-y-8">
              <div className={cardCls}>
                <h2 className={headCls}>Five Reward Functions</h2>
                <p className={textCls}>
                  DTE uses a multi-component shaped reward to encourage both correct answers and proper formatting. The total reward sums to a maximum of 4.0 per response. Models output structured responses using {'<reasoning>'} and {'<answer>'} XML tags.
                </p>
              </div>

              {[
                { name: 'Correctness Reward', weight: 2.0, desc: 'Checks if the extracted answer matches the consensus answer from multi-agent debate via exact string match after normalization.', formula: 'r_correct = 2.0 * 1[extract(y) == y*]', color: isDark ? 'text-green-400' : 'text-green-600' },
                { name: 'Integer/Numeric Format', weight: 0.5, desc: 'Checks whether the response contains a properly formatted numeric answer. Encourages clean numeric outputs.', formula: 'r_int = 0.5 * 1[is_numeric(extract(y))]', color: isDark ? 'text-electric-300' : 'text-navy-600' },
                { name: 'Strict XML Format', weight: 0.5, desc: 'Verifies exact adherence to the XML template: properly nested <reasoning> and <answer> tags with no extraneous content.', formula: 'r_strict = 0.5 * 1[exact_xml_match(y)]', color: isDark ? 'text-cyan-300' : 'text-teal-600' },
                { name: 'Soft XML Format', weight: 0.5, desc: 'A lenient format check awarding partial credit if the response contains required XML tags even if not perfectly structured.', formula: 'r_soft = 0.5 * 1[contains_xml_tags(y)]', color: isDark ? 'text-purple-400' : 'text-purple-600' },
                { name: 'XML Count Reward', weight: 0.5, desc: 'Granular scoring based on the number of correctly formed XML tag pairs. Graduated feedback for partial compliance.', formula: 'r_count = 0.5 * (correct_tags / expected_tags)', color: isDark ? 'text-amber-400' : 'text-amber-600' },
              ].map((r, i) => (
                <motion.div key={r.name} initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4, delay: i * 0.08 }} className={`${cardCls} relative overflow-hidden`}>
                  {/* Left accent */}
                  <div className={`absolute top-0 left-0 w-px h-full ${
                    isDark
                      ? 'bg-gradient-to-b from-transparent via-electric-500/30 to-transparent'
                      : 'bg-gradient-to-b from-transparent via-navy-400/20 to-transparent'
                  }`} />
                  <div className="flex items-start justify-between mb-3 flex-wrap gap-2">
                    <div>
                      <div className={labelCls}>Reward {i + 1}</div>
                      <h3 className={`text-lg font-display font-bold ${isDark ? 'text-white' : 'text-navy-900'}`}>{r.name}</h3>
                    </div>
                    <span className={`text-sm font-mono font-bold px-3 py-1 rounded-lg ${isDark ? 'bg-white/[0.04] border border-white/[0.06]' : 'bg-navy-50 border border-navy-200/20'} ${r.color}`}>w = {r.weight}</span>
                  </div>
                  <p className={`text-sm mb-3 ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>{r.desc}</p>
                  <code className={`text-xs ${mathCls} block p-3 rounded-lg ${isDark ? 'bg-[#0a0f1e] border border-white/[0.06]' : 'bg-[#f7f8fc] border border-navy-200/20'}`}>{r.formula}</code>
                </motion.div>
              ))}

              <div className={`${cardCls} border-2 ${isDark ? 'border-electric-500/20' : 'border-navy-300/30'}`}>
                <h3 className={`text-lg font-display font-bold mb-3 ${isDark ? 'text-white' : 'text-navy-900'}`}>Total Shaped Reward</h3>
                <div className={`p-4 rounded-xl font-mono text-sm border ${isDark ? 'bg-[#0a0f1e] border-white/[0.06]' : 'bg-[#f7f8fc] border-navy-200/20'}`}>
                  <div className={mathCls}>r(q, y) = 2.0*correctness + 0.5*int_format + 0.5*strict_xml + 0.5*soft_xml + 0.5*xml_count</div>
                  <div className={`mt-2 text-xs ${isDark ? 'text-gray-500' : 'text-gray-400'}`}>Maximum total reward: 4.0 per response</div>
                </div>
              </div>
            </motion.div>
          )}
        </div>
      </section>
    </PageWrapper>
  )
}

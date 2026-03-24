import { motion } from 'framer-motion'
import { useTheme } from '../context/ThemeContext'
import PageWrapper from '../components/PageWrapper'
import SectionHeading from '../components/SectionHeading'

const members = [
  {
    name: 'Gaurav Srivastava',
    role: 'Lead Researcher',
    email: 'gks@vt.edu',
    affiliation: 'Virginia Tech, CS',
    initial: 'GS',
    note: 'Lead author',
  },
  {
    name: 'Zhenyu Bi',
    role: 'Researcher',
    email: 'zhenyub@vt.edu',
    affiliation: 'Virginia Tech, CS',
    initial: 'ZB',
  },
  {
    name: 'Meng Lu',
    role: 'Researcher',
    email: 'menglu@vt.edu',
    affiliation: 'Virginia Tech, CS',
    initial: 'ML',
  },
  {
    name: 'Xuan Wang',
    role: 'Corresponding Author',
    email: 'xuanw@vt.edu',
    affiliation: 'Virginia Tech, CS',
    initial: 'XW',
    note: 'Corresponding author',
  },
]

export default function Team() {
  const { isDark } = useTheme()

  return (
    <PageWrapper>
      <section className="pt-24 pb-20">
        <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8">
          <SectionHeading
            badge="Team"
            title="Research Team"
            subtitle="Department of Computer Science, Virginia Tech"
          />

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-6 mt-12">
            {members.map((m, i) => (
              <motion.div
                key={m.name}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.5, delay: i * 0.1 }}
                whileHover={{ y: -4 }}
                className={`relative group p-6 rounded-2xl transition-all duration-300 overflow-hidden ${
                  isDark
                    ? 'glass-card hover:border-electric-500/25 hover:shadow-lg hover:shadow-electric-500/[0.08]'
                    : 'glass-card-light hover:border-navy-300/40 hover:shadow-lg hover:shadow-navy-500/[0.08]'
                }`}
              >
                {/* Gradient background on hover */}
                <div className={`absolute inset-0 rounded-2xl opacity-0 group-hover:opacity-100 transition-opacity duration-500 ${
                  isDark
                    ? 'bg-gradient-to-br from-electric-500/[0.03] via-transparent to-cyan-500/[0.03]'
                    : 'bg-gradient-to-br from-navy-500/[0.02] via-transparent to-electric-500/[0.02]'
                }`} />

                <div className="flex items-start gap-4 relative z-10">
                  <div className={`w-14 h-14 rounded-xl flex items-center justify-center font-bold text-lg font-display shrink-0 transition-all duration-300 ${
                    isDark
                      ? 'bg-gradient-to-br from-electric-500/20 to-cyan-500/15 text-electric-300 border border-electric-500/20 group-hover:border-electric-400/35 group-hover:shadow-md group-hover:shadow-electric-500/15'
                      : 'bg-gradient-to-br from-navy-500/10 to-electric-500/8 text-navy-600 border border-navy-300/30 group-hover:border-navy-400/40 group-hover:shadow-md group-hover:shadow-navy-500/10'
                  }`}>
                    {m.initial}
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 flex-wrap">
                      <h3 className={`text-lg font-display font-bold ${isDark ? 'text-white' : 'text-navy-900'}`}>
                        {m.name}
                      </h3>
                      {m.note && (
                        <span className={`text-[10px] font-mono uppercase tracking-wider px-2 py-0.5 rounded ${
                          isDark ? 'bg-electric-500/10 text-electric-400' : 'bg-navy-100 text-navy-500'
                        }`}>
                          {m.note}
                        </span>
                      )}
                    </div>
                    <p className={`text-sm mt-0.5 ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>
                      {m.role}
                    </p>
                    <p className={`text-xs mt-1 ${isDark ? 'text-gray-500' : 'text-gray-400'}`}>
                      {m.affiliation}
                    </p>
                    <a
                      href={`mailto:${m.email}`}
                      className={`inline-flex items-center gap-1.5 text-xs mt-3 transition-colors duration-200 ${
                        isDark ? 'text-electric-400 hover:text-electric-300' : 'text-navy-500 hover:text-navy-700'
                      }`}
                    >
                      <svg className="w-3 h-3" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
                        <path d="M3 8l7.89 5.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                      </svg>
                      {m.email}
                    </a>
                  </div>
                </div>
              </motion.div>
            ))}
          </div>

          {/* Acknowledgments */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.5, delay: 0.3 }}
            className={`mt-12 p-6 rounded-2xl ${isDark ? 'glass-card' : 'glass-card-light'}`}
          >
            <h3 className={`text-lg font-display font-semibold mb-3 ${isDark ? 'text-white' : 'text-navy-900'}`}>
              Acknowledgments
            </h3>
            <p className={`text-sm leading-relaxed ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>
              This work was supported by NSF NAIRR Pilot with PSC Neocortex and NCSA Delta; Amazon, Cisco Research,
              Commonwealth Cyber Initiative, Amazon-Virginia Tech Center for Efficient and Robust Machine Learning,
              and the Sanghani Center for AI and Data Analytics at Virginia Tech.
            </p>
          </motion.div>
        </div>
      </section>
    </PageWrapper>
  )
}

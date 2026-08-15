import React from 'react'
import { motion } from 'framer-motion'
import { LIMITS } from '../data/content'
import { AlertTriangle, Clock, ShieldAlert } from 'lucide-react'

function getIcon(idx) {
  if (idx === 0) return <ShieldAlert className="text-accentRed" size={20} />
  if (idx === 1) return <Clock className="text-accentYellow" size={20} />
  return <AlertTriangle className="text-accentBlue" size={20} />
}

export default function HonestLimits() {
  return (
    <section id="limits" className="relative w-full py-20 px-4 sm:px-6 lg:px-8 bg-[#0d1117] border-b border-[#30363d]/60 z-10">
      <div className="max-w-4xl mx-auto">
        <div className="text-center mb-16">
          <h2 className="text-3xl font-bold bg-gradient-to-r from-accentRed via-accentYellow to-accentBlue bg-clip-text text-transparent mb-2">
            Honest Limits & Scope
          </h2>
          <p className="text-gray-400 text-sm">
            Scope of the current evidence and verification contract
          </p>
        </div>

        <div className="space-y-6">
          {LIMITS.map((limit, idx) => (
            <motion.div
              key={idx}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: idx * 0.1 }}
              className="glass-panel p-5 rounded-2xl flex items-start gap-4 border border-border/80"
            >
              <div className="p-2.5 rounded-xl bg-surface/80 border border-border">
                {getIcon(idx)}
              </div>
              <div>
                <h3 className="text-sm font-bold text-white mb-1">{limit.title}</h3>
                <p className="text-gray-300 text-xs leading-relaxed">{limit.desc}</p>
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  )
}

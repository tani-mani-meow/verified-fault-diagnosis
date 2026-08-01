import React from 'react'
import { STATS } from '../data/content'
import { motion } from 'framer-motion'

export default function StatsGrid() {
  return (
    <section className="relative w-full py-16 px-4 sm:px-6 lg:px-8 bg-[#0d1117]/80 border-b border-[#30363d]/60 z-10">
      <div className="max-w-7xl mx-auto">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
          {STATS.map((stat, idx) => (
            <motion.div
              key={idx}
              initial={{ opacity: 0, scale: 0.95 }}
              whileInView={{ opacity: 1, scale: 1 }}
              viewport={{ once: true }}
              transition={{ delay: idx * 0.05 }}
              className="glass-panel p-6 rounded-2xl border border-border/60 hover:border-accentBlue/40 transition-colors flex flex-col justify-center items-center text-center shadow-md"
            >
              <span className="text-3xl font-extrabold text-white bg-gradient-to-r from-accentBlue to-accentPurple bg-clip-text text-transparent mb-1">
                {stat.value}
              </span>
              <span className="text-[10px] uppercase font-bold text-gray-500 tracking-wider">
                {stat.label}
              </span>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  )
}

import React from 'react'
import { motion } from 'framer-motion'
import { NOVELTY } from '../data/content'
import { ArrowRight, Star } from 'lucide-react'

function NoveltyCard({ item, index }) {
  return (
    <motion.div
      style={{ perspective: 1000 }}
      whileHover={{ 
        rotateY: index === 0 ? 8 : index === 2 ? -8 : 4,
        rotateX: 6,
        scale: 1.02,
        y: -10
      }}
      transition={{ type: 'spring', stiffness: 300, damping: 20 }}
      className="glass-panel p-6 rounded-2xl flex flex-col justify-between h-[280px] shadow-xl border border-border/80 transition-shadow hover:shadow-[#58a6ff]/10"
    >
      <div>
        <div className="flex items-center gap-2 mb-3">
          <div className="p-1.5 rounded bg-accentPurple/10 text-accentPurple">
            <Star size={16} fill="#bc8cff" />
          </div>
          <h3 className="text-base font-bold text-white">{item.title}</h3>
        </div>

        <div className="space-y-4">
          <div className="text-xs">
            <span className="text-gray-500 uppercase font-semibold tracking-wider block mb-1">Traditional Approach</span>
            <p className="text-[#f85149] font-medium">{item.past}</p>
          </div>

          <div className="flex items-center justify-center my-1 text-gray-500">
            <ArrowRight size={14} className="rotate-90 sm:rotate-0" />
          </div>

          <div className="text-xs">
            <span className="text-accentGreen uppercase font-semibold tracking-wider block mb-1">Our Verification Pipeline</span>
            <p className="text-accentGreen font-medium">{item.ours}</p>
          </div>
        </div>
      </div>
    </motion.div>
  )
}

export default function NoveltySection() {
  return (
    <section id="novelty" className="relative w-full py-20 px-4 sm:px-6 lg:px-8 bg-[#0d1117]/60 border-b border-[#30363d]/60 z-10">
      <div className="max-w-7xl mx-auto">
        <div className="text-center mb-16">
          <h2 className="text-3xl font-bold bg-gradient-to-r from-accentPurple to-accentBlue bg-clip-text text-transparent mb-2">
            Novelty & Comparison
          </h2>
          <p className="text-gray-400 text-sm max-w-xl mx-auto">
            How our verified diagnostic framework compares against existing state-of-the-art LLM methods for predictive maintenance
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          {NOVELTY.map((item, idx) => (
            <NoveltyCard key={idx} item={item} index={idx} />
          ))}
        </div>
      </div>
    </section>
  )
}

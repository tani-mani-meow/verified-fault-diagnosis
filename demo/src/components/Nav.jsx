import React, { useState, useEffect } from 'react'

export default function Nav() {
  const [scrollProgress, setScrollProgress] = useState(0)

  useEffect(() => {
    const handleScroll = () => {
      const totalScroll = document.documentElement.scrollHeight - window.innerHeight
      if (totalScroll > 0) {
        setScrollProgress((window.scrollY / totalScroll) * 100)
      }
    }
    window.addEventListener('scroll', handleScroll)
    return () => window.removeEventListener('scroll', handleScroll)
  }, [])

  const navItems = [
    { label: 'Overview', id: 'overview' },
    { label: 'Pipeline', id: 'pipeline' },
    { label: 'Gradient', id: 'gradient' },
    { label: 'Demo', id: 'demo' },
    { label: 'Ablation', id: 'ablation' },
    { label: 'Novelty', id: 'novelty' },
    { label: 'Limits', id: 'limits' }
  ]

  const scrollToSection = (id) => {
    const el = document.getElementById(id)
    if (el) {
      el.scrollIntoView({ behavior: 'smooth' })
    }
  }

  return (
    <nav className="fixed top-0 left-0 w-full z-50 bg-[#0d1117]/80 backdrop-blur-md border-b border-[#30363d]/60">
      {/* Scroll Progress Bar */}
      <div 
        className="h-[3px] bg-gradient-to-r from-accentBlue via-accentPurple to-accentGreen transition-all duration-100" 
        style={{ width: `${scrollProgress}%` }}
      />
      
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <span className="text-xl font-bold bg-gradient-to-r from-accentBlue to-accentPurple bg-clip-text text-transparent">
            Verified Fault Diagnosis
          </span>
          <span className="hidden md:inline-block px-2 py-0.5 text-xs rounded border border-[#30363d] bg-surface text-gray-400">
            Llama 3.1 70B
          </span>
        </div>
        
        <div className="flex space-x-1 md:space-x-4">
          {navItems.map((item) => (
            <button
              key={item.id}
              onClick={() => scrollToSection(item.id)}
              className="px-2 py-1.5 md:px-3 text-xs md:text-sm font-medium text-gray-300 hover:text-white rounded-md transition-colors"
            >
              {item.label}
            </button>
          ))}
        </div>
      </div>
    </nav>
  )
}

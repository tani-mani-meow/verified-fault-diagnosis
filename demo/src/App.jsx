import React from 'react'
import Nav from './components/Nav'
import HeroScene from './components/HeroScene'
import StatsGrid from './components/StatsGrid'
import PipelineScene from './components/PipelineScene'
import GradientScene from './components/GradientScene'
import VerificationDemo from './components/VerificationDemo'
import MechanismGuard from './components/MechanismGuard'
import AblationHeatmap from './components/AblationHeatmap'
import NoveltySection from './components/NoveltySection'
import HonestLimits from './components/HonestLimits'
import { Github, Database, FileText } from 'lucide-react'

export default function App() {
  return (
    <div className="relative min-h-screen bg-bg text-text">
      {/* Top Navbar & Scroll Progress */}
      <Nav />
      
      {/* Sections */}
      <HeroScene />
      <StatsGrid />
      <PipelineScene />
      <GradientScene />
      <VerificationDemo />
      <MechanismGuard />
      <AblationHeatmap />
      <NoveltySection />
      <HonestLimits />
      
      {/* Footer */}
      <footer className="relative w-full py-12 px-4 sm:px-6 lg:px-8 bg-[#0d1117]/95 border-t border-[#30363d]/60 z-10">
        <div className="max-w-7xl mx-auto flex flex-col md:flex-row items-center justify-between gap-6">
          <div className="flex flex-col items-center md:items-start gap-1">
            <span className="text-base font-bold bg-gradient-to-r from-accentBlue to-accentPurple bg-clip-text text-transparent">
              Verified Fault Diagnosis
            </span>
            <span className="text-xs text-gray-500">
              Interactive 3D Research Showcase &copy; 2026
            </span>
          </div>

          <div className="flex gap-6 text-xs text-gray-400">
            <a 
              href="https://github.com/atharvoid/verified-fault-diagnosis" 
              target="_blank" 
              rel="noreferrer" 
              className="flex items-center gap-1.5 hover:text-white transition-colors"
            >
              <Github size={14} /> GitHub Code
            </a>
            <a 
              href="https://ieeexplore.ieee.org/document/7311508" 
              target="_blank" 
              rel="noreferrer" 
              className="flex items-center gap-1.5 hover:text-white transition-colors"
            >
              <Database size={14} /> IITK Dataset (Verma 2016)
            </a>
            <a 
              href="https://integrate.api.nvidia.com" 
              target="_blank" 
              rel="noreferrer" 
              className="flex items-center gap-1.5 hover:text-white transition-colors"
            >
              <FileText size={14} /> Llama 3.1 70B (NIM)
            </a>
          </div>
        </div>
      </footer>
    </div>
  )
}

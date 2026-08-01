import React, { useState, useEffect } from 'react'
import { Canvas } from '@react-three/fiber'
import { Stars, Float, Text, OrbitControls } from '@react-three/drei'
import { DEMO_CARDS } from '../data/content'
import { motion, AnimatePresence } from 'framer-motion'
import { Check, X, Shield, AlertTriangle, Play } from 'lucide-react'

function FloatingCard({ card, index, activeCard, onClick }) {
  const isSelected = activeCard === card.id
  
  return (
    <Float speed={2} rotationIntensity={0.3} floatIntensity={0.8}>
      <Text
        font="https://fonts.gstatic.com/s/outfit/v11/0oWk0yFbf1dQZ5cTaFSf124.woff"
        fontSize={0.24}
        position={[(index - 1) * 1.8, 0, 0]}
        color="#ffffff"
        backgroundColor={card.color + "22"}
        padding={[0.2, 0.3]}
        borderRadius={0.1}
        strokeWidth="3%"
        strokeColor={card.color}
        onClick={onClick}
        className="cursor-pointer"
      >
        {card.title}
      </Text>
    </Float>
  )
}

export default function VerificationDemo() {
  const [activeCard, setActiveCard] = useState(null)
  const [visibleClaims, setVisibleClaims] = useState(0)
  const [showVerdict, setShowVerdict] = useState(false)

  const selectedCard = DEMO_CARDS.find(c => c.id === activeCard)

  useEffect(() => {
    if (activeCard && selectedCard) {
      setVisibleClaims(0)
      setShowVerdict(false)
      
      // Animate claims one by one
      const timers = []
      selectedCard.claims.forEach((_, idx) => {
        timers.push(
          setTimeout(() => {
            setVisibleClaims(prev => prev + 1)
          }, (idx + 1) * 400)
        )
      })

      // Show verdict stamp at the end
      timers.push(
        setTimeout(() => {
          setShowVerdict(true)
        }, (selectedCard.claims.length + 1) * 400)
      )

      return () => timers.forEach(t => clearTimeout(t))
    }
  }, [activeCard])

  return (
    <div id="demo" className="relative w-full h-screen bg-[#0d1117] flex items-center justify-center overflow-hidden border-b border-[#30363d]/60">
      
      {/* 3D Scene */}
      <div className="absolute inset-0 z-0">
        <Canvas camera={{ position: [0, 0, 4.5], fof: 50 }}>
          <ambientLight intensity={0.4} />
          <pointLight position={[10, 10, 10]} intensity={1.5} color="#58a6ff" />
          
          <Stars radius={100} depth={50} count={2000} factor={4} saturation={0.5} fade speed={1} />
          
          {DEMO_CARDS.map((card, idx) => (
            <FloatingCard
              key={card.id}
              card={card}
              index={idx}
              activeCard={activeCard}
              onClick={() => setActiveCard(card.id)}
            />
          ))}
          
          <OrbitControls enableZoom={false} maxPolarAngle={Math.PI / 2} minPolarAngle={Math.PI / 2} />
        </Canvas>
      </div>

      {/* Title Overlay */}
      <div className="absolute top-20 left-1/2 transform -translate-x-1/2 text-center z-10 pointer-events-none w-full max-w-xl px-4">
        <h2 className="text-3xl font-bold bg-gradient-to-r from-accentBlue to-accentPurple bg-clip-text text-transparent mb-2">
          Live Verification Demo
        </h2>
        <p className="text-gray-400 text-sm">
          Click a floating 3D status card below to run the deterministic verifier and watch claims evaluate in real-time
        </p>
      </div>

      {/* Detailed Modal/Slide-up Panel (Framer Motion) */}
      <AnimatePresence>
        {activeCard && selectedCard && (
          <motion.div
            initial={{ opacity: 0, y: 50 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 50 }}
            className="absolute inset-0 z-20 flex items-center justify-center bg-bg/85 backdrop-blur-md p-4 sm:p-6"
          >
            <div className="w-full max-w-3xl glass-panel rounded-2xl shadow-2xl p-6 relative max-h-[85vh] overflow-y-auto">
              {/* Close button */}
              <button
                onClick={() => setActiveCard(null)}
                className="absolute top-4 right-4 p-2 rounded-full hover:bg-border/50 text-gray-400 hover:text-white transition-colors"
              >
                <X size={20} />
              </button>

              {/* Title & Status */}
              <div className="border-b border-border/80 pb-4 mb-4">
                <h3 className="text-xl font-bold flex items-center gap-2" style={{ color: selectedCard.color }}>
                  {selectedCard.title}
                </h3>
                <p className="text-xs text-gray-400 uppercase tracking-widest mt-1">Live Pipeline Trace</p>
              </div>

              {/* Generated LLM Text Box */}
              <div className="bg-surface/50 border border-border/50 rounded-xl p-4 mb-6">
                <h4 className="text-xs font-semibold text-gray-400 uppercase mb-2">Generated Technician Explanation</h4>
                <p className="text-gray-200 text-sm leading-relaxed italic">
                  "{selectedCard.llm}"
                </p>
              </div>

              {/* Claims Table */}
              <div className="mb-6">
                <h4 className="text-xs font-semibold text-gray-400 uppercase mb-3">Extracted Factual Claims Checklist</h4>
                <div className="space-y-2.5">
                  {selectedCard.claims.map((claim, idx) => {
                    const isVisible = idx < visibleClaims
                    return (
                      <motion.div
                        key={idx}
                        initial={{ opacity: 0, x: -20 }}
                        animate={isVisible ? { opacity: 1, x: 0 } : { opacity: 0, x: -20 }}
                        transition={{ duration: 0.3 }}
                        className={`flex items-center justify-between p-3 rounded-lg border bg-surface/30 border-l-4 ${
                          isVisible 
                            ? claim.ok 
                              ? 'border-l-accentGreen border-border' 
                              : 'border-l-accentRed border-border'
                            : 'border-transparent border-transparent'
                        }`}
                      >
                        <div className="flex flex-col">
                          <span className="text-sm font-semibold text-white">{claim.feat}</span>
                          <span className="text-xs text-gray-400">
                            Claimed: <strong className="text-gray-300">{claim.says}</strong> | Actual: {claim.actual}
                          </span>
                        </div>
                        
                        {isVisible && (
                          <div className={`p-1.5 rounded-full ${claim.ok ? 'bg-accentGreen/10 text-accentGreen' : 'bg-accentRed/10 text-accentRed'}`}>
                            {claim.ok ? <Check size={16} /> : <X size={16} />}
                          </div>
                        )}
                      </motion.div>
                    )
                  })}
                </div>
              </div>

              {/* Verdict Rubber Stamp (Framer Motion) */}
              <AnimatePresence>
                {showVerdict && (
                  <motion.div
                    initial={{ scale: 2, opacity: 0, rotate: -15 }}
                    animate={{ scale: 1, opacity: 1, rotate: -4 }}
                    transition={{ type: 'spring', damping: 12 }}
                    className="flex justify-center items-center mt-6"
                  >
                    <div 
                      className="px-6 py-3 rounded-xl border-4 font-black uppercase text-lg tracking-wider flex items-center gap-3 bg-bg/95"
                      style={{ 
                        color: selectedCard.color, 
                        borderColor: selectedCard.color,
                        boxShadow: `0 0 20px ${selectedCard.color}44`
                      }}
                    >
                      {selectedCard.title.includes("RELEASED") ? (
                        <Shield size={24} className="text-accentGreen" />
                      ) : (
                        <AlertTriangle size={24} className="text-accentRed" />
                      )}
                      {selectedCard.verdict}
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

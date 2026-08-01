import React, { useState, useRef } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { OrbitControls, Stars, Text, Float } from '@react-three/drei'
import { MECHANISM_CHIPS } from '../data/content'
import { motion, AnimatePresence } from 'framer-motion'
import { X, CheckCircle, AlertTriangle } from 'lucide-react'

function CentralCore() {
  const meshRef = useRef()

  useFrame((state) => {
    if (meshRef.current) {
      meshRef.current.rotation.y += 0.005
      meshRef.current.rotation.z = Math.sin(state.clock.getElapsedTime()) * 0.2
    }
  })

  return (
    <mesh ref={meshRef}>
      <icosahedronGeometry args={[0.5, 1]} />
      <meshStandardMaterial
        color="#bc8cff"
        emissive="#bc8cff"
        emissiveIntensity={0.3}
        wireframe
      />
    </mesh>
  )
}

function ChipNode({ chip, index, activeChip, onClick }) {
  const meshRef = useRef()
  const isSelected = activeChip?.name === chip.name
  const color = chip.supported ? '#3fb950' : '#f85149'

  // Radial positioning
  const angle = (index * 2 * Math.PI) / 7
  const radius = 2.0
  const posX = Math.cos(angle) * radius
  const posY = Math.sin(angle) * radius

  useFrame((state) => {
    if (meshRef.current) {
      // Gentle spin
      meshRef.current.rotation.z = angle
      if (isSelected) {
        meshRef.current.scale.setScalar(1.2 + Math.sin(state.clock.getElapsedTime() * 8) * 0.05)
      } else {
        meshRef.current.scale.setScalar(1.0)
      }
    }
  })

  return (
    <group position={[posX, posY, 0]}>
      <mesh
        ref={meshRef}
        onClick={onClick}
        className="cursor-pointer"
      >
        <boxGeometry args={[0.7, 0.3, 0.15]} />
        <meshStandardMaterial
          color={color}
          roughness={0.2}
          metalness={0.8}
          emissive={color}
          emissiveIntensity={isSelected ? 1.0 : 0.2}
        />
      </mesh>

      <Text
        font="https://fonts.gstatic.com/s/outfit/v11/0oWk0yFbf1dQZ5cTaFSf124.woff"
        fontSize={0.16}
        position={[0, 0, 0.1]}
        color="#ffffff"
        anchorX="center"
        anchorY="middle"
      >
        {chip.name}
      </Text>
    </group>
  )
}

export default function MechanismGuard() {
  const [activeChip, setActiveChip] = useState(null)

  return (
    <div className="relative w-full h-[600px] bg-[#0d1117]/40 flex items-center justify-center overflow-hidden border-b border-[#30363d]/60">
      
      {/* 3D Scene */}
      <div className="absolute inset-0 z-0">
        <Canvas camera={{ position: [0, 0, 3.8], fov: 50 }}>
          <ambientLight intensity={0.4} />
          <pointLight position={[10, 10, 10]} intensity={1.5} color="#58a6ff" />
          
          <Stars radius={100} depth={50} count={1000} factor={4} saturation={0.5} fade speed={1} />
          
          <CentralCore />
          
          {MECHANISM_CHIPS.map((chip, idx) => (
            <ChipNode
              key={chip.name}
              chip={chip}
              index={idx}
              activeChip={activeChip}
              onClick={() => setActiveChip(chip)}
            />
          ))}
          
          <OrbitControls enableZoom={false} enablePan={false} />
        </Canvas>
      </div>

      {/* Header Overlay */}
      <div className="absolute top-10 left-1/2 transform -translate-x-1/2 text-center z-10 pointer-events-none w-full max-w-xl px-4">
        <h2 className="text-2xl font-bold bg-gradient-to-r from-accentPurple to-accentBlue bg-clip-text text-transparent mb-1">
          Causal Mechanism Guard
        </h2>
        <p className="text-gray-400 text-xs">
          Click any class chip to check statistical support for impulsive wording.
          <br />
          Global Thresholds: Kurtosis &ge; 2.84 | Crest Factor &ge; 7.08
        </p>
      </div>

      {/* Detail Panel */}
      <AnimatePresence>
        {activeChip && (
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.95 }}
            className="absolute z-10 w-[90%] max-w-md bg-surface/95 border border-border/80 backdrop-blur-md rounded-xl p-5 shadow-2xl"
          >
            <button
              onClick={() => setActiveChip(null)}
              className="absolute top-3 right-3 p-1 rounded-full hover:bg-border/50 text-gray-400 hover:text-white"
            >
              <X size={16} />
            </button>

            <div className="flex items-center gap-2 border-b border-border/50 pb-3 mb-4">
              <span className={`text-base font-bold ${activeChip.supported ? 'text-accentGreen' : 'text-accentRed'}`}>
                Class: {activeChip.name}
              </span>
              <span className={`text-[10px] px-2 py-0.5 rounded-full font-bold uppercase tracking-wider ${
                activeChip.supported ? 'bg-accentGreen/10 text-accentGreen' : 'bg-accentRed/10 text-accentRed'
              }`}>
                {activeChip.supported ? 'Supported' : 'Not Supported'}
              </span>
            </div>

            <div className="space-y-3 text-xs">
              <div className="flex justify-between items-center bg-bg/50 p-2.5 rounded border border-border/50">
                <span>Kurtosis: <strong>{activeChip.kurt}</strong></span>
                <span className="text-gray-500">vs</span>
                <span>Threshold: &ge; 2.84</span>
                {activeChip.kurt >= 2.84 ? (
                  <CheckCircle size={14} className="text-accentGreen" />
                ) : (
                  <AlertTriangle size={14} className="text-accentRed" />
                )}
              </div>

              <div className="flex justify-between items-center bg-bg/50 p-2.5 rounded border border-border/50">
                <span>Crest Factor: <strong>{activeChip.crest}</strong></span>
                <span className="text-gray-500">vs</span>
                <span>Threshold: &ge; 7.08</span>
                {activeChip.crest >= 7.08 ? (
                  <CheckCircle size={14} className="text-accentGreen" />
                ) : (
                  <AlertTriangle size={14} className="text-accentRed" />
                )}
              </div>

              <p className="text-gray-300 italic mt-3 bg-surface border-l-2 border-border p-2">
                {activeChip.desc}
              </p>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

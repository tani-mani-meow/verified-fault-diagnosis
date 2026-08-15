import React, { useState, useRef, useEffect } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { OrbitControls, Stars, Float, Text, Line } from '@react-three/drei'
import { GRADIENT_BARS } from '../data/content'
import { motion, AnimatePresence } from 'framer-motion'

function Bar({ data, index, hoveredBar, setHoveredBar }) {
  const meshRef = useRef()
  const targetHeight = data.height * 2.5
  const isHovered = hoveredBar === index

  useFrame((state) => {
    if (meshRef.current) {
      // Animate growth
      const currentScaleY = meshRef.current.scale.y
      const diff = 1.0 - currentScaleY
      if (diff > 0.001) {
        meshRef.current.scale.y += diff * 0.05
        meshRef.current.position.y = (meshRef.current.scale.y * targetHeight) / 2
      }
      
      // Floating animation for hovered bar
      if (isHovered) {
        meshRef.current.rotation.y += 0.015
      } else {
        meshRef.current.rotation.y = index * 0.1
      }
    }
  })

  return (
    <group position={[(index - 1) * 1.8, 0, 0]}>
      {/* Platform/Base under each bar */}
      <mesh position={[0, -0.05, 0]}>
        <cylinderGeometry args={[0.5, 0.55, 0.1, 16]} />
        <meshStandardMaterial color="#21262d" roughness={0.5} />
      </mesh>

      {/* The 3D Bar */}
      <mesh
        ref={meshRef}
        scale={[1, 0, 1]}
        position={[0, 0, 0]}
        onPointerOver={() => setHoveredBar(index)}
        onPointerOut={() => setHoveredBar(null)}
        className="cursor-pointer"
      >
        <boxGeometry args={[0.6, targetHeight, 0.6]} />
        <meshStandardMaterial
          color={data.color}
          roughness={0.1}
          metalness={0.8}
          emissive={data.color}
          emissiveIntensity={isHovered ? 0.8 : 0.2}
        />
      </mesh>

      {/* Height labels inside the scene */}
      <Text
        font="https://fonts.gstatic.com/s/outfit/v11/0oWk0yFbf1dQZ5cTaFSf124.woff"
        fontSize={0.22}
        position={[0, targetHeight + 0.3, 0]}
        color="#ffffff"
        anchorX="center"
        anchorY="bottom"
      >
        {data.height.toFixed(2)}
      </Text>

      {/* Condition Name label */}
      <Text
        font="https://fonts.gstatic.com/s/inter/v18/UcCO3FwrK3iLTeHuS_fvQtMwCp50KnMw2bo.woff"
        fontSize={0.15}
        position={[0, -0.3, 0.2]}
        color="#8b949e"
        anchorX="center"
        anchorY="top"
      >
        {data.label}
      </Text>
    </group>
  )
}

export default function GradientScene() {
  const [hoveredBar, setHoveredBar] = useState(null)
  
  // Heights scaled for the connecting arrow
  const p1 = [-1.8, 0.34 * 2.5 + 0.3, 0]
  const p2 = [0, 0.63 * 2.5 + 0.3, 0]
  const p3 = [1.8, 0.96 * 2.5 + 0.3, 0]

  return (
    <div id="gradient" className="relative w-full h-screen bg-[#0d1117] flex items-center justify-center overflow-hidden border-b border-[#30363d]/60">
      
      {/* 3D Scene */}
      <div className="absolute inset-0 z-0">
        <Canvas camera={{ position: [0, 1.2, 4.5], fov: 50 }}>
          <ambientLight intensity={0.4} />
          <pointLight position={[10, 10, 10]} intensity={1.5} color="#58a6ff" />
          <pointLight position={[-10, -10, -10]} intensity={0.5} color="#bc8cff" />
          
          <Stars radius={100} depth={50} count={2000} factor={4} saturation={0.5} fade speed={1} />
          
          {/* Ground Platform */}
          <mesh position={[0, -0.15, 0]} rotation={[-Math.PI / 2, 0, 0]}>
            <planeGeometry args={[10, 10]} />
            <meshStandardMaterial color="#0d1117" roughness={0.9} />
          </mesh>

          {/* Connecting gradient arrow line */}
          <Line
            points={[p1, p2, p3]}
            color="#bc8cff"
            lineWidth={3}
            dashed
            dashScale={2}
          />
          
          {/* 3D Bars */}
          {GRADIENT_BARS.map((bar, idx) => (
            <Bar
              key={idx}
              data={bar}
              index={idx}
              hoveredBar={hoveredBar}
              setHoveredBar={setHoveredBar}
            />
          ))}
          
          <OrbitControls enableZoom={false} maxPolarAngle={Math.PI / 2.1} minPolarAngle={Math.PI / 2.5} />
        </Canvas>
      </div>

      {/* Floating Info Overlay for hovered bar */}
      <div className="absolute top-20 left-1/2 transform -translate-x-1/2 text-center z-10 pointer-events-none w-full max-w-xl px-4">
        <h2 className="text-3xl font-bold bg-gradient-to-r from-accentRed via-accentYellow to-accentGreen bg-clip-text text-transparent mb-2">
          3-Condition Faithfulness Gradient
        </h2>
        <p className="text-gray-400 text-sm">
          Compare saved responses under three grounding conditions and strict replay release counts.
        </p>
      </div>

      {/* Tooltip detail overlay (Framer Motion) */}
      <div className="absolute bottom-16 left-1/2 transform -translate-x-1/2 z-10 w-full max-w-lg px-4">
        <AnimatePresence mode="wait">
          {hoveredBar !== null ? (
            <motion.div
              key={hoveredBar}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: 10 }}
              className="glass-panel p-4 rounded-xl text-center shadow-lg border border-border/80"
            >
              <h3 className="text-base font-bold text-white mb-1 flex justify-center items-center gap-2">
                <span style={{ color: GRADIENT_BARS[hoveredBar].color }}>
                  {GRADIENT_BARS[hoveredBar].label}
                </span>
                <span className="text-xs px-2 py-0.5 rounded bg-surface border border-border text-gray-400">
                  Mean claim faithfulness: {GRADIENT_BARS[hoveredBar].height}
                </span>
              </h3>
              
              <div className="flex justify-center gap-4 text-xs font-semibold my-2">
                <span className="text-accentGreen">{GRADIENT_BARS[hoveredBar].release}</span>
                <span className="text-gray-500">|</span>
                <span className="text-accentRed">{GRADIENT_BARS[hoveredBar].escalate}</span>
              </div>
              
              <p className="text-gray-300 text-xs leading-relaxed mt-1">
                {GRADIENT_BARS[hoveredBar].desc}
              </p>
            </motion.div>
          ) : (
            <motion.div
              key="default"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="text-center py-4 pointer-events-none"
            >
              <span className="text-gray-400 text-xs uppercase tracking-widest bg-surface/50 border border-border/50 px-4 py-2 rounded-full">
                100 saved responses per condition / strict-v1 offline replay
              </span>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  )
}

import React, { useState, useRef } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { OrbitControls, Stars, Float, Line, Html } from '@react-three/drei'
import { PIPELINE_NODES } from '../data/content'
import { X, ChevronRight, Zap } from 'lucide-react'
import { motion, AnimatePresence } from 'framer-motion'

function PipelineNode({ node, position, isSelected, onClick }) {
  const meshRef = useRef()
  const [hovered, setHovered] = useState(false)

  useFrame((state) => {
    if (meshRef.current) {
      meshRef.current.rotation.y += 0.01
      meshRef.current.rotation.x = Math.sin(state.clock.getElapsedTime() * 2) * 0.1
    }
  })

  return (
    <group position={position}>
      {/* Node Sphere */}
      <mesh
        ref={meshRef}
        onClick={onClick}
        onPointerOver={() => setHovered(true)}
        onPointerOut={() => setHovered(false)}
        className="cursor-pointer"
      >
        <sphereGeometry args={[0.35, 32, 32]} />
        <meshStandardMaterial
          color={node.color}
          emissive={node.color}
          emissiveIntensity={hovered || isSelected ? 1.5 : 0.4}
          roughness={0.2}
          metalness={0.8}
        />
      </mesh>

      {/* Floating Label in 3D */}
      <Html distanceFactor={6} position={[0, 0.6, 0]} center>
        <div 
          onClick={onClick}
          className={`px-3 py-1 rounded-md border text-xs font-semibold whitespace-nowrap cursor-pointer transition-all duration-300 ${
            isSelected 
              ? 'bg-surface text-white border-white scale-110 shadow-lg' 
              : 'bg-bg/90 text-gray-300 border-border/80 hover:text-white hover:border-gray-400'
          }`}
          style={{ borderColor: isSelected ? node.color : undefined }}
        >
          {node.name}
        </div>
      </Html>
    </group>
  )
}

function DataParticles({ points }) {
  const particleRef = useRef()

  useFrame((state) => {
    if (particleRef.current) {
      const t = (state.clock.getElapsedTime() * 0.4) % 1.0
      
      // Determine which segment the particle is on
      const numSegments = points.length - 1
      const segmentIdx = Math.floor(t * numSegments)
      const segmentT = (t * numSegments) - segmentIdx
      
      const startPoint = points[segmentIdx]
      const endPoint = points[segmentIdx + 1]
      
      if (startPoint && endPoint) {
        particleRef.current.position.x = startPoint[0] + (endPoint[0] - startPoint[0]) * segmentT
        particleRef.current.position.y = startPoint[1] + (endPoint[1] - startPoint[1]) * segmentT
        particleRef.current.position.z = startPoint[2] + (endPoint[2] - startPoint[2]) * segmentT
      }
    }
  })

  return (
    <mesh ref={particleRef}>
      <sphereGeometry args={[0.08, 16, 16]} />
      <meshBasicMaterial color="#ffffff" toneMapped={false} />
    </mesh>
  )
}

export default function PipelineScene() {
  const [selectedNode, setSelectedNode] = useState(null)
  
  const nodePositions = [
    [-3, 0, 0],     // Classify
    [-1, 0.6, 0],   // Ground
    [1, -0.6, 0],   // Explain
    [3, 0, 0]       // Verify
  ]

  const points = nodePositions

  return (
    <div id="pipeline" className="relative w-full h-screen bg-[#0d1117] flex items-center justify-center overflow-hidden border-b border-[#30363d]/60">
      
      {/* 3D Scene */}
      <div className="absolute inset-0 z-0">
        <Canvas camera={{ position: [0, 0, 5], fov: 50 }}>
          <ambientLight intensity={0.3} />
          <pointLight position={[10, 10, 10]} intensity={1.5} color="#58a6ff" />
          <pointLight position={[-10, -10, -10]} intensity={0.5} color="#bc8cff" />
          
          <Stars radius={100} depth={50} count={2000} factor={4} saturation={0.5} fade speed={1} />
          
          {/* Animated Connecting Lines */}
          <Line
            points={points}
            color="#30363d"
            lineWidth={2}
          />
          <Line
            points={points}
            color="#58a6ff"
            lineWidth={0.5}
            opacity={0.3}
            transparent
          />

          {/* Animated Flowing Particles */}
          <DataParticles points={points} />
          <DataParticles points={[[...points[0]], [...points[1]], [...points[2]], [...points[3]]]} />
          
          {/* Nodes */}
          {PIPELINE_NODES.map((node, idx) => (
            <PipelineNode
              key={node.id}
              node={node}
              position={nodePositions[idx]}
              isSelected={selectedNode?.id === node.id}
              onClick={() => setSelectedNode(node)}
            />
          ))}
          
          <OrbitControls enableZoom={false} maxPolarAngle={Math.PI / 2} minPolarAngle={Math.PI / 2} />
        </Canvas>
      </div>

      {/* Static Overlay Titles */}
      <div className="absolute top-20 left-1/2 transform -translate-x-1/2 text-center z-10 pointer-events-none">
        <h2 className="text-3xl font-bold bg-gradient-to-r from-accentBlue to-accentGreen bg-clip-text text-transparent mb-2">
          Pipeline Architecture
        </h2>
        <p className="text-gray-400 text-sm max-w-md">
          Click any 3D node to inspect the steps of the fault explain-verify-gate pipeline
        </p>
      </div>

      {/* Side Detail Panel (Framer Motion) */}
      <AnimatePresence>
        {selectedNode && (
          <motion.div
            initial={{ x: '100%' }}
            animate={{ x: 0 }}
            exit={{ x: '100%' }}
            transition={{ type: 'spring', damping: 20 }}
            className="absolute right-0 top-0 h-full w-full sm:w-[420px] z-20 glass-panel p-6 pt-24 flex flex-col justify-between shadow-2xl"
          >
            <div>
              <div className="flex items-center justify-between border-b border-border/80 pb-4 mb-6">
                <div>
                  <h3 className="text-xl font-bold text-white flex items-center gap-2">
                    <span style={{ color: selectedNode.color }}>{selectedNode.name}</span>
                  </h3>
                  <p className="text-xs text-gray-400 mt-1 uppercase tracking-widest">{selectedNode.sub}</p>
                </div>
                <button
                  onClick={() => setSelectedNode(null)}
                  className="p-1 rounded-full hover:bg-border/50 text-gray-400 hover:text-white transition-colors"
                >
                  <X size={20} />
                </button>
              </div>

              <div className="space-y-4">
                <p className="text-gray-300 text-sm leading-relaxed">
                  {selectedNode.desc}
                </p>
                
                {/* Visual Pipeline flow indicator */}
                <div className="rounded-lg bg-surface/50 border border-border/50 p-4 mt-6">
                  <h4 className="text-xs font-semibold text-gray-400 uppercase mb-3 flex items-center gap-1.5">
                    <Zap size={12} className="text-accentYellow" /> Data Flow Sequence
                  </h4>
                  <div className="flex items-center gap-1 text-xs text-gray-400">
                    <span className={selectedNode.id === 'classify' ? 'text-accentBlue font-bold' : ''}>Classify</span>
                    <ChevronRight size={12} />
                    <span className={selectedNode.id === 'ground' ? 'text-accentPurple font-bold' : ''}>Ground</span>
                    <ChevronRight size={12} />
                    <span className={selectedNode.id === 'explain' ? 'text-accentYellow font-bold' : ''}>Explain</span>
                    <ChevronRight size={12} />
                    <span className={selectedNode.id === 'verify' ? 'text-accentGreen font-bold' : ''}>Verify</span>
                  </div>
                </div>
              </div>
            </div>

            <div className="border-t border-border/50 pt-4 flex justify-between items-center text-xs text-gray-400">
              <span>Status: Active</span>
              <button 
                onClick={() => setSelectedNode(null)}
                className="text-accentBlue hover:underline"
              >
                Close details
              </button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

import React, { useRef } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { Stars, Float, Text, OrbitControls } from '@react-three/drei'
import { STATS } from '../data/content'

function FloatingTitle() {
  const meshRef = useRef()
  
  useFrame((state) => {
    if (meshRef.current) {
      meshRef.current.rotation.y = Math.sin(state.clock.getElapsedTime() * 0.5) * 0.2
      meshRef.current.rotation.x = Math.cos(state.clock.getElapsedTime() * 0.5) * 0.1
    }
  })

  return (
    <group ref={meshRef}>
      {/* 3D-like metallic text using drei's Text with premium material */}
      <Text
        font="https://fonts.gstatic.com/s/outfit/v11/0oWk0yFbf1dQZ5cTaFSf124.woff"
        fontSize={0.8}
        maxWidth={8}
        textAlign="center"
        position={[0, 0.5, 0]}
        color="#ffffff"
        anchorX="center"
        anchorY="middle"
      >
        VERIFIED FAULT DIAGNOSIS
        <meshStandardMaterial
          color="#ffffff"
          metalness={0.9}
          roughness={0.1}
          emissive="#1a2b4c"
          emissiveIntensity={0.2}
        />
      </Text>

      {/* Subtitle */}
      <Text
        font="https://fonts.gstatic.com/s/inter/v18/UcCO3FwrK3iLTeHuS_fvQtMwCp50KnMw2bo.woff"
        fontSize={0.25}
        maxWidth={6}
        textAlign="center"
        position={[0, -0.3, 0]}
        color="#8b949e"
        anchorX="center"
        anchorY="middle"
      >
        Can an LLM explain a machine fault — and prove it's telling the truth?
      </Text>
    </group>
  )
}

function FloatingBadges() {
  const badges = [
    { text: "95.56% Accuracy", pos: [-2.5, 1.2, 0.5], color: "#58a6ff" },
    { text: "96% Faithfulness", pos: [2.5, 1.2, 0.5], color: "#3fb950" },
    { text: "17 Tests", pos: [-2.5, -0.8, 0.5], color: "#bc8cff" },
    { text: "Llama 3.1 70B", pos: [2.5, -0.8, 0.5], color: "#d29922" }
  ]

  return (
    <>
      {badges.map((badge, idx) => (
        <Float key={idx} speed={2} rotationIntensity={0.5} floatIntensity={1}>
          <Text
            font="https://fonts.gstatic.com/s/outfit/v11/0oWk0yFbf1dQZ5cTaFSf124.woff"
            fontSize={0.16}
            position={badge.pos}
            color="#ffffff"
            backgroundColor={badge.color + "33"}
            padding={[0.1, 0.2]}
            borderRadius={0.05}
            strokeWidth="2%"
            strokeColor={badge.color}
          >
            {badge.text}
          </Text>
        </Float>
      ))}
    </>
  )
}

export default function HeroScene() {
  return (
    <div id="overview" className="relative w-full h-screen bg-[#0d1117] overflow-hidden flex flex-col justify-between">
      {/* 3D Canvas Background */}
      <div className="absolute inset-0 z-0">
        <Canvas camera={{ position: [0, 0, 5.5], fov: 60 }}>
          <ambientLight intensity={0.5} />
          <pointLight position={[10, 10, 10]} intensity={1.5} color="#58a6ff" />
          <pointLight position={[-10, -10, -10]} intensity={0.8} color="#bc8cff" />
          
          <Stars radius={100} depth={50} count={5000} factor={4} saturation={0.5} fade speed={1.5} />
          
          <Float speed={1.5} rotationIntensity={0.2} floatIntensity={0.5}>
            <FloatingTitle />
          </Float>
          
          <FloatingBadges />
          
          <OrbitControls enableZoom={false} autoRotate autoRotateSpeed={0.3} maxPolarAngle={Math.PI / 2} minPolarAngle={Math.PI / 2} />
        </Canvas>
      </div>

      {/* Hero Content Overlay (Header space and scroll indicator) */}
      <div className="relative z-10 w-full max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-24 pb-8 flex-1 flex flex-col justify-between pointer-events-none">
        <div /> {/* Spacer */}
        
        {/* Scroll Indicator */}
        <div className="w-full flex flex-col items-center justify-center animate-bounce">
          <span className="text-gray-400 text-xs font-semibold uppercase tracking-widest mb-2">Scroll to explore</span>
          <div className="w-[18px] h-[28px] border-2 border-gray-500 rounded-full flex justify-center pt-1.5">
            <div className="w-[4px] h-[6px] bg-accentBlue rounded-full" />
          </div>
        </div>
      </div>
    </div>
  )
}

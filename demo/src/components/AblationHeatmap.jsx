import React, { useState, useRef } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { OrbitControls, Stars, Text, Float } from '@react-three/drei'
import { HEATMAP } from '../data/content'
import { motion, AnimatePresence } from 'framer-motion'
import { Star } from 'lucide-react'

// Color interpolation based on F1 Score
function getHeatmapColor(val) {
  // Low value -> dark gray/green, High value -> bright emerald green
  if (val < 0.4) return '#21262d' // Low SVM/Time
  if (val < 0.6) return '#1b4f30' // Moderate
  if (val < 0.9) return '#106e33' // High
  return '#3fb950' // Very High (0.9+)
}

function GridBlock({ val, rowIdx, colIdx, hoveredCell, setHoveredCell }) {
  const meshRef = useRef()
  const targetHeight = val * 2.0
  const isHovered = hoveredCell && hoveredCell.row === rowIdx && hoveredCell.col === colIdx
  
  useFrame((state) => {
    if (meshRef.current) {
      // Animate growth from flat to full height
      const currentScaleY = meshRef.current.scale.y
      const diff = 1.0 - currentScaleY
      if (diff > 0.001) {
        meshRef.current.scale.y += diff * 0.05
        meshRef.current.position.y = (meshRef.current.scale.y * targetHeight) / 2
      }
    }
  })

  // Position: X is column, Z is row
  const posX = (colIdx - 1.5) * 1.3
  const posZ = (rowIdx - 1.5) * 1.3

  return (
    <group position={[posX, 0, posZ]}>
      {/* 3D block */}
      <mesh
        ref={meshRef}
        scale={[1, 0, 1]}
        position={[0, 0, 0]}
        onPointerOver={(e) => {
          e.stopPropagation()
          setHoveredCell({ row: rowIdx, col: colIdx, val })
        }}
        onPointerOut={() => setHoveredCell(null)}
        className="cursor-pointer"
      >
        <boxGeometry args={[0.9, targetHeight, 0.9]} />
        <meshStandardMaterial
          color={getHeatmapColor(val)}
          roughness={0.2}
          metalness={0.7}
          emissive={isHovered ? '#3fb950' : '#000000'}
          emissiveIntensity={isHovered ? 0.8 : 0}
        />
      </mesh>

      {/* Floating Gold Star for WPT × LogReg cell (row = 2, col = 3, value = 0.998) */}
      {rowIdx === 2 && colIdx === 3 && (
        <Float speed={3} rotationIntensity={1} floatIntensity={0.5}>
          <mesh position={[0, targetHeight + 0.3, 0]}>
            <octahedronGeometry args={[0.15, 0]} />
            <meshStandardMaterial color="#d29922" emissive="#d29922" emissiveIntensity={1.5} />
          </mesh>
        </Float>
      )}

      {/* Value label on top of block */}
      <Text
        font="https://fonts.gstatic.com/s/outfit/v11/0oWk0yFbf1dQZ5cTaFSf124.woff"
        fontSize={0.16}
        position={[0, targetHeight + 0.05, 0]}
        color="#ffffff"
        anchorX="center"
        anchorY="bottom"
        rotation={[-Math.PI / 6, 0, 0]}
      >
        {val.toFixed(3)}
      </Text>
    </group>
  )
}

export default function AblationHeatmap() {
  const [hoveredCell, setHoveredCell] = useState(null)

  return (
    <div id="ablation" className="relative w-full h-screen bg-[#0d1117] flex items-center justify-center overflow-hidden border-b border-[#30363d]/60">
      
      {/* 3D Canvas */}
      <div className="absolute inset-0 z-0">
        <Canvas camera={{ position: [3.5, 3.5, 5], fov: 50 }}>
          <ambientLight intensity={0.4} />
          <pointLight position={[10, 10, 10]} intensity={1.5} color="#58a6ff" />
          <pointLight position={[-10, -10, -10]} intensity={0.5} color="#bc8cff" />
          
          <Stars radius={100} depth={50} count={1000} factor={4} saturation={0.5} fade speed={1} />
          
          {/* Ground Grid platform */}
          <mesh position={[0, -0.05, 0]} rotation={[-Math.PI / 2, 0, 0]}>
            <planeGeometry args={[6, 6]} />
            <meshStandardMaterial color="#161b22" roughness={0.9} />
          </mesh>

          {/* Row and Col Label Texts in 3D space */}
          {HEATMAP.rows.map((row, idx) => (
            <Text
              key={idx}
              font="https://fonts.gstatic.com/s/inter/v18/UcCO3FwrK3iLTeHuS_fvQtMwCp50KnMw2bo.woff"
              fontSize={0.18}
              position={[-2.8, 0, (idx - 1.5) * 1.3]}
              color="#8b949e"
              anchorX="right"
              anchorY="middle"
              rotation={[0, Math.PI / 6, 0]}
            >
              {row}
            </Text>
          ))}

          {HEATMAP.cols.map((col, idx) => (
            <Text
              key={idx}
              font="https://fonts.gstatic.com/s/inter/v18/UcCO3FwrK3iLTeHuS_fvQtMwCp50KnMw2bo.woff"
              fontSize={0.18}
              position={[(idx - 1.5) * 1.3, 0, 2.5]}
              color="#8b949e"
              anchorX="center"
              anchorY="top"
              rotation={[-Math.PI / 2, 0, 0]}
            >
              {col}
            </Text>
          ))}

          {/* Blocks */}
          {HEATMAP.grid.map((rowArr, rowIdx) =>
            rowArr.map((val, colIdx) => (
              <GridBlock
                key={`${rowIdx}-${colIdx}`}
                val={val}
                rowIdx={rowIdx}
                colIdx={colIdx}
                hoveredCell={hoveredCell}
                setHoveredCell={setHoveredCell}
              />
            ))
          )}
          
          <OrbitControls enableZoom={false} maxPolarAngle={Math.PI / 2.2} minPolarAngle={Math.PI / 3} />
        </Canvas>
      </div>

      {/* Header Overlay */}
      <div className="absolute top-20 left-1/2 transform -translate-x-1/2 text-center z-10 pointer-events-none w-full max-w-xl px-4">
        <h2 className="text-3xl font-bold bg-gradient-to-r from-accentPurple to-accentGreen bg-clip-text text-transparent mb-2">
          Feature-Family × Classifier Ablation
        </h2>
        <p className="text-gray-400 text-sm">
          F1-Macro scores for all combinations. The gold star highlights the optimal WPT × Logistic Regression setup.
        </p>
      </div>

      {/* Hover Info Tooltip */}
      <div className="absolute bottom-16 left-1/2 transform -translate-x-1/2 z-10 w-full max-w-lg px-4">
        <AnimatePresence mode="wait">
          {hoveredCell ? (
            <motion.div
              key={`${hoveredCell.row}-${hoveredCell.col}`}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: 10 }}
              className="glass-panel p-4 rounded-xl text-center shadow-lg border border-border/80"
            >
              <h3 className="text-base font-bold text-white mb-1 flex justify-center items-center gap-2">
                <span>{HEATMAP.cols[hoveredCell.col]} × {HEATMAP.rows[hoveredCell.row]}</span>
              </h3>
              <p className="text-accentGreen text-lg font-black mt-1">
                F1 Score: {hoveredCell.val.toFixed(3)}
              </p>
              {hoveredCell.row === 2 && hoveredCell.col === 3 && (
                <div className="flex items-center justify-center gap-1.5 text-xs text-accentYellow mt-1 font-semibold">
                  <Star size={14} fill="#d29922" /> Top Performer (WPT engineering carries the weight!)
                </div>
              )}
            </motion.div>
          ) : (
            <motion.div
              key="default"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="glass-panel p-4 rounded-xl text-center shadow-lg border border-border/60"
            >
              <p className="text-gray-300 text-xs leading-relaxed font-medium">
                👉 <strong>Insight</strong>: Logistic Regression on WPT-only (0.998) beats XGBoost on full features (0.973). Handcrafted feature representation does the heavy lifting.
              </p>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  )
}

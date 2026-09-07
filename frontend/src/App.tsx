import { useRef, useEffect, useState, useMemo } from 'react'
import { Canvas } from '@react-three/fiber'
import { OrbitControls, Stats } from '@react-three/drei'
import * as THREE from 'three'

const COLOR_MAP: Record<number, string> = {
  255: '#444444', // UNSEEN / Out of BEV Bounds (Dark Grey)
  0: '#a3a3a3',   // UNKNOWN (Light Grey)
  9: '#639921',   // Drivable Terrain (Green)
  10: '#639921',
  11: '#639921',
  12: '#639921',
  17: '#639921',
  1: '#d4547d',   // Dynamic Objects (Pink)
  2: '#d4547d',
  3: '#d4547d',
  4: '#d4547d',
  5: '#d4547d',
  6: '#d4547d',
  7: '#d4547d',
  8: '#d4547d',
  13: '#d95930',  // Static Obstacles (Orange)
  14: '#d95930',
  15: '#d95930',
  16: '#d95930',
  18: '#d95930',
  19: '#d95930',
}

function GridCells({ gridData }: { gridData: any }) {
  const meshRef = useRef<THREE.InstancedMesh>(null)
  
  const dummy = useMemo(() => new THREE.Object3D(), [])
  const color = useMemo(() => new THREE.Color(), [])

  useEffect(() => {
    if (!meshRef.current || !gridData) return
    const { positions, labels, sizes } = gridData

    const count = positions.length / 3
    for (let i = 0; i < count; i++) {
      const x = positions[i * 3]
      const y = positions[i * 3 + 1]
      const z = positions[i * 3 + 2]
      const s = sizes[i]
      const label = labels[i]

      dummy.position.set(x, y, z / 2) 
      dummy.scale.set(s, s, Math.max(0.1, z)) 
      dummy.updateMatrix()
      meshRef.current.setMatrixAt(i, dummy.matrix)

      const hex = COLOR_MAP[label] || '#444444'
      color.set(hex)
      meshRef.current.setColorAt(i, color)
    }
    
    meshRef.current.instanceMatrix.needsUpdate = true
    if (meshRef.current.instanceColor) {
      meshRef.current.instanceColor.needsUpdate = true
    }
  }, [gridData, dummy, color])

  if (!gridData) return null

  return (
    <instancedMesh ref={meshRef} args={[undefined, undefined, gridData.positions.length / 3]}>
      <boxGeometry args={[1, 1, 1]} />
      <meshStandardMaterial />
    </instancedMesh>
  )
}

export default function App() {
  const [frameId, setFrameId] = useState(0)
  const [data, setData] = useState<any>(null)
  const [loading, setLoading] = useState(false)
  const [isPlaying, setIsPlaying] = useState(false)
  const timerRef = useRef<any>(null)

  const fetchFrame = async (id: number) => {
    setLoading(true)
    try {
      const res = await fetch(`http://localhost:8000/api/frame/${id}`)
      if (res.ok) {
        const json = await res.json()
        setData(json)
      }
    } catch (e) {
      console.error(e)
    }
    setLoading(false)
  }

  // Auto-fetch when frameId changes
  useEffect(() => {
    fetchFrame(frameId)
  }, [frameId])

  // Playback Loop for Simulated Real-Time Data
  useEffect(() => {
    if (isPlaying) {
      timerRef.current = setInterval(() => {
        setFrameId(prev => prev + 1)
      }, 300) // ~3 FPS simulated speed
    } else {
      clearInterval(timerRef.current)
    }
    return () => clearInterval(timerRef.current)
  }, [isPlaying])

  const warning = data?.metrics?.collision_warning;

  return (
    <div style={{ width: '100vw', height: '100vh', display: 'flex', fontFamily: 'sans-serif' }}>
      
      {/* Sidebar UI */}
      <div style={{ width: '320px', padding: '20px', background: '#1e1e1e', color: 'white', position: 'relative' }}>
        <h2>Autonomous LiDAR System</h2>
        
        {/* COLLISION WARNING OVERLAY */}
        {warning && (
          <div style={{
            background: 'rgba(220, 38, 38, 0.9)', 
            padding: '15px', 
            borderRadius: '8px', 
            marginBottom: '20px',
            animation: 'pulse 1s infinite',
            textAlign: 'center',
            fontWeight: 'bold',
            border: '2px solid red'
          }}>
            ⚠️ CRITICAL PROXIMITY WARNING ⚠️
          </div>
        )}
        
        <div style={{ marginTop: '20px', display: 'flex', gap: '10px', alignItems: 'center' }}>
          <button onClick={() => setFrameId(prev => Math.max(0, prev - 1))} disabled={isPlaying || frameId <= 0}>Prev</button>
          <button onClick={() => setFrameId(prev => prev + 1)} disabled={isPlaying}>Next</button>
          <button 
            onClick={() => setIsPlaying(!isPlaying)}
            style={{ background: isPlaying ? '#dc2626' : '#2563eb', color: 'white', border: 'none', padding: '8px 16px', borderRadius: '4px', cursor: 'pointer', fontWeight: 'bold' }}
          >
            {isPlaying ? '⏹ Stop' : '▶️ Auto-Drive'}
          </button>
        </div>
        <p style={{marginTop: '10px'}}><strong>Sequence Frame:</strong> {frameId}</p>
        
        {data && data.metrics && (
          <div style={{ marginTop: '30px', background: '#2d2d2d', padding: '15px', borderRadius: '8px' }}>
            <h3 style={{ margin: '0 0 10px 0' }}>Live Telemetry</h3>
            <p><strong>Raw Points:</strong> {data.metrics.raw_points_count.toLocaleString()}</p>
            <p><strong>Foveated Cells:</strong> {data.metrics.foveated_cells_count.toLocaleString()}</p>
            
            <div style={{ marginTop: '15px', padding: '10px', background: '#111', borderRadius: '4px' }}>
              <p style={{ color: '#4ade80', margin: 0, fontWeight: 'bold' }}>
                Memory Saved: {data.metrics.memory_reduction_factor}x
              </p>
            </div>
          </div>
        )}

        {/* CSS for the flashing pulse animation */}
        <style>
          {`
            @keyframes pulse {
              0% { opacity: 1; }
              50% { opacity: 0.5; }
              100% { opacity: 1; }
            }
          `}
        </style>
      </div>

      {/* 3D WebGL Canvas */}
      <div style={{ flex: 1, background: '#111', position: 'relative' }}>
        {loading && !isPlaying && <div style={{ position: 'absolute', top: 20, left: 20, color: 'white', zIndex: 10 }}>Processing...</div>}
        
        <Canvas camera={{ position: [0, -40, 30], up: [0, 0, 1], fov: 45 }}>
          <color attach="background" args={['#111']} />
          <ambientLight intensity={0.5} />
          <directionalLight position={[10, 10, 20]} intensity={1} />
          
          <GridCells gridData={data?.grid} />
          
          <OrbitControls makeDefault />
          <Stats />
        </Canvas>
      </div>
    </div>
  )
}

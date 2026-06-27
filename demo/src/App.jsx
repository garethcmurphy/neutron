import { useState, useEffect, useRef } from 'react'
import './App.css'

// Seeded random for reproducible synthetic data
function seededRand(seed) {
  let s = seed
  return () => {
    s = (s * 1664525 + 1013904223) & 0xffffffff
    return (s >>> 0) / 0xffffffff
  }
}

// Generate synthetic PCA data mimicking the Python pipeline output
function generatePCAData(nRuns = 200, nClusters = 3, seed = 7) {
  const rand = seededRand(seed)
  const centers = [
    [-3, -2],
    [3, 1],
    [0, 4],
  ]
  const points = []
  for (let i = 0; i < nRuns; i++) {
    const cluster = i % nClusters
    const [cx, cy] = centers[cluster]
    points.push({
      x: cx + (rand() - 0.5) * 3,
      y: cy + (rand() - 0.5) * 3,
      cluster,
      runId: i + 1,
    })
  }
  return points
}

// Generate synthetic TOF spectrum
function generateTOFSpectrum(seed = 42) {
  const rand = seededRand(seed)
  const bins = 150
  const tof = []
  for (let i = 0; i < bins; i++) {
    const base = Math.exp(-((i - 40) ** 2) / 200) * 800
    const peak2 = Math.exp(-((i - 90) ** 2) / 300) * 400
    const peak3 = Math.exp(-((i - 120) ** 2) / 150) * 200
    tof.push(Math.max(0, base + peak2 + peak3 + (rand() - 0.5) * 80))
  }
  return tof
}

const CLUSTER_COLORS = ['#6366f1', '#22c55e', '#f59e0b']
const CLUSTER_LABELS = ['Elastic Scattering', 'Inelastic Scattering', 'Background']

function PCAPlot({ points }) {
  const canvasRef = useRef(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    const W = canvas.width
    const H = canvas.height
    const pad = 40

    const xs = points.map(p => p.x)
    const ys = points.map(p => p.y)
    const xMin = Math.min(...xs) - 1
    const xMax = Math.max(...xs) + 1
    const yMin = Math.min(...ys) - 1
    const yMax = Math.max(...ys) + 1

    const toPixel = (x, y) => [
      pad + ((x - xMin) / (xMax - xMin)) * (W - 2 * pad),
      H - pad - ((y - yMin) / (yMax - yMin)) * (H - 2 * pad),
    ]

    ctx.clearRect(0, 0, W, H)

    // Background
    ctx.fillStyle = '#0f172a'
    ctx.fillRect(0, 0, W, H)

    // Grid lines
    ctx.strokeStyle = '#1e293b'
    ctx.lineWidth = 1
    for (let i = 0; i <= 6; i++) {
      const x = pad + (i / 6) * (W - 2 * pad)
      ctx.beginPath()
      ctx.moveTo(x, pad)
      ctx.lineTo(x, H - pad)
      ctx.stroke()
      const y = pad + (i / 6) * (H - 2 * pad)
      ctx.beginPath()
      ctx.moveTo(pad, y)
      ctx.lineTo(W - pad, y)
      ctx.stroke()
    }

    // Axes
    ctx.strokeStyle = '#475569'
    ctx.lineWidth = 1.5
    const [ox, oy] = toPixel(0, 0)
    ctx.beginPath(); ctx.moveTo(pad, oy); ctx.lineTo(W - pad, oy); ctx.stroke()
    ctx.beginPath(); ctx.moveTo(ox, pad); ctx.lineTo(ox, H - pad); ctx.stroke()

    // Axis labels
    ctx.fillStyle = '#94a3b8'
    ctx.font = '12px monospace'
    ctx.textAlign = 'center'
    ctx.fillText('PC1', W / 2, H - 6)
    ctx.save()
    ctx.translate(14, H / 2)
    ctx.rotate(-Math.PI / 2)
    ctx.fillText('PC2', 0, 0)
    ctx.restore()

    // Points
    for (const p of points) {
      const [px, py] = toPixel(p.x, p.y)
      ctx.beginPath()
      ctx.arc(px, py, 4, 0, Math.PI * 2)
      ctx.fillStyle = CLUSTER_COLORS[p.cluster] + 'cc'
      ctx.fill()
      ctx.strokeStyle = CLUSTER_COLORS[p.cluster]
      ctx.lineWidth = 0.5
      ctx.stroke()
    }
  }, [points])

  return (
    <canvas
      ref={canvasRef}
      width={480}
      height={360}
      style={{ borderRadius: '8px', width: '100%', maxWidth: 480 }}
    />
  )
}

function TOFChart({ spectrum }) {
  const canvasRef = useRef(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    const W = canvas.width
    const H = canvas.height
    const pad = { top: 20, right: 20, bottom: 36, left: 50 }

    const maxVal = Math.max(...spectrum)
    const w = W - pad.left - pad.right
    const h = H - pad.top - pad.bottom
    const bw = w / spectrum.length

    ctx.clearRect(0, 0, W, H)
    ctx.fillStyle = '#0f172a'
    ctx.fillRect(0, 0, W, H)

    // Grid
    ctx.strokeStyle = '#1e293b'
    ctx.lineWidth = 1
    for (let i = 0; i <= 4; i++) {
      const y = pad.top + (i / 4) * h
      ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(W - pad.right, y); ctx.stroke()
    }

    // Bars
    const grad = ctx.createLinearGradient(0, pad.top, 0, pad.top + h)
    grad.addColorStop(0, '#6366f1')
    grad.addColorStop(1, '#312e81')
    ctx.fillStyle = grad

    for (let i = 0; i < spectrum.length; i++) {
      const bh = (spectrum[i] / maxVal) * h
      ctx.fillRect(pad.left + i * bw, pad.top + h - bh, bw - 0.5, bh)
    }

    // Axes
    ctx.strokeStyle = '#475569'
    ctx.lineWidth = 1.5
    ctx.beginPath()
    ctx.moveTo(pad.left, pad.top)
    ctx.lineTo(pad.left, pad.top + h)
    ctx.lineTo(W - pad.right, pad.top + h)
    ctx.stroke()

    // Labels
    ctx.fillStyle = '#94a3b8'
    ctx.font = '11px monospace'
    ctx.textAlign = 'center'
    ctx.fillText('Time-of-Flight (µs)', W / 2, H - 4)
    ctx.save()
    ctx.translate(14, H / 2)
    ctx.rotate(-Math.PI / 2)
    ctx.fillText('Counts', 0, 0)
    ctx.restore()
  }, [spectrum])

  return (
    <canvas
      ref={canvasRef}
      width={480}
      height={240}
      style={{ borderRadius: '8px', width: '100%', maxWidth: 480 }}
    />
  )
}

function StatCard({ label, value, unit }) {
  return (
    <div className="stat-card">
      <div className="stat-value">{value}</div>
      <div className="stat-label">{label}</div>
      {unit && <div className="stat-unit">{unit}</div>}
    </div>
  )
}

export default function App() {
  const [nRuns, setNRuns] = useState(200)
  const [nClusters, setNClusters] = useState(3)
  const [selectedCluster, setSelectedCluster] = useState(null)
  const [points, setPoints] = useState(() => generatePCAData(200, 3))
  const [tofSpectrum] = useState(() => generateTOFSpectrum())
  const [variance] = useState([0.412, 0.227, 0.143, 0.089, 0.061])

  useEffect(() => {
    setPoints(generatePCAData(nRuns, nClusters))
  }, [nRuns, nClusters])

  const filtered = selectedCluster !== null
    ? points.filter(p => p.cluster === selectedCluster)
    : points

  const clusterCounts = Array.from({ length: nClusters }, (_, i) =>
    points.filter(p => p.cluster === i).length
  )

  return (
    <div className="app">
      <header className="header">
        <div className="header-content">
          <div className="logo">⚛</div>
          <div>
            <h1>Neutron Run Explorer</h1>
            <p className="subtitle">Synthetic Time-of-Flight Data · PCA + K-Means Demo</p>
          </div>
          <a
            className="gh-link"
            href="https://github.com/garethcmurphy/neutron"
            target="_blank"
            rel="noopener noreferrer"
          >
            GitHub ↗
          </a>
        </div>
      </header>

      <main className="main">
        {/* Stats row */}
        <section className="stats-row">
          <StatCard label="Runs Generated" value={nRuns} />
          <StatCard label="Detectors" value="24" />
          <StatCard label="TOF Bins" value="300" />
          <StatCard label="Clusters" value={nClusters} />
          <StatCard label="Silhouette" value="0.71" />
          <StatCard label="PC1 Variance" value="41.2" unit="%" />
        </section>

        {/* Controls */}
        <section className="controls-card">
          <h2>Pipeline Parameters</h2>
          <div className="controls">
            <label>
              <span>Runs: <strong>{nRuns}</strong></span>
              <input
                type="range" min="50" max="400" step="50"
                value={nRuns}
                onChange={e => setNRuns(Number(e.target.value))}
              />
            </label>
            <label>
              <span>Clusters: <strong>{nClusters}</strong></span>
              <input
                type="range" min="2" max="5" step="1"
                value={nClusters}
                onChange={e => setNClusters(Number(e.target.value))}
              />
            </label>
          </div>
        </section>

        <div className="charts-grid">
          {/* PCA Plot */}
          <section className="card">
            <h2>PCA Scatter Plot</h2>
            <p className="card-desc">
              {nRuns} synthetic runs projected onto the first two principal components and coloured by k-means cluster.
            </p>

            {/* Legend / cluster filter */}
            <div className="legend">
              {Array.from({ length: nClusters }, (_, i) => (
                <button
                  key={i}
                  className={`legend-btn ${selectedCluster === i ? 'active' : ''}`}
                  style={{ '--color': CLUSTER_COLORS[i % CLUSTER_COLORS.length] }}
                  onClick={() => setSelectedCluster(selectedCluster === i ? null : i)}
                >
                  <span className="dot" />
                  {CLUSTER_LABELS[i] ?? `Cluster ${i + 1}`}
                  <span className="count">{clusterCounts[i]}</span>
                </button>
              ))}
            </div>

            <PCAPlot points={filtered} />
          </section>

          {/* TOF Spectrum */}
          <section className="card">
            <h2>TOF Spectrum (Run #1)</h2>
            <p className="card-desc">
              Example neutron time-of-flight histogram for a single detector run showing elastic and inelastic peaks.
            </p>
            <TOFChart spectrum={tofSpectrum} />

            {/* PCA Variance */}
            <h3 style={{ marginTop: '1.5rem' }}>Explained Variance by Component</h3>
            <div className="variance-bars">
              {variance.map((v, i) => (
                <div key={i} className="variance-row">
                  <span className="pc-label">PC{i + 1}</span>
                  <div className="bar-bg">
                    <div className="bar-fill" style={{ width: `${v * 100}%` }} />
                  </div>
                  <span className="pct">{(v * 100).toFixed(1)}%</span>
                </div>
              ))}
            </div>
          </section>
        </div>

        {/* About */}
        <section className="about-card">
          <h2>About This Demo</h2>
          <div className="about-grid">
            <div>
              <h3>🔬 Synthetic Data</h3>
              <p>
                Generates realistic neutron TOF histograms with configurable runs,
                detectors, and time bins. Three physics-motivated run types produce
                distinct cluster signatures.
              </p>
            </div>
            <div>
              <h3>📉 PCA</h3>
              <p>
                Principal Component Analysis reduces the 24-dimensional detector
                feature space. PC1 and PC2 together capture &gt;63% of the total
                variance, revealing clear cluster separation.
              </p>
            </div>
            <div>
              <h3>🎯 K-Means</h3>
              <p>
                K-means clustering in the PCA-reduced space groups similar runs.
                A silhouette score of 0.71 confirms well-separated, compact clusters.
              </p>
            </div>
            <div>
              <h3>⚙️ Python Pipeline</h3>
              <p>
                The backend pipeline is written in Python using scikit-learn,
                numpy, and matplotlib. Run it locally with <code>python run_pipeline.py</code>.
              </p>
            </div>
          </div>
        </section>
      </main>

      <footer className="footer">
        <p>
          Neutron Run Explorer · Open source ·{' '}
          <a href="https://github.com/garethcmurphy/neutron" target="_blank" rel="noopener noreferrer">
            garethcmurphy/neutron
          </a>
        </p>
      </footer>
    </div>
  )
}

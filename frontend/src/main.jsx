import React, { useEffect, useMemo, useRef, useState } from 'react'
import { createRoot } from 'react-dom/client'
import {
  Activity, AudioLines, Check, ChevronLeft, Clock3, FileAudio,
  Gauge, Mic, Pause, Play, RotateCcw, ShieldCheck, Upload, X,
  CircleAlert
} from 'lucide-react'
import './styles.css'

const API_URL = (import.meta.env.VITE_API_URL || '').replace(/\/$/, '')

const PIPELINE = [
  ['Input received', 'Preparing the audio signal'],
  ['Pre-processing', 'Normalizing and standardizing audio'],
  ['Feature analysis', 'Inspecting acoustic patterns'],
  ['Voice classification', 'Running the detection model'],
  ['Analysis complete', 'Preparing the confidence report']
]

function fmtTime(sec) {
  const s = Math.max(0, Math.floor(sec))
  return `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`
}

function formatBytes(bytes = 0) {
  if (!bytes) return '—'
  const units = ['B', 'KB', 'MB', 'GB']
  let n = bytes, i = 0
  while (n >= 1024 && i < units.length - 1) { n /= 1024; i++ }
  return `${n.toFixed(i ? 1 : 0)} ${units[i]}`
}

function normalizeResult(data) {
  const prediction = String(data?.prediction || data?.label || data?.result || '').toUpperCase()
  const ai = Number(data?.ai_probability ?? data?.fake_probability ?? data?.synthetic_probability)
  const human = Number(data?.human_probability ?? data?.real_probability ?? data?.bona_fide_probability)
  let aiP = Number.isFinite(ai) ? ai : NaN
  let humanP = Number.isFinite(human) ? human : NaN
  if (!Number.isFinite(aiP) && Number.isFinite(humanP)) aiP = 1 - humanP
  if (!Number.isFinite(humanP) && Number.isFinite(aiP)) humanP = 1 - aiP

  const aiPred = /AI|FAKE|SYNTH|SPOOF/.test(prediction)
  const isAI = Number.isFinite(aiP) ? aiP >= 0.5 : aiPred
  const confidence = Number(data?.confidence)
  const conf = Number.isFinite(confidence) ? confidence : (isAI ? aiP : humanP)
  if (!Number.isFinite(aiP) && Number.isFinite(conf)) aiP = isAI ? conf : 1 - conf
  if (!Number.isFinite(humanP) && Number.isFinite(conf)) humanP = isAI ? 1 - conf : conf

  return {
    isAI,
    aiProbability: Number.isFinite(aiP) ? aiP : null,
    humanProbability: Number.isFinite(humanP) ? humanP : null,
    confidence: Number.isFinite(conf) ? conf : null,
    model: data?.model_version || data?.model || 'SVM baseline',
    processing: data?.processing_time_ms ?? data?.processing_time ?? null
  }
}

function Waveform({ active = false }) {
  return (
    <div className={`waveform ${active ? 'active' : ''}`} aria-hidden="true">
      {Array.from({ length: 42 }).map((_, i) => <i key={i} style={{ '--i': i }} />)}
    </div>
  )
}

function App() {
  const [file, setFile] = useState(null)
  const [sourceUrl, setSourceUrl] = useState('')
  const [recording, setRecording] = useState(false)
  const [seconds, setSeconds] = useState(0)
  const [busy, setBusy] = useState(false)
  const [stage, setStage] = useState(0)
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [dragging, setDragging] = useState(false)
  const [playing, setPlaying] = useState(false)
  const inputRef = useRef(null)
  const mediaRecorder = useRef(null)
  const chunks = useRef([])
  const timerRef = useRef(null)
  const audioRef = useRef(null)

  useEffect(() => () => {
    if (sourceUrl) URL.revokeObjectURL(sourceUrl)
    clearInterval(timerRef.current)
  }, [sourceUrl])

  useEffect(() => {
    if (!busy) return
    setStage(0)
    const timings = [550, 1350, 2350, 3650]
    const ids = timings.map((ms, i) => setTimeout(() => setStage(i + 1), ms))
    return () => ids.forEach(clearTimeout)
  }, [busy])

  const fileMeta = useMemo(() => file ? {
    name: file.name,
    size: formatBytes(file.size),
    type: file.type || 'Audio file'
  } : null, [file])

  function chooseFile(f) {
    if (!f) return
    setError('')
    setResult(null)
    if (sourceUrl) URL.revokeObjectURL(sourceUrl)
    setSourceUrl(URL.createObjectURL(f))
    setFile(f)
  }

  function onFileInput(e) { chooseFile(e.target.files?.[0]) }

  async function startRecording() {
    setError('')
    setResult(null)
    if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === 'undefined') {
      setError('Microphone recording is not available in this browser.')
      return
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const mimeTypes = ['audio/webm;codecs=opus', 'audio/webm', 'audio/ogg;codecs=opus']
      const mimeType = mimeTypes.find(type => MediaRecorder.isTypeSupported(type))
      const rec = mimeType ? new MediaRecorder(stream, { mimeType }) : new MediaRecorder(stream)
      chunks.current = []
      rec.ondataavailable = e => e.data.size && chunks.current.push(e.data)
      rec.onstop = () => {
        stream.getTracks().forEach(t => t.stop())
        const blobType = rec.mimeType || 'audio/webm'
        const blob = new Blob(chunks.current, { type: blobType })
        const extension = blobType.includes('ogg') ? 'ogg' : 'webm'
        chooseFile(new File([blob], `microphone-${Date.now()}.${extension}`, { type: blobType }))
      }
      mediaRecorder.current = rec
      rec.start()
      setRecording(true)
      setSeconds(0)
      timerRef.current = setInterval(() => setSeconds(s => s + 1), 1000)
    } catch (recordingError) {
      console.error('Unable to start microphone recording:', recordingError)
      setError('Microphone permission was denied or unavailable.')
    }
  }

  function stopRecording() {
    mediaRecorder.current?.stop()
    setRecording(false)
    clearInterval(timerRef.current)
  }

  function reset() {
    setBusy(false); setResult(null); setFile(null); setError(''); setStage(0); setSeconds(0); setPlaying(false)
    if (sourceUrl) URL.revokeObjectURL(sourceUrl)
    setSourceUrl('')
    if (inputRef.current) inputRef.current.value = ''
  }

  async function analyze() {
    if (!file) return
    setBusy(true); setResult(null); setError('')
    const body = new FormData()
    body.append('file', file)
    try {
      const started = performance.now()
      const response = await fetch(`${API_URL}/api/analyze`, { method: 'POST', body })
      const raw = await response.text()
      let data
      try { data = JSON.parse(raw) } catch { throw new Error(raw || `Server returned ${response.status}`) }
      if (!response.ok) throw new Error(data.detail || data.error || `Analysis failed (${response.status})`)
      const normalized = normalizeResult(data)
      normalized.clientProcessing = Math.round(performance.now() - started)
      setStage(5)
      setTimeout(() => { setResult(normalized); setBusy(false) }, 500)
    } catch (e) {
      setBusy(false)
      setError(e.message.includes('Failed to fetch')
        ? `Could not connect to the analysis API at ${API_URL}.`
        : e.message)
    }
  }

  function togglePlay() {
    if (!audioRef.current) return
    if (playing) audioRef.current.pause()
    else audioRef.current.play()
    setPlaying(!playing)
  }

  if (result) {
    const human = result.humanProbability
    const ai = result.aiProbability
    const confidence = result.confidence
    return (
      <div className="app">
        <header className="topbar">
          <button className="brand" onClick={reset}><span className="brand-mark"><AudioLines size={17}/></span><span>VoiceCheck</span></button>
          <div className="online"><span /> Detector online</div>
        </header>

        <main className="dashboard">
          <div className="dashboard-head">
            <div>
              <div className="eyebrow">ANALYSIS REPORT</div>
              <h1>Voice analysis</h1>
              <p>Detection summary for the submitted recording.</p>
            </div>
            <button className="secondary-btn" onClick={reset}><RotateCcw size={16}/> New analysis</button>
          </div>

          <section className="result-grid">
            <div className={`verdict-card ${result.isAI ? 'ai' : 'human'}`}>
              <div className="verdict-icon">{result.isAI ? <CircleAlert size={28}/> : <ShieldCheck size={28}/>}</div>
              <div className="eyebrow">VERDICT</div>
              <h2>{result.isAI ? 'Likely AI-generated' : 'Likely human'}</h2>
              <p>The model classified this recording as {result.isAI ? 'synthetic' : 'bona fide'}.</p>
              <div className="confidence-row">
                <div>
                  <span>Confidence</span>
                  <strong>{confidence == null ? '—' : `${Math.round(confidence * 100)}%`}</strong>
                </div>
                <div className="confidence-track"><span style={{ width: `${Math.max(0, Math.min(100, (confidence ?? 0) * 100))}%` }}/></div>
              </div>
            </div>

            <div className="panel probability-panel">
              <div className="panel-title"><span>Classification confidence</span><Gauge size={18}/></div>
              <div className="prob-list">
                <div className="prob-line"><span><i className="dot human-dot"/>Human voice</span><strong>{human == null ? '—' : `${Math.round(human * 100)}%`}</strong></div>
                <div className="bar"><span style={{width:`${(human ?? 0) * 100}%`}}/></div>
                <div className="prob-line"><span><i className="dot ai-dot"/>AI-generated</span><strong>{ai == null ? '—' : `${Math.round(ai * 100)}%`}</strong></div>
                <div className="bar"><span style={{width:`${(ai ?? 0) * 100}%`}}/></div>
              </div>
              <div className="note">Probabilities are model estimates, not certainty.</div>
            </div>
          </section>

          <section className="dashboard-lower">
            <div className="panel audio-panel">
              <div className="panel-title"><span>Submitted audio</span><FileAudio size={18}/></div>
              <div className="audio-file">
                <div className="file-icon"><FileAudio size={20}/></div>
                <div className="file-copy"><strong>{file?.name || 'Recording'}</strong><span>{formatBytes(file?.size)} · {file?.type || 'audio'}</span></div>
                <button className="icon-btn" onClick={togglePlay}>{playing ? <Pause size={17}/> : <Play size={17}/>}</button>
              </div>
              <Waveform active={playing}/>
              <audio ref={audioRef} src={sourceUrl} onEnded={() => setPlaying(false)} />
            </div>

            <div className="panel details-panel">
              <div className="panel-title"><span>Analysis details</span><Activity size={18}/></div>
              <div className="detail"><span>Model</span><strong>{result.model}</strong></div>
              <div className="detail"><span>Processing time</span><strong>{result.processing != null ? `${result.processing} ms` : `${result.clientProcessing} ms`}</strong></div>
              <div className="detail"><span>Input</span><strong>{file?.name || 'Microphone'}</strong></div>
              <div className="detail"><span>Status</span><strong className="success"><Check size={15}/> Complete</strong></div>
            </div>
          </section>
          <div className="dashboard-foot"><span>VoiceCheck · Local prototype</span><span>Detection may produce false positives and false negatives.</span></div>
        </main>
      </div>
    )
  }

  return (
    <div className="app">
      <header className="topbar">
        <button className="brand" onClick={reset}><span className="brand-mark"><AudioLines size={17}/></span><span>VoiceCheck</span></button>
        <div className="online"><span /> Detector online</div>
      </header>

      <main className="home">
        <section className="hero">
          <div className="eyebrow">AI VOICE DETECTION</div>
          <h1>Is this voice<br/><em>human?</em></h1>
          <p>Upload a recording or use your microphone to analyze whether the voice is likely human or AI-generated.</p>
        </section>

        <section className="input-shell">
          <div
            className={`dropzone ${dragging ? 'dragging' : ''} ${file ? 'has-file' : ''}`}
            onDragOver={e => { e.preventDefault(); setDragging(true) }}
            onDragLeave={() => setDragging(false)}
            onDrop={e => { e.preventDefault(); setDragging(false); chooseFile(e.dataTransfer.files?.[0]) }}
            onClick={() => inputRef.current?.click()}
          >
            <input ref={inputRef} type="file" accept="audio/*,.wav,.mp3,.m4a,.flac,.ogg,.webm" onChange={onFileInput}/>
            {file ? (
              <>
                <div className="file-icon large"><FileAudio size={25}/></div>
                <strong>{file.name}</strong>
                <span>{fileMeta.size} · Ready to analyze</span>
                <button className="clear-file" onClick={e => { e.stopPropagation(); reset() }}><X size={15}/> Remove</button>
              </>
            ) : (
              <>
                <div className="upload-icon"><Upload size={23}/></div>
                <strong>Drop an audio file here</strong>
                <span>or click to browse · WAV, MP3, M4A, FLAC, OGG, WEBM</span>
              </>
            )}
          </div>

          <div className="or"><span>OR</span></div>

          <button className={`mic-card ${recording ? 'recording' : ''}`} onClick={recording ? stopRecording : startRecording}>
            <span className="mic-button">{recording ? <Pause size={20}/> : <Mic size={20}/>}</span>
            <span><strong>{recording ? `Recording ${fmtTime(seconds)}` : 'Record from microphone'}</strong><small>{recording ? 'Click to stop recording' : 'Use your browser microphone'}</small></span>
            {recording && <span className="record-dot"/>}
          </button>

          {error && <div className="error"><CircleAlert size={17}/>{error}</div>}

          <button className="analyze-btn" disabled={!file || busy} onClick={analyze}>
            <span>{busy ? 'Analyzing recording' : 'Analyze voice'}</span>
            {!busy && <span>→</span>}
          </button>
        </section>

        {busy && (
          <section className="pipeline panel">
            <div className="pipeline-head"><div><div className="eyebrow">ANALYSIS IN PROGRESS</div><h3>Examining the voice signal</h3></div><div className="loader-ring"/></div>
            <div className="pipeline-steps">
              {PIPELINE.map(([title, desc], i) => {
                const done = stage > i
                const current = stage === i
                return <div className={`step ${done ? 'done' : ''} ${current ? 'current' : ''}`} key={title}>
                  <div className="step-marker">{done ? <Check size={14}/> : <span>{i + 1}</span>}</div>
                  <div><strong>{title}</strong><small>{desc}</small></div>
                </div>
              })}
            </div>
            <Waveform active/>
          </section>
        )}

        {!busy && !file && (
          <div className="trust-row"><span><ShieldCheck size={16}/> Local analysis</span><span><Clock3 size={16}/> Fast results</span><span><Gauge size={16}/> Confidence report</span></div>
        )}

        <footer>VoiceCheck · Local prototype <span>•</span> Detection is probabilistic and may produce false positives or false negatives.</footer>
      </main>
    </div>
  )
}

createRoot(document.getElementById('root')).render(<App />)

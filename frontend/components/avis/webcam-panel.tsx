'use client'

import { useState, useRef, useCallback } from 'react'
import { Brain, Camera, Upload, Video, Loader2 } from 'lucide-react'
import Webcam from 'react-webcam'
import { cn } from '@/lib/utils'

interface WebcamPanelProps {
  isProcessing: boolean
  onAnalyzeStart: () => void
  onAnalyzeComplete: (data: any) => void
  onAnalyzeError: (err: any) => void
}

export function WebcamPanel({ isProcessing, onAnalyzeStart, onAnalyzeComplete, onAnalyzeError }: WebcamPanelProps) {
  const [tab, setTab] = useState<'live' | 'upload'>('live')
  const webcamRef = useRef<Webcam>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleAnalyze = useCallback(async (file: File) => {
    onAnalyzeStart()
    try {
      const formData = new FormData()
      formData.append('file', file)

      const response = await fetch('http://127.0.0.1:8000/api/v1/predict', {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) throw new Error('API Error')
      
      const data = await response.json()
      onAnalyzeComplete(data)
    } catch (err) {
      console.error(err)
      onAnalyzeError(err)
      alert("Failed to connect to Python Backend. Is FastAPI running on port 8000?")
    }
  }, [onAnalyzeStart, onAnalyzeComplete, onAnalyzeError])

  const handleCapture = useCallback(() => {
    if (tab === 'live' && webcamRef.current) {
      const imageSrc = webcamRef.current.getScreenshot()
      if (imageSrc) {
        // Convert base64 to file
        fetch(imageSrc)
          .then(res => res.blob())
          .then(blob => {
            const file = new File([blob], "snapshot.jpg", { type: "image/jpeg" })
            handleAnalyze(file)
          })
      }
    } else if (tab === 'upload' && fileInputRef.current?.files?.[0]) {
      handleAnalyze(fileInputRef.current.files[0])
    }
  }, [tab, handleAnalyze])

  return (
    <section className="flex flex-col gap-6" aria-label="Visual input">
      {/* Source tabs */}
      <div className="glass inline-flex w-fit items-center gap-1 rounded-full p-1">
        <button
          type="button"
          onClick={() => setTab('live')}
          className={cn(
            'flex items-center gap-2 rounded-full px-4 py-2 text-sm font-medium transition-colors',
            tab === 'live' ? 'bg-primary text-primary-foreground' : 'text-muted-foreground hover:text-foreground'
          )}
        >
          <Video className="size-4" aria-hidden="true" /> Live Webcam
        </button>
        <button
          type="button"
          onClick={() => setTab('upload')}
          className={cn(
            'flex items-center gap-2 rounded-full px-4 py-2 text-sm font-medium transition-colors',
            tab === 'upload' ? 'bg-primary text-primary-foreground' : 'text-muted-foreground hover:text-foreground'
          )}
        >
          <Upload className="size-4" aria-hidden="true" /> Upload Image
        </button>
      </div>

      {/* Feed area */}
      <div className="glass glow-ring relative aspect-[4/3] w-full overflow-hidden rounded-3xl bg-black/40">
        
        {tab === 'live' ? (
          <Webcam
            audio={false}
            ref={webcamRef}
            screenshotFormat="image/jpeg"
            className="h-full w-full object-cover"
            videoConstraints={{ facingMode: "user" }}
          />
        ) : (
          <div className="flex h-full flex-col items-center justify-center gap-4 text-center">
             <input 
                type="file" 
                accept="image/*" 
                ref={fileInputRef} 
                className="hidden" 
                onChange={(e) => {
                    if(e.target.files?.[0]) handleAnalyze(e.target.files[0])
                }}
             />
             <button onClick={() => fileInputRef.current?.click()} className="grid size-16 place-items-center rounded-2xl bg-primary/15 ring-1 ring-primary/30 hover:bg-primary/25 transition">
                <Upload className="size-7 text-primary" aria-hidden="true" />
             </button>
             <p className="text-base font-medium text-foreground">Click to upload image</p>
          </div>
        )}

        {/* Scanning line animation during processing */}
        {isProcessing && <div className="animate-scan pointer-events-none absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-primary to-transparent shadow-[0_0_15px_var(--primary)]" />}
      </div>

      {/* Primary action */}
      <button
        onClick={handleCapture}
        disabled={isProcessing}
        type="button"
        className="group relative flex w-full items-center justify-center gap-3 overflow-hidden rounded-2xl bg-primary px-6 py-4 text-base font-semibold text-primary-foreground shadow-[0_0_28px_-6px_var(--primary)] transition-all duration-300 hover:-translate-y-0.5 hover:shadow-[0_0_40px_-4px_var(--primary)] active:translate-y-0 disabled:opacity-50 disabled:hover:translate-y-0"
      >
        {isProcessing ? <Loader2 className="size-5 animate-spin" /> : <Brain className="size-5" aria-hidden="true" />}
        {isProcessing ? 'Analyzing Faces...' : 'Capture & Analyze Snapshot'}
      </button>
    </section>
  )
}

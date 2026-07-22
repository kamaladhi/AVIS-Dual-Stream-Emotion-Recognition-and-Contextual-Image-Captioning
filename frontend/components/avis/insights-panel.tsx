import { Quote, Sparkles, Timer, Activity } from 'lucide-react'
import { EmotionBar } from './emotion-bar'

const EMOTION_COLORS: Record<string, string> = {
  'happy': 'oklch(0.78 0.14 195)',
  'surprise': 'oklch(0.62 0.19 274)',
  'neutral': 'oklch(0.7 0.02 265)',
  'sad': 'oklch(0.6 0.12 250)',
  'fear': 'oklch(0.7 0.16 320)',
  'angry': 'oklch(0.65 0.2 20)',
  'disgust': 'oklch(0.65 0.18 145)',
}

interface InsightsPanelProps {
  result?: any
  isProcessing?: boolean
}

export function InsightsPanel({ result, isProcessing }: InsightsPanelProps) {
  
  const agenticCaption = result?.agentic_caption || (isProcessing ? "Analyzing visual data..." : "Please capture or upload an image to generate an empathetic analysis.")
  const processingTime = result?.processing_time_ms ? (result.processing_time_ms / 1000).toFixed(1) + 's' : '--'
  
  // Format emotions
  let emotionsList = []
  if (result?.faces && result.faces.length > 0) {
    const face = result.faces[0] // analyze first face
    const probs = face.emotion_probs
    emotionsList = Object.keys(probs).map(key => ({
      label: key.charAt(0).toUpperCase() + key.slice(1),
      value: Math.round(probs[key] * 100),
      color: EMOTION_COLORS[key.toLowerCase()] || 'oklch(0.7 0.02 265)'
    })).sort((a, b) => b.value - a.value)
    if(emotionsList.length > 0) emotionsList[0] = { ...emotionsList[0], dominant: true }
  } else {
    emotionsList = Object.keys(EMOTION_COLORS).map(key => ({
      label: key.charAt(0).toUpperCase() + key.slice(1),
      value: 0,
      color: EMOTION_COLORS[key]
    }))
  }

  return (
    <section className={`flex flex-col gap-6 transition-opacity duration-300 ${isProcessing ? 'opacity-50' : 'opacity-100'}`} aria-label="AI insights">
      {/* Agentic caption */}
      <article className="glass rounded-3xl p-6 sm:p-7">
        <header className="mb-4 flex items-center gap-2.5">
          <span className="grid size-8 place-items-center rounded-lg bg-primary/15 ring-1 ring-primary/30">
            <Sparkles className="size-4 text-primary" aria-hidden="true" />
          </span>
          <h2 className="text-sm font-semibold uppercase tracking-widest text-muted-foreground">
            AI Agentic Caption
          </h2>
        </header>
        <div className="relative rounded-2xl bg-secondary/40 p-5">
          <Quote
            className="absolute -left-1 -top-1 size-8 text-primary/25"
            aria-hidden="true"
          />
          <blockquote className="pl-6 text-lg font-medium leading-relaxed text-pretty text-foreground/90">
            {agenticCaption}
          </blockquote>
        </div>
      </article>

      {/* Emotion analysis */}
      <article className="glass rounded-3xl p-6 sm:p-7">
        <header className="mb-5 flex items-center gap-2.5">
          <span className="grid size-8 place-items-center rounded-lg bg-[var(--color-accent-cyan)]/15 ring-1 ring-[var(--color-accent-cyan)]/30">
            <Activity
              className="size-4 text-[var(--color-accent-cyan)]"
              aria-hidden="true"
            />
          </span>
          <h2 className="text-sm font-semibold uppercase tracking-widest text-muted-foreground">
            Emotion Analysis
          </h2>
        </header>
        <div className="flex flex-col gap-4">
          {emotionsList.map((e: any) => (
            <EmotionBar key={e.label} {...e} />
          ))}
        </div>
      </article>

      {/* Stat badges */}
      <div className="flex flex-wrap items-center gap-3">
        <div className="glass inline-flex items-center gap-2 rounded-full px-4 py-2">
          <Timer className="size-4 text-primary" aria-hidden="true" />
          <span className="text-sm text-muted-foreground">Processing Time</span>
          <span className="text-sm font-semibold tabular-nums text-foreground">
            {processingTime}
          </span>
        </div>
        {result?.num_faces > 0 && (
          <div className="glass inline-flex items-center gap-2 rounded-full px-4 py-2">
            <span className="size-2 rounded-full bg-[var(--color-accent-cyan)] shadow-[0_0_8px_var(--color-accent-cyan)]" />
            <span className="text-sm text-muted-foreground">Faces Detected</span>
            <span className="text-sm font-semibold tabular-nums text-foreground">
              {result.num_faces}
            </span>
          </div>
        )}
      </div>
    </section>
  )
}

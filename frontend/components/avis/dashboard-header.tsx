import { BrainCircuit } from 'lucide-react'

export function DashboardHeader() {
  return (
    <header className="flex flex-col gap-6 sm:flex-row sm:items-center sm:justify-between">
      <div className="flex items-center gap-4">
        <span className="glow-ring grid size-12 place-items-center rounded-2xl bg-primary/15">
          <BrainCircuit className="size-6 text-primary" aria-hidden="true" />
        </span>
        <div>
          <h1 className="text-xl font-semibold tracking-tight text-glow sm:text-2xl">
            AVIS
          </h1>
          <p className="text-sm text-muted-foreground">
            Affective Visual Intelligence System
          </p>
        </div>
      </div>

      <div className="glass flex items-center gap-2.5 self-start rounded-full px-4 py-2 sm:self-auto">
        <span className="animate-pulse-dot size-2 rounded-full bg-[var(--color-accent-cyan)] shadow-[0_0_10px_var(--color-accent-cyan)]" />
        <span className="text-sm font-medium text-foreground">
          System Online
        </span>

      </div>
    </header>
  )
}

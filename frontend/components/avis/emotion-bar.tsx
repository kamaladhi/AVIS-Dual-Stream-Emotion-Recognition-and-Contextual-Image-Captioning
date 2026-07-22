interface EmotionBarProps {
  label: string
  value: number
  color: string
  dominant?: boolean
}

export function EmotionBar({ label, value, color, dominant }: EmotionBarProps) {
  return (
    <div className="flex items-center gap-4">
      <span className="w-20 shrink-0 text-sm font-medium text-muted-foreground">
        {label}
      </span>
      <div className="relative h-2.5 flex-1 overflow-hidden rounded-full bg-secondary/60">
        <div
          className="absolute inset-y-0 left-0 rounded-full transition-[width] duration-700 ease-out"
          style={{
            width: `${value}%`,
            backgroundColor: color,
            boxShadow: dominant
              ? `0 0 16px -2px ${color}, 0 0 4px 0 ${color}`
              : `0 0 8px -3px ${color}`,
          }}
        />
      </div>
      <span
        className="w-11 shrink-0 text-right text-sm font-semibold tabular-nums"
        style={{ color: dominant ? color : undefined }}
      >
        {value}%
      </span>
    </div>
  )
}

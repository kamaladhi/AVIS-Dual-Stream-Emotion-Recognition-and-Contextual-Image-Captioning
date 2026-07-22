'use client'

import { useState } from 'react'
import { DashboardHeader } from '@/components/avis/dashboard-header'
import { WebcamPanel } from '@/components/avis/webcam-panel'
import { InsightsPanel } from '@/components/avis/insights-panel'

export default function Page() {
  const [analysisResult, setAnalysisResult] = useState<any>(null)
  const [isProcessing, setIsProcessing] = useState(false)

  return (
    <main className="avis-bg relative min-h-screen w-full px-5 py-8 sm:px-8 lg:px-12 lg:py-12">
      <div className="mx-auto flex max-w-6xl flex-col gap-8 lg:gap-10">
        <DashboardHeader />
        <div className="grid grid-cols-1 gap-8 lg:grid-cols-2 lg:gap-10">
          <WebcamPanel 
            isProcessing={isProcessing}
            onAnalyzeStart={() => setIsProcessing(true)} 
            onAnalyzeComplete={(data) => {
              setAnalysisResult(data)
              setIsProcessing(false)
            }} 
            onAnalyzeError={() => setIsProcessing(false)}
          />
          <InsightsPanel result={analysisResult} isProcessing={isProcessing} />
        </div>
      </div>
    </main>
  )
}

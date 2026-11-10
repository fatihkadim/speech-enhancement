import { useEffect, useMemo, useRef, useState } from 'react'
import { decodeMatrix, type EnhanceResult } from '../lib/api'
import { MASK_LUT, SPEC_LUT, matrixToImage } from '../lib/colormap'
import { formatTime } from '../lib/audio'
import type { Track } from '../lib/usePlayer'

type Props = {
  result: EnhanceResult
  time: number
  onSeek: (seconds: number) => void
  track: Track
  onTrack: (t: Track) => void
}

/**
 * Gürültülü ve temizlenmiş spektrogramlar üst üste durur; sürüklenebilir bir sınır,
 * sağ tarafta temizlenmiş halini gösterir. Spektrograma tıklamak o ana atlar.
 */
export function SpectroCompare({ result, time, onSeek, track, onTrack }: Props) {
  const { spectrogram: s, duration } = result
  const noisyRef = useRef<HTMLCanvasElement>(null)
  const cleanRef = useRef<HTMLCanvasElement>(null)
  const boxRef = useRef<HTMLDivElement>(null)
  const [split, setSplit] = useState(0.5)
  const [dragging, setDragging] = useState(false)

  useEffect(() => {
    for (const [ref, b64] of [
      [noisyRef, s.noisy],
      [cleanRef, s.enhanced],
    ] as const) {
      const c = ref.current
      if (!c) continue
      c.width = s.frames
      c.height = s.bins
      c.getContext('2d')!.putImageData(matrixToImage(decodeMatrix(b64), s.bins, s.frames, SPEC_LUT), 0, 0)
    }
  }, [s])

  const fractionAt = (clientX: number) => {
    const r = boxRef.current!.getBoundingClientRect()
    return Math.min(1, Math.max(0, (clientX - r.left) / r.width))
  }

  const kHzTicks = useMemo(() => {
    const top = s.max_hz / 1000
    return Array.from({ length: Math.floor(top / 2) + 1 }, (_, i) => i * 2)
  }, [s.max_hz])

  const timeTicks = useMemo(() => {
    const step = duration > 20 ? 5 : duration > 8 ? 2 : 1
    return Array.from({ length: Math.floor(duration / step) + 1 }, (_, i) => i * step)
  }, [duration])

  const pct = `${split * 100}%`

  return (
    <figure className="select-none">
      <div className="flex items-end justify-between pb-2 text-sm">
        <button
          type="button"
          onClick={() => onTrack('noisy')}
          className={`rounded-full px-3 py-1 font-medium transition-colors ${
            track === 'noisy' ? 'bg-noisy text-white' : 'text-noisy hover:bg-noisy-soft'
          }`}
        >
          Gürültülü girdi
        </button>
        <button
          type="button"
          onClick={() => onTrack('enhanced')}
          className={`rounded-full px-3 py-1 font-medium transition-colors ${
            track === 'enhanced' ? 'bg-clean text-white' : 'text-clean hover:bg-clean-soft'
          }`}
        >
          Temizlenmiş çıktı
        </button>
      </div>

      <div className="flex gap-2">
        <div className="relative w-7 shrink-0 text-right text-[11px] text-muted" aria-hidden>
          {kHzTicks.map((k) => (
            <span key={k} className="absolute right-0 -translate-y-1/2" style={{ top: `${100 - (k * 1000 * 100) / s.max_hz}%` }}>
              {k}
            </span>
          ))}
        </div>

        <div
          ref={boxRef}
          className="relative h-[clamp(220px,38vh,420px)] flex-1 cursor-crosshair overflow-hidden rounded-lg bg-ink"
          onPointerDown={(e) => {
            if ((e.target as HTMLElement).dataset.handle) return
            onSeek(fractionAt(e.clientX) * duration)
          }}
          onPointerMove={(e) => dragging && setSplit(fractionAt(e.clientX))}
          onPointerUp={() => setDragging(false)}
          onPointerCancel={() => setDragging(false)}
        >
          <canvas ref={noisyRef} className="absolute inset-0 h-full w-full" />
          <canvas
            ref={cleanRef}
            className="absolute inset-0 h-full w-full"
            style={{ clipPath: `inset(0 0 0 ${pct})` }}
          />

          {/* Oynatma konumu */}
          <div
            className="pointer-events-none absolute inset-y-0 w-px bg-white/80"
            style={{ left: `${(time / duration) * 100}%` }}
          />

          {/* Karşılaştırma sınırı */}
          <div className="absolute inset-y-0 -ml-px w-0.5 bg-white" style={{ left: pct }}>
            <div
              data-handle="1"
              role="slider"
              tabIndex={0}
              aria-label="Karşılaştırma sınırı"
              aria-valuemin={0}
              aria-valuemax={100}
              aria-valuenow={Math.round(split * 100)}
              aria-valuetext={`Solda yüzde ${Math.round(split * 100)} gürültülü, sağda temizlenmiş`}
              onPointerDown={(e) => {
                e.currentTarget.setPointerCapture(e.pointerId)
                setDragging(true)
              }}
              onPointerMove={(e) => dragging && setSplit(fractionAt(e.clientX))}
              onPointerUp={() => setDragging(false)}
              onKeyDown={(e) => {
                if (e.key === 'ArrowLeft') setSplit((v) => Math.max(0, v - 0.05))
                if (e.key === 'ArrowRight') setSplit((v) => Math.min(1, v + 0.05))
              }}
              className="absolute top-1/2 left-1/2 flex h-11 w-6 -translate-x-1/2 -translate-y-1/2 cursor-ew-resize items-center justify-center gap-[3px] rounded-md bg-white shadow-[0_2px_10px_rgba(22,26,35,0.35)]"
            >
              <span data-handle="1" className="h-4 w-px bg-ink/40" />
              <span data-handle="1" className="h-4 w-px bg-ink/40" />
            </div>
          </div>
        </div>
      </div>

      <div className="relative mt-1 ml-9 h-4 text-[11px] text-muted" aria-hidden>
        {timeTicks.map((t) => (
          <span
            key={t}
            className={`absolute ${t / duration > 0.97 ? '-translate-x-full' : t === 0 ? '' : '-translate-x-1/2'}`}
            style={{ left: `${(t / duration) * 100}%` }}
          >
            {formatTime(t)}
          </span>
        ))}
      </div>
      <figcaption className="sr-only">
        Gürültülü ve temizlenmiş spektrogramların karşılaştırması. Dikey eksen frekans (kHz), yatay eksen zaman.
      </figcaption>
    </figure>
  )
}

/** Modelin her zaman-frekans hücresine uyguladığı maske. */
export function MaskView({ result }: { result: EnhanceResult }) {
  const ref = useRef<HTMLCanvasElement>(null)
  const { spectrogram: s } = result
  useEffect(() => {
    const c = ref.current
    if (!c) return
    c.width = s.frames
    c.height = s.mask_bins
    c.getContext('2d')!.putImageData(matrixToImage(decodeMatrix(s.mask), s.mask_bins, s.frames, MASK_LUT), 0, 0)
  }, [s])

  return (
    <div>
      <canvas ref={ref} className="h-28 w-full rounded-lg bg-ink" />
      <div className="mt-2 flex items-center gap-3 text-xs text-muted">
        <span className="text-noisy">Silinen</span>
        <span
          className="h-2 w-40 rounded-full"
          style={{ background: 'linear-gradient(90deg, rgb(184,110,18), rgb(42,48,60), rgb(63,182,168))' }}
        />
        <span className="text-clean">Korunan</span>
      </div>
    </div>
  )
}

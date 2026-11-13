import { Mic, Smartphone, Square, Upload } from 'lucide-react'
import { useEffect, useRef, useState, type ReactNode } from 'react'
import type { Sample } from '../lib/api'
import { formatTime } from '../lib/audio'

type Props = {
  samples: Sample[]
  activeSampleId: string | null
  busy: boolean
  onFile: (file: Blob, title: string) => void
  onSample: (s: Sample) => void
}

export function InputPanel({ samples, activeSampleId, busy, onFile, onSample }: Props) {
  const testSamples = samples.filter((s) => s.kind === 'test')
  const realSamples = samples.filter((s) => s.kind === 'real')
  const item = (s: Sample, icon: ReactNode, right: string) => (
    <li key={s.id}>
      <button
        type="button"
        disabled={busy}
        onClick={() => onSample(s)}
        aria-current={activeSampleId === s.id}
        className="group flex w-full items-center gap-3 py-2.5 text-left transition-colors hover:bg-surface disabled:cursor-wait aria-[current=true]:bg-surface"
      >
        {icon}
        <span className="flex-1">{s.label.charAt(0).toLocaleUpperCase('tr') + s.label.slice(1)}</span>
        <span className="text-sm text-muted tabular-nums">{right}</span>
      </button>
    </li>
  )

  return (
    <div className="flex flex-col gap-8">
      <DropZone busy={busy} onFile={onFile} />
      <Recorder busy={busy} onFile={onFile} />
      {testSamples.length > 0 && (
        <section aria-labelledby="samples-title">
          <h2 id="samples-title" className="text-base font-semibold">
            Örneklerle dene
          </h2>
          <p className="mt-1 text-sm text-muted">
            Test bölmesinden kayıtlar. Temiz hali bilindiği için ölçümler de gösterilir.
          </p>
          <ul className="mt-3 divide-y divide-line border-y border-line">
            {testSamples.map((s) => {
              const snr = s.snr ?? 0
              return item(s, <SnrMeter snr={snr} />, `${snr > 0 ? `+${snr}` : snr} dB`)
            })}
          </ul>
          <p className="mt-2 text-xs text-muted">Çubuklar sinyal-gürültü oranını gösterir; az çubuk, daha çok gürültü.</p>
        </section>
      )}
      {realSamples.length > 0 && (
        <section aria-labelledby="real-title">
          <h2 id="real-title" className="text-base font-semibold">
            Gerçek kayıtlar
          </h2>
          <p className="mt-1 text-sm text-muted">
            Telefonla alınmış sesli mesajlar. Temiz hali olmadığı için ölçüm yok; kulakla ve spektrogramla değerlendir.
          </p>
          <ul className="mt-3 divide-y divide-line border-y border-line">
            {realSamples.map((s) =>
              item(s, <Smartphone className="ml-1 size-4 text-ink-soft" aria-hidden />, formatTime(s.duration)),
            )}
          </ul>
        </section>
      )}
    </div>
  )
}

/** -5 … 15 dB SNR'ı beş çubukla gösterir. */
function SnrMeter({ snr }: { snr: number }) {
  const level = Math.round((snr + 5) / 5) + 1 // -5 → 1, 15 → 5
  return (
    <span className="flex h-4 items-end gap-[2px] pl-1" aria-hidden>
      {[1, 2, 3, 4, 5].map((i) => (
        <span
          key={i}
          className={`w-[3px] rounded-sm ${i <= level ? 'bg-clean' : 'bg-line'}`}
          style={{ height: `${30 + i * 14}%` }}
        />
      ))}
    </span>
  )
}

function DropZone({ busy, onFile }: { busy: boolean; onFile: Props['onFile'] }) {
  const input = useRef<HTMLInputElement>(null)
  const [over, setOver] = useState(false)

  const take = (files: FileList | null) => {
    const f = files?.[0]
    if (f) onFile(f, f.name.replace(/\.[^.]+$/, ''))
  }

  return (
    <div
      onDragOver={(e) => {
        e.preventDefault()
        setOver(true)
      }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => {
        e.preventDefault()
        setOver(false)
        if (!busy) take(e.dataTransfer.files)
      }}
      className={`rounded-xl border-2 border-dashed p-6 transition-colors ${
        over ? 'border-clean bg-clean-soft' : 'border-line bg-surface'
      }`}
    >
      <Upload className="size-6 text-ink-soft" aria-hidden />
      <p className="mt-3 font-semibold">Ses dosyasını buraya bırak</p>
      <p className="mt-1 text-sm text-muted">WAV, MP3, M4A, OGG; en fazla iki dakika.</p>
      <button
        type="button"
        disabled={busy}
        onClick={() => input.current?.click()}
        className="mt-4 rounded-lg bg-ink px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-ink-soft disabled:opacity-50"
      >
        Dosya seç
      </button>
      <input
        ref={input}
        type="file"
        accept="audio/*,.wav,.flac,.ogg,.mp3,.m4a"
        className="hidden"
        onChange={(e) => {
          take(e.target.files)
          e.target.value = ''
        }}
      />
    </div>
  )
}

function Recorder({ busy, onFile }: { busy: boolean; onFile: Props['onFile'] }) {
  const [recording, setRecording] = useState(false)
  const [elapsed, setElapsed] = useState(0)
  const [error, setError] = useState<string | null>(null)
  const recorder = useRef<MediaRecorder | null>(null)
  const timer = useRef<number>(0)

  useEffect(() => () => window.clearInterval(timer.current), [])

  const start = async () => {
    setError(null)
    let stream: MediaStream
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        // Tarayıcının kendi gürültü bastırması kapalı: modelin işini görmek istiyoruz
        audio: { noiseSuppression: false, echoCancellation: false, autoGainControl: false },
      })
    } catch {
      setError('Mikrofona erişilemedi. Tarayıcıda mikrofon iznini açıp tekrar dene.')
      return
    }
    const chunks: Blob[] = []
    const rec = new MediaRecorder(stream)
    rec.ondataavailable = (e) => chunks.push(e.data)
    rec.onstop = () => {
      stream.getTracks().forEach((t) => t.stop())
      window.clearInterval(timer.current)
      setRecording(false)
      onFile(new Blob(chunks, { type: rec.mimeType }), 'Mikrofon kaydı')
    }
    rec.start()
    recorder.current = rec
    setRecording(true)
    setElapsed(0)
    const t0 = Date.now()
    timer.current = window.setInterval(() => {
      const s = (Date.now() - t0) / 1000
      setElapsed(s)
      if (s >= 60) rec.stop()
    }, 200)
  }

  return (
    <section aria-labelledby="rec-title">
      <h2 id="rec-title" className="text-base font-semibold">
        Ya da kendin kaydet
      </h2>
      <p className="mt-1 text-sm text-muted">Gürültülü bir ortamda konuş; tarayıcının gürültü bastırması kapalıdır.</p>
      {recording ? (
        <button
          type="button"
          onClick={() => recorder.current?.stop()}
          className="mt-3 flex items-center gap-3 rounded-lg bg-danger px-4 py-2 text-sm font-medium text-white"
        >
          <Square className="size-4 fill-current" aria-hidden />
          Kaydı bitir
          <span className="opacity-80">{formatTime(elapsed)}</span>
        </button>
      ) : (
        <button
          type="button"
          disabled={busy}
          onClick={start}
          className="mt-3 flex items-center gap-2 rounded-lg border border-line bg-surface px-4 py-2 text-sm font-medium transition-colors hover:border-ink-soft disabled:opacity-50"
        >
          <Mic className="size-4" aria-hidden />
          Kaydı başlat
        </button>
      )}
      {error && <p className="mt-2 text-sm text-danger">{error}</p>}
    </section>
  )
}

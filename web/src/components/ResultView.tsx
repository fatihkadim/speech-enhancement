import { Download, Pause, Play } from 'lucide-react'
import { useEffect, useMemo, useRef } from 'react'
import { resultAudioUrl, type EnhanceResult, type MetricSet } from '../lib/api'
import { formatTime } from '../lib/audio'
import { usePlayer, type Track } from '../lib/usePlayer'
import { MaskView, SpectroCompare } from './SpectroCompare'

const num = (v: number, digits: number) =>
  v.toLocaleString('tr-TR', { minimumFractionDigits: digits, maximumFractionDigits: digits })

const TRACK_LABEL: Record<Track, string> = { noisy: 'Gürültülü', enhanced: 'Temizlenmiş', clean: 'Asıl temiz kayıt' }

export function ResultView({ result }: { result: EnhanceResult }) {
  const urls = useMemo(() => {
    const u: Partial<Record<Track, string>> = {
      noisy: resultAudioUrl(result.id, 'noisy'),
      enhanced: resultAudioUrl(result.id, 'enhanced'),
    }
    if (result.has_clean) u.clean = resultAudioUrl(result.id, 'clean')
    return u
  }, [result])
  const player = usePlayer(urls)
  const tracks = Object.keys(urls) as Track[]

  const playerRef = useRef(player)
  playerRef.current = player

  // Boşluk: oynat/durdur, A: gürültülü, B: temizlenmiş
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement).closest('input, textarea, button, a, [role=slider]')) return
      const p = playerRef.current
      if (e.code === 'Space') {
        e.preventDefault()
        void p.toggle()
      } else if (e.key === 'a' || e.key === 'A') p.setTrack('noisy')
      else if (e.key === 'b' || e.key === 'B') p.setTrack('enhanced')
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  return (
    <article className="flex flex-col gap-8">
      <header className="flex flex-wrap items-baseline justify-between gap-x-6 gap-y-1">
        <h2 className="text-2xl font-semibold tracking-tight">{result.title}</h2>
        <p className="text-sm text-muted">
          {num(result.duration, 1)} sn kayıt, {result.processing_ms < 1000 ? `${Math.round(result.processing_ms)} ms` : `${num(result.processing_ms / 1000, 1)} sn`}
          {result.device.startsWith('cuda') ? ' içinde GPU’da' : ' içinde CPU’da'} temizlendi
        </p>
      </header>

      <SpectroCompare
        result={result}
        time={player.time}
        onSeek={player.seek}
        track={player.track}
        onTrack={player.setTrack}
      />

      <div className="flex flex-wrap items-center gap-4">
        <button
          type="button"
          onClick={() => void player.toggle()}
          aria-label={player.playing ? 'Durdur' : 'Oynat'}
          className="flex size-12 items-center justify-center rounded-full bg-ink text-white transition-colors hover:bg-ink-soft"
        >
          {player.playing ? <Pause className="size-5 fill-current" /> : <Play className="ml-0.5 size-5 fill-current" />}
        </button>

        <div role="radiogroup" aria-label="Dinlenen sürüm" className="flex rounded-full border border-line bg-surface p-1">
          {tracks.map((t) => (
            <button
              key={t}
              type="button"
              role="radio"
              aria-checked={player.track === t}
              onClick={() => player.setTrack(t)}
              className={`rounded-full px-4 py-1.5 text-sm font-medium transition-colors ${
                player.track === t
                  ? t === 'noisy'
                    ? 'bg-noisy text-white'
                    : t === 'enhanced'
                      ? 'bg-clean text-white'
                      : 'bg-ink text-white'
                  : 'text-muted hover:text-ink'
              }`}
            >
              {TRACK_LABEL[t]}
            </button>
          ))}
        </div>

        <span className="text-sm text-muted">
          {formatTime(player.time)} / {formatTime(player.duration || result.duration)}
        </span>

        <a
          href={resultAudioUrl(result.id, 'enhanced')}
          download={`${result.title.replace(/[^\p{L}\p{N}]+/gu, '_')}_temiz.wav`}
          className="ml-auto flex items-center gap-2 rounded-lg border border-line bg-surface px-4 py-2 text-sm font-medium transition-colors hover:border-ink-soft"
        >
          <Download className="size-4" aria-hidden />
          Temizlenmiş sesi indir
        </a>
      </div>
      <p className="-mt-5 text-xs text-muted">
        Sürümler birlikte çalar; geçiş yaptığında aynı andan devam eder. Kısayollar: boşluk, A ve B.
      </p>

      <div className="grid gap-10 lg:grid-cols-[1fr_minmax(260px,340px)]">
        <section aria-labelledby="mask-title">
          <h3 id="mask-title" className="font-semibold">
            Modelin maskesi
          </h3>
          <p className="mt-1 mb-3 max-w-prose text-sm text-muted">
            U-Net her zaman-frekans hücresi için 0 ile 1 arasında bir katsayı üretir. Konuşmanın harmonikleri korunur,
            aradaki gürültü bastırılır.
          </p>
          <MaskView result={result} />
        </section>
        <Metrics result={result} />
      </div>
    </article>
  )
}

function Metrics({ result }: { result: EnhanceResult }) {
  if (!result.metrics) {
    return (
      <section aria-labelledby="metrics-title">
        <h3 id="metrics-title" className="font-semibold">
          Ölçümler
        </h3>
        <dl className="mt-3 divide-y divide-line border-y border-line text-sm">
          <Row label="Kaldırılan enerji" value={`${num(result.removed_db, 1)} dB`} />
          <Row label="Süre" value={`${num(result.duration, 1)} sn`} />
          <Row label="İşlem süresi" value={`${Math.round(result.processing_ms)} ms`} />
        </dl>
        <p className="mt-2 text-xs text-muted">
          Bu kaydın temiz hali bilinmediği için SI-SNR, PESQ ve STOI hesaplanamaz. Ölçümleri görmek için test bölmesinden bir örnek seç.
        </p>
      </section>
    )
  }
  const { noisy, enhanced } = result.metrics
  const rows: { key: keyof MetricSet; label: string; hint: string; fmt: (v: number) => string; unit?: string }[] = [
    { key: 'si_snr', label: 'SI-SNR', hint: 'sinyal/bozulma oranı', fmt: (v) => num(v, 1), unit: 'dB' },
    { key: 'pesq', label: 'PESQ', hint: 'algılanan kalite, 1–4,5', fmt: (v) => num(v, 2) },
    { key: 'stoi', label: 'STOI', hint: 'anlaşılırlık, 0–1', fmt: (v) => num(v, 3) },
  ]
  return (
    <section aria-labelledby="metrics-title">
      <h3 id="metrics-title" className="font-semibold">
        Ölçümler
      </h3>
      <p className="mt-1 text-sm text-muted">Asıl temiz kayda göre, önce ve sonra.</p>
      <dl className="mt-3 divide-y divide-line border-y border-line">
        {rows.map((r) => {
          const a = noisy[r.key]
          const b = enhanced[r.key]
          if (a == null || b == null) return null
          const d = b - a
          return (
            <div key={r.key} className="grid grid-cols-[1fr_auto] items-baseline gap-x-4 py-3">
              <dt>
                <span className="font-medium">{r.label}</span>
                <span className="block text-xs text-muted">{r.hint}</span>
              </dt>
              <dd className="text-right">
                <span className="text-noisy">{r.fmt(a)}</span>
                <span className="px-1.5 text-muted">›</span>
                <span className="text-lg font-semibold text-clean">{r.fmt(b)}</span>
                {r.unit && <span className="ml-1 text-xs text-muted">{r.unit}</span>}
                <span className={`block text-xs ${d >= 0 ? 'text-clean' : 'text-danger'}`}>
                  {d >= 0 ? '+' : '−'}
                  {r.fmt(Math.abs(d))}
                </span>
              </dd>
            </div>
          )
        })}
      </dl>
      {noisy.pesq == null && (
        <p className="mt-2 text-xs text-muted">PESQ bu bilgisayarda kurulu olmadığı için gösterilmiyor.</p>
      )}
    </section>
  )
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between py-2.5">
      <dt className="text-muted">{label}</dt>
      <dd className="font-medium">{value}</dd>
    </div>
  )
}

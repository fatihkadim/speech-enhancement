import { Cpu, Zap } from 'lucide-react'
import { useCallback, useEffect, useRef, useState } from 'react'
import { InputPanel } from './components/InputPanel'
import { ResultView } from './components/ResultView'
import { enhanceFile, enhanceSample, getInfo, type EnhanceResult, type Info, type Sample } from './lib/api'
import { toWav16k } from './lib/audio'

type Source = { kind: 'sample'; sample: Sample } | { kind: 'file'; wav: Blob; title: string }

export default function App() {
  const [info, setInfo] = useState<Info | null>(null)
  const [model, setModel] = useState<string>('')
  const [source, setSource] = useState<Source | null>(null)
  const [result, setResult] = useState<EnhanceResult | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const runId = useRef(0)
  const stageRef = useRef<HTMLElement>(null)

  useEffect(() => {
    getInfo()
      .then((i) => {
        setInfo(i)
        // Paylaşılabilir bağlantı: ?ornek=snr-5_hen&model=e1_si_snr
        const params = new URLSearchParams(window.location.search)
        const m = i.models.find((x) => x.id === params.get('model')) ?? i.models[0]
        setModel(m?.id ?? '')
        const s = i.samples.find((x) => x.id === params.get('ornek')?.replace(' ', '+')) // elle yazılan + boşluğa dönüşür
        if (s && m) void run({ kind: 'sample', sample: s }, m.id)
      })
      .catch((e: Error) => setError(e.message))
  }, [])

  const run = useCallback(async (src: Source, modelId: string) => {
    const id = ++runId.current
    setSource(src)
    const url = new URL(window.location.href)
    url.search = src.kind === 'sample' ? new URLSearchParams({ ornek: src.sample.id, model: modelId }).toString() : ''
    window.history.replaceState(null, '', url)
    setBusy(true)
    setError(null)
    try {
      const res =
        src.kind === 'sample' ? await enhanceSample(src.sample.id, modelId) : await enhanceFile(src.wav, src.title, modelId)
      if (id === runId.current) {
        setResult(res)
        if (window.matchMedia('(max-width: 1023px)').matches)
          requestAnimationFrame(() => stageRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' }))
      }
    } catch (e) {
      if (id === runId.current) setError((e as Error).message)
    } finally {
      if (id === runId.current) setBusy(false)
    }
  }, [])

  const onFile = async (file: Blob, title: string) => {
    setError(null)
    try {
      const { wav, seconds } = await toWav16k(file)
      if (info && seconds > info.max_seconds) {
        setError(`Kayıt ${Math.round(seconds)} saniye; en fazla ${info.max_seconds} saniye yükleyebilirsin.`)
        return
      }
      void run({ kind: 'file', wav, title }, model)
    } catch (e) {
      setError((e as Error).message)
    }
  }

  const onModel = (id: string) => {
    setModel(id)
    if (source) void run(source, id)
  }

  const current = info?.models.find((m) => m.id === model)

  return (
    <div className="mx-auto flex min-h-dvh max-w-[1400px] flex-col px-5 sm:px-8">
      <header className="flex flex-wrap items-center justify-between gap-4 border-b border-line py-5">
        <div className="flex items-center gap-3">
          <Logo />
          <div>
            <h1 className="text-lg leading-none font-semibold tracking-tight">SesTemiz</h1>
            <p className="mt-1 text-sm text-muted">U-Net ile konuşmadan gürültü giderme</p>
          </div>
        </div>

        {info && (
          <div className="flex flex-wrap items-center gap-4">
            <div role="radiogroup" aria-label="Model" className="flex rounded-full border border-line bg-surface p-1">
              {info.models.map((m) => (
                <button
                  key={m.id}
                  type="button"
                  role="radio"
                  aria-checked={model === m.id}
                  disabled={busy}
                  onClick={() => onModel(m.id)}
                  title={m.detail}
                  className={`rounded-full px-4 py-1.5 text-sm font-medium transition-colors disabled:cursor-wait ${
                    model === m.id ? 'bg-ink text-white' : 'text-muted hover:text-ink'
                  }`}
                >
                  {m.label}
                </button>
              ))}
            </div>
            <span className="flex items-center gap-1.5 text-sm text-muted" title={info.gpu ?? 'İşlemci'}>
              {info.gpu ? <Zap className="size-4 text-clean" aria-hidden /> : <Cpu className="size-4" aria-hidden />}
              {info.gpu ? info.gpu.replace(/^NVIDIA (GeForce )?/, '') : 'CPU'}
            </span>
          </div>
        )}
      </header>

      <main className="grid flex-1 gap-10 py-8 lg:grid-cols-[320px_1fr] lg:gap-14">
        <aside>
          <InputPanel
            samples={info?.samples ?? []}
            activeSampleId={source?.kind === 'sample' ? source.sample.id : null}
            busy={busy || !info}
            onFile={onFile}
            onSample={(s) => void run({ kind: 'sample', sample: s }, model)}
          />
        </aside>

        <section ref={stageRef} aria-live="polite" className="min-w-0 scroll-mt-4">
          {error && (
            <div role="alert" className="mb-6 rounded-lg border border-danger/30 bg-danger/5 px-4 py-3 text-sm text-danger">
              {error}
            </div>
          )}
          {busy ? (
            <Working label={source?.kind === 'sample' ? source.sample.label : source?.title} model={current?.label} />
          ) : result ? (
            <ResultView key={result.id} result={result} />
          ) : (
            <Empty ready={!!info} />
          )}
        </section>
      </main>

      <footer className="border-t border-line py-5 text-sm text-muted">
        Yapay Sinir Ağları dönem projesi. Fatih Kadim, Mehmet Göktuğ Elbir, Mehmet Burak Albayrak.
      </footer>
    </div>
  )
}

function Empty({ ready }: { ready: boolean }) {
  return (
    <div className="flex h-full min-h-[360px] flex-col justify-center rounded-xl bg-ink px-8 py-12 text-white sm:px-12">
      <FakeSpectrum />
      <h2 className="mt-8 max-w-lg text-display font-semibold tracking-tight">
        Gürültüyü dinle, sonra modelin neyi sildiğini gör.
      </h2>
      <p className="mt-4 max-w-md text-white/70">
        {ready
          ? 'Soldan bir örnek seç, dosya yükle ya da mikrofonla kaydet. Sonuçta iki sürümü yan yana dinler, spektrogramlarını karşılaştırırsın.'
          : 'Sunucuya bağlanılıyor...'}
      </p>
    </div>
  )
}

/** Boş ekranda: solda gürültülü, sağda temiz görünen dekoratif çubuklar. */
function FakeSpectrum() {
  const bars = Array.from({ length: 48 }, (_, i) => {
    const speech = Math.abs(Math.sin(i * 0.55)) * 0.7 + Math.abs(Math.sin(i * 0.21)) * 0.3
    const noise = 0.35 + ((i * 37) % 11) / 22
    const clean = i >= 24
    return { h: clean ? speech * 0.9 + 0.05 : Math.min(1, speech * 0.6 + noise * 0.5), clean }
  })
  return (
    <div className="flex h-24 items-end gap-[3px]" aria-hidden>
      {bars.map((b, i) => (
        <span
          key={i}
          className={`flex-1 rounded-sm ${b.clean ? 'bg-[#3fb6a8]' : 'bg-[#d08a2e]'}`}
          style={{ height: `${b.h * 100}%`, opacity: b.clean ? 0.95 : 0.75 }}
        />
      ))}
    </div>
  )
}

function Working({ label, model }: { label?: string; model?: string }) {
  return (
    <div className="flex flex-col gap-4">
      <p className="text-2xl font-semibold tracking-tight">
        {label ? label.charAt(0).toLocaleUpperCase('tr') + label.slice(1) : 'Kayıt'}
      </p>
      <div className="relative h-[clamp(220px,38vh,420px)] overflow-hidden rounded-lg bg-ink">
        <div className="scan-line absolute inset-y-0 w-1/3 bg-gradient-to-r from-transparent via-[#3fb6a8]/35 to-transparent" />
      </div>
      <p className="text-sm text-muted">{model ? `${model} modeliyle temizleniyor...` : 'Temizleniyor...'}</p>
    </div>
  )
}

function Logo() {
  return (
    <svg viewBox="0 0 32 32" className="size-9" aria-hidden>
      <rect width="32" height="32" rx="7" fill="#161A23" />
      <g strokeLinecap="round" strokeWidth="2.4">
        <path d="M7 13v6M11 9v14M15 12v8" stroke="#D08A2E" />
        <path d="M19 14v4M23 11v10M27 15v2" stroke="#3FB6A8" />
      </g>
    </svg>
  )
}

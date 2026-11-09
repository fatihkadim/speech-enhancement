export type ModelInfo = { id: string; label: string; detail: string }

export type Sample = {
  id: string
  snr: number
  noise: string
  label: string
  has_clean: boolean
}

export type Info = {
  device: string
  gpu: string | null
  max_seconds: number
  models: ModelInfo[]
  samples: Sample[]
}

export type MetricSet = { si_snr: number; pesq: number | null; stoi: number }

export type EnhanceResult = {
  id: string
  title: string
  model: string
  sr: number
  duration: number
  processing_ms: number
  device: string
  removed_db: number
  has_clean: boolean
  metrics: { noisy: MetricSet; enhanced: MetricSet } | null
  spectrogram: {
    bins: number
    frames: number
    max_hz: number
    noisy: string
    enhanced: string
    mask_bins: number
    mask: string
  }
}

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  let res: Response
  try {
    res = await fetch(url, init)
  } catch {
    throw new Error('Sunucuya ulaşılamadı. "uv run python -m src.api" komutuyla API çalışıyor mu?')
  }
  if (!res.ok) {
    const body = await res.json().catch(() => null)
    throw new Error(body?.detail ?? `Sunucu hatası (${res.status}).`)
  }
  return res.json() as Promise<T>
}

export const getInfo = () => request<Info>('/api/info')

export function enhanceSample(sampleId: string, model: string) {
  const form = new FormData()
  form.set('sample_id', sampleId)
  form.set('model', model)
  return request<EnhanceResult>('/api/enhance/sample', { method: 'POST', body: form })
}

export function enhanceFile(wav: Blob, title: string, model: string) {
  const form = new FormData()
  form.set('file', wav, 'kayit.wav')
  form.set('model', model)
  form.set('title', title)
  return request<EnhanceResult>('/api/enhance', { method: 'POST', body: form })
}

export const resultAudioUrl = (id: string, kind: 'noisy' | 'enhanced' | 'clean') =>
  `/api/results/${id}/${kind}.wav`

export function decodeMatrix(b64: string): Uint8Array {
  const bin = atob(b64)
  const out = new Uint8Array(bin.length)
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i)
  return out
}

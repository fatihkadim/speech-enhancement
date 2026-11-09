/**
 * Tarayıcının çözebildiği her sesi (mp3, m4a, webm kaydı...) 16 kHz mono 16-bit WAV'a çevirir.
 * Böylece sunucu tarafında ek bir çözücüye gerek kalmaz.
 */
export async function toWav16k(blob: Blob, sampleRate = 16000): Promise<{ wav: Blob; seconds: number }> {
  const data = await blob.arrayBuffer()
  const ctx = new AudioContext()
  let decoded: AudioBuffer
  try {
    decoded = await ctx.decodeAudioData(data)
  } catch {
    throw new Error('Bu dosya bir ses dosyası olarak açılamadı.')
  } finally {
    void ctx.close()
  }

  const length = Math.max(1, Math.ceil(decoded.duration * sampleRate))
  const offline = new OfflineAudioContext(1, length, sampleRate)
  const src = offline.createBufferSource()
  src.buffer = decoded // çok kanallı kaynak tek kanala otomatik karıştırılır
  src.connect(offline.destination)
  src.start()
  const mono = (await offline.startRendering()).getChannelData(0)
  return { wav: encodeWav(mono, sampleRate), seconds: decoded.duration }
}

function encodeWav(samples: Float32Array, sampleRate: number): Blob {
  const buffer = new ArrayBuffer(44 + samples.length * 2)
  const view = new DataView(buffer)
  const write = (offset: number, s: string) => [...s].forEach((c, i) => view.setUint8(offset + i, c.charCodeAt(0)))
  write(0, 'RIFF')
  view.setUint32(4, 36 + samples.length * 2, true)
  write(8, 'WAVE')
  write(12, 'fmt ')
  view.setUint32(16, 16, true)
  view.setUint16(20, 1, true) // PCM
  view.setUint16(22, 1, true) // mono
  view.setUint32(24, sampleRate, true)
  view.setUint32(28, sampleRate * 2, true)
  view.setUint16(32, 2, true)
  view.setUint16(34, 16, true)
  write(36, 'data')
  view.setUint32(40, samples.length * 2, true)
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]))
    view.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true)
  }
  return new Blob([buffer], { type: 'audio/wav' })
}

export function formatTime(seconds: number): string {
  if (!Number.isFinite(seconds)) return '0:00'
  const m = Math.floor(seconds / 60)
  const s = Math.floor(seconds % 60)
  return `${m}:${s.toString().padStart(2, '0')}`
}

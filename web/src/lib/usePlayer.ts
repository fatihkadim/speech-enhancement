import { useCallback, useEffect, useRef, useState } from 'react'

export type Track = 'noisy' | 'enhanced' | 'clean'

/**
 * Aynı kaydın sürümlerini birlikte çalar ve yalnızca seçileni duyurur.
 * Böylece "gürültülü / temizlenmiş" arasında geçiş, kaldığı yerden kesintisiz olur.
 */
export function usePlayer(urls: Partial<Record<Track, string>>) {
  const audios = useRef<Partial<Record<Track, HTMLAudioElement>>>({})
  const [track, setTrack] = useState<Track>('enhanced')
  const [playing, setPlaying] = useState(false)
  const [time, setTime] = useState(0)
  const [duration, setDuration] = useState(0)
  const trackRef = useRef(track)
  trackRef.current = track

  const key = JSON.stringify(urls)
  useEffect(() => {
    const created: Partial<Record<Track, HTMLAudioElement>> = {}
    for (const [name, url] of Object.entries(urls) as [Track, string][]) {
      const a = new Audio(url)
      a.preload = 'auto'
      a.muted = name !== trackRef.current
      created[name] = a
    }
    audios.current = created
    setPlaying(false)
    setTime(0)
    setDuration(0)

    const lead = created.noisy ?? Object.values(created)[0]
    if (!lead) return
    const onMeta = () => setDuration(lead.duration)
    const onEnd = () => {
      Object.values(created).forEach((a) => a.pause())
      setPlaying(false)
    }
    lead.addEventListener('loadedmetadata', onMeta)
    lead.addEventListener('ended', onEnd)

    let raf = 0
    const tick = () => {
      const active = created[trackRef.current] ?? lead
      setTime(active.currentTime)
      // Sürümler arasında 40 ms'den fazla kayma olursa hizala
      for (const a of Object.values(created)) {
        if (a !== active && Math.abs(a.currentTime - active.currentTime) > 0.04) a.currentTime = active.currentTime
      }
      raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)

    return () => {
      cancelAnimationFrame(raf)
      lead.removeEventListener('loadedmetadata', onMeta)
      lead.removeEventListener('ended', onEnd)
      Object.values(created).forEach((a) => {
        a.pause()
        a.src = ''
      })
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key])

  useEffect(() => {
    for (const [name, a] of Object.entries(audios.current) as [Track, HTMLAudioElement][]) a.muted = name !== track
  }, [track])

  const play = useCallback(async () => {
    const all = Object.values(audios.current)
    const t = audios.current[trackRef.current]?.currentTime ?? 0
    all.forEach((a) => (a.currentTime = t))
    await Promise.all(all.map((a) => a.play().catch(() => undefined)))
    setPlaying(true)
  }, [])

  const pause = useCallback(() => {
    Object.values(audios.current).forEach((a) => a.pause())
    setPlaying(false)
  }, [])

  const seek = useCallback((seconds: number) => {
    Object.values(audios.current).forEach((a) => (a.currentTime = seconds))
    setTime(seconds)
  }, [])

  return { track, setTrack, playing, play, pause, toggle: playing ? pause : play, seek, time, duration }
}

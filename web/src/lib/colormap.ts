/** Spektrogram renk ölçeği: sessizlik koyu mürekkep, güçlü bileşenler açık kum rengi. */
const STOPS: [number, [number, number, number]][] = [
  [0.0, [22, 26, 35]],
  [0.3, [33, 52, 84]],
  [0.55, [14, 124, 123]],
  [0.78, [126, 200, 160]],
  [1.0, [246, 240, 200]],
]

/** Maske: 0 (silinen) kehribar, 1 (korunan) deniz yeşili. */
const MASK_STOPS: [number, [number, number, number]][] = [
  [0.0, [184, 110, 18]],
  [0.5, [42, 48, 60]],
  [1.0, [63, 182, 168]],
]

function buildLut(stops: typeof STOPS): Uint8ClampedArray {
  const lut = new Uint8ClampedArray(256 * 3)
  for (let i = 0; i < 256; i++) {
    const t = i / 255
    let k = 0
    while (k < stops.length - 2 && t > stops[k + 1][0]) k++
    const [t0, c0] = stops[k]
    const [t1, c1] = stops[k + 1]
    const u = (t - t0) / (t1 - t0)
    for (let ch = 0; ch < 3; ch++) lut[i * 3 + ch] = c0[ch] + (c1[ch] - c0[ch]) * u
  }
  return lut
}

export const SPEC_LUT = buildLut(STOPS)
export const MASK_LUT = buildLut(MASK_STOPS)

/** (bins x frames, satır = frekans) matrisini düşük frekans altta olacak şekilde ImageData'ya çevirir. */
export function matrixToImage(data: Uint8Array, bins: number, frames: number, lut: Uint8ClampedArray): ImageData {
  const img = new ImageData(frames, bins)
  for (let f = 0; f < bins; f++) {
    const row = bins - 1 - f
    for (let t = 0; t < frames; t++) {
      const v = data[f * frames + t]
      const o = (row * frames + t) * 4
      img.data[o] = lut[v * 3]
      img.data[o + 1] = lut[v * 3 + 1]
      img.data[o + 2] = lut[v * 3 + 2]
      img.data[o + 3] = 255
    }
  }
  return img
}

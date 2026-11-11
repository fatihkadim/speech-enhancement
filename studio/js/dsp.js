/* SesTemiz: tarayıcı tarafı ses yardımcıları: çözme, 16 kHz'e indirme, WAV kodlama, FFT/spektrogram, tepe değerleri. */
window.DSP = (() => {
  const SR = 16000;
  let _ctx = null;
  const ctx = () => (_ctx ??= new (window.AudioContext || window.webkitAudioContext)());

  /** Herhangi bir ses dosyasını (wav/mp3/ogg/webm…) mono 16 kHz Float32Array'e çevirir. */
  async function decodeToMono16k(arrayBuffer) {
    // 16 kHz'lik bir bağlamda çözmek tek adımda 16 kHz verir; dosya zaten 16 kHz ise yeniden örnekleme
    // hiç yapılmaz (modelin gördüğü ses, dosyadakiyle örnek örnek aynı kalır).
    const decoded = await new OfflineAudioContext(1, 1, SR).decodeAudioData(arrayBuffer.slice(0));
    const n = decoded.numberOfChannels, out = new Float32Array(decoded.length);
    for (let c = 0; c < n; c++) {
      const ch = decoded.getChannelData(c);
      for (let i = 0; i < out.length; i++) out[i] += ch[i] / n; // sunucudaki gibi kanal ortalaması
    }
    return out;
  }

  /** Float32 [-1,1] -> 16-bit PCM mono WAV Blob. */
  function encodeWav(samples, sr = SR) {
    const buf = new ArrayBuffer(44 + samples.length * 2);
    const v = new DataView(buf);
    const str = (o, s) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
    str(0, "RIFF"); v.setUint32(4, 36 + samples.length * 2, true); str(8, "WAVE");
    str(12, "fmt "); v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
    v.setUint32(24, sr, true); v.setUint32(28, sr * 2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true);
    str(36, "data"); v.setUint32(40, samples.length * 2, true);
    for (let i = 0, o = 44; i < samples.length; i++, o += 2) {
      // soundfile/libsndfile ile aynı ölçek (×32768): 16-bit dosya → tarayıcı → sunucu yolu kayıpsız
      v.setInt16(o, Math.max(-32768, Math.min(32767, Math.round(samples[i] * 32768))), true);
    }
    return new Blob([buf], { type: "audio/wav" });
  }

  /** Yerinde, iteratif radix-2 FFT. */
  function fft(re, im) {
    const n = re.length;
    for (let i = 1, j = 0; i < n; i++) {
      let bit = n >> 1;
      for (; j & bit; bit >>= 1) j ^= bit;
      j ^= bit;
      if (i < j) { [re[i], re[j]] = [re[j], re[i]]; [im[i], im[j]] = [im[j], im[i]]; }
    }
    for (let len = 2; len <= n; len <<= 1) {
      const ang = (-2 * Math.PI) / len, wr = Math.cos(ang), wi = Math.sin(ang);
      for (let i = 0; i < n; i += len) {
        let cr = 1, ci = 0;
        for (let k = 0; k < len / 2; k++) {
          const a = i + k, b = a + len / 2;
          const tr = re[b] * cr - im[b] * ci, ti = re[b] * ci + im[b] * cr;
          re[b] = re[a] - tr; im[b] = im[a] - ti; re[a] += tr; im[a] += ti;
          const ncr = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = ncr;
        }
      }
    }
  }

  /** Modelle aynı STFT ayarları (n_fft=512, hop=128): dB cinsinden (frames × bins) genlik. */
  function spectrogram(samples, nfft = 512, hop = 128) {
    const bins = nfft / 2;
    const frames = Math.max(1, Math.floor((samples.length - nfft) / hop) + 1);
    const win = new Float32Array(nfft).map((_, i) => 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / nfft));
    const out = new Float32Array(frames * bins);
    const re = new Float32Array(nfft), im = new Float32Array(nfft);
    let max = -Infinity;
    for (let f = 0; f < frames; f++) {
      const off = f * hop;
      for (let i = 0; i < nfft; i++) { re[i] = (samples[off + i] || 0) * win[i]; im[i] = 0; }
      fft(re, im);
      for (let b = 0; b < bins; b++) {
        const db = 10 * Math.log10(re[b] * re[b] + im[b] * im[b] + 1e-10);
        out[f * bins + b] = db;
        if (db > max) max = db;
      }
    }
    return { data: out, frames, bins, max };
  }

  /** Görselleştirme için n kovaya bölünmüş tepe genlikleri (0..1). */
  function peaks(samples, n) {
    const out = new Float32Array(n);
    const step = samples.length / n;
    let gmax = 1e-6;
    for (let i = 0; i < n; i++) {
      let m = 0;
      const a = Math.floor(i * step), b = Math.min(samples.length, Math.floor((i + 1) * step));
      for (let j = a; j < b; j++) { const v = Math.abs(samples[j]); if (v > m) m = v; }
      out[i] = m; if (m > gmax) gmax = m;
    }
    return { values: out, max: gmax };
  }

  function rms(samples) {
    let s = 0;
    for (let i = 0; i < samples.length; i++) s += samples[i] * samples[i];
    return Math.sqrt(s / Math.max(1, samples.length));
  }

  return { SR, ctx, decodeToMono16k, encodeWav, spectrogram, peaks, rms };
})();

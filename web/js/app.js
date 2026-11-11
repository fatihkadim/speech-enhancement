/* SesTemiz: arayüz mantığı: yönlendirme, tema/dil, animasyonlar, stüdyo, örnekler, sonuçlar. */
(() => {
  "use strict";

  // ------------------------------------------------------------------ yardımcılar
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];
  const html = document.documentElement;
  const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const wait = (ms) => new Promise((r) => setTimeout(r, ms));
  const store = {
    get(k) { try { return localStorage.getItem(k); } catch { return null; } },
    set(k, v) { try { localStorage.setItem(k, v); } catch { /* yoksay */ } },
  };
  const cssVar = (n) => getComputedStyle(html).getPropertyValue(n).trim();
  const lang = () => (html.lang === "en" ? "en" : "tr");
  const t = (k) => I18N[lang()][k] ?? I18N.tr[k] ?? k;
  const fmt = (n, dec = 2, signed = false) => {
    if (n == null || Number.isNaN(n)) return "·";
    let s = Math.abs(n).toFixed(dec);
    if (lang() === "tr") s = s.replace(".", ",");
    const sign = n < 0 && Number(s.replace(",", ".")) !== 0 ? "−" : signed && Number(s.replace(",", ".")) !== 0 ? "+" : "";
    return sign + s;
  };
  const mmss = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
  const hexToRgb = (h) => {
    h = h.replace("#", "");
    if (h.length === 3) h = [...h].map((c) => c + c).join("");
    const n = parseInt(h, 16);
    return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
  };
  const fitCanvas = (c) => {
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const w = Math.max(1, Math.round(c.clientWidth * dpr)), h = Math.max(1, Math.round(c.clientHeight * dpr));
    if (c.width !== w || c.height !== h) { c.width = w; c.height = h; }
    return { w, h, dpr, g: c.getContext("2d") };
  };

  function toast(msg, type = "") {
    const el = document.createElement("div");
    el.className = `toast ${type}`;
    el.textContent = msg;
    $("#toasts").append(el);
    setTimeout(() => { el.classList.add("out"); setTimeout(() => el.remove(), 400); }, 3200);
  }

  // ------------------------------------------------------------------ veri
  const SAMPLE_BASE = "../samples/";
  const SAMPLES = [
    { id: "snr-5_hen", snr: -5, noise: "hen" },
    { id: "snr+0_hen", snr: 0, noise: "hen" },
    { id: "snr+5_sheep", snr: 5, noise: "sheep" },
    { id: "snr+10_mouse_click", snr: 10, noise: "mouse_click" },
    { id: "snr+15_train", snr: 15, noise: "train" },
  ];
  const sampleUrl = (id, kind) => `${SAMPLE_BASE}${id.split("/").map(encodeURIComponent).join("/")}_${kind}.wav`;
  // Türkçe örnekler: src/make_tr_samples.py'nin ürettiği samples/tr/denemeler.json'dan yüklenir
  let TR = null;
  const trReady = fetch(`${SAMPLE_BASE}tr/denemeler.json`, { cache: "no-cache" })
    .then((r) => (r.ok ? r.json() : null))
    .then((j) => {
      if (j?.items?.length) TR = { ...j, items: j.items.map((it) => ({ ...it, id: `tr/${it.id}`, lang: "tr" })) };
      return TR;
    })
    .catch(() => null);
  const findSample = (id) => SAMPLES.find((s) => s.id === id) || TR?.items.find((s) => s.id === id);
  const sampleName = (id) => id.split("/").pop();
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
  trReady.then(() => TR?.items.forEach((it) => { it.text = esc(it.text || ""); }));

  const EXPS = ["baseline", "e1_mse", "e1_si_snr", "e2_no_skip", "e3_depth3", "e3_depth5", "e4_direct"];
  // [ortalama, std] ve baseline'a fark (Δ). Kaynak: docs/results.md
  const R = {
    test: {
      e0: { si_snr: 4.50, pesq: 1.367, stoi: 0.833 },
      baseline: { si_snr: [12.78, .27], pesq: [1.970, .042], stoi: [.891, .004] },
      e1_mse: { si_snr: [12.55, .59], pesq: [1.945, .033], stoi: [.895, .005], d: [-0.23, -0.025, 0.004] },
      e1_si_snr: { si_snr: [13.49, .20], pesq: [1.942, .007], stoi: [.892, .002], d: [0.71, -0.028, 0.001] },
      e2_no_skip: { si_snr: [11.60, .26], pesq: [1.802, .031], stoi: [.874, .004], d: [-1.18, -0.168, -0.017] },
      e3_depth3: { si_snr: [12.50, 0], pesq: [1.917, .015], stoi: [.886, .001], d: [-0.28, -0.054, -0.006] },
      e3_depth5: { si_snr: [13.14, .14], pesq: [2.043, .035], stoi: [.897, .002], d: [0.36, 0.072, 0.006] },
      e4_direct: { si_snr: [12.50, .26], pesq: [1.947, .037], stoi: [.891, .003], d: [-0.28, -0.024, -0.001] },
    },
    unseen: {
      e0: { si_snr: 5.03, pesq: 1.385, stoi: 0.832 },
      baseline: { si_snr: [13.16, .28], pesq: [2.075, .050], stoi: [.891, .003] },
      e1_mse: { si_snr: [12.85, .55], pesq: [2.030, .037], stoi: [.894, .006], d: [-0.31, -0.045, 0.003] },
      e1_si_snr: { si_snr: [13.99, .02], pesq: [2.050, .005], stoi: [.892, .002], d: [0.83, -0.025, 0.001] },
      e2_no_skip: { si_snr: [11.58, .29], pesq: [1.831, .027], stoi: [.873, .004], d: [-1.58, -0.244, -0.018] },
      e3_depth3: { si_snr: [12.84, 0], pesq: [2.028, .017], stoi: [.884, .001], d: [-0.32, -0.047, -0.007] },
      e3_depth5: { si_snr: [13.45, .14], pesq: [2.157, .031], stoi: [.898, .001], d: [0.29, 0.082, 0.006] },
      e4_direct: { si_snr: [12.75, .23], pesq: [2.035, .035], stoi: [.890, .002], d: [-0.41, -0.040, -0.002] },
    },
  };
  const SNRS = [-5, 0, 5, 10, 15];
  const E5 = {
    baseline: [12.64, 11.03, 7.70, 6.12, 2.79], e1_mse: [12.81, 10.83, 7.68, 5.44, 2.20],
    e1_si_snr: [13.08, 11.78, 8.46, 6.48, 3.97], e2_no_skip: [11.48, 9.51, 6.31, 4.94, 2.16],
    e3_depth3: [12.16, 10.76, 7.42, 5.69, 2.85], e3_depth5: [13.07, 11.46, 8.14, 6.15, 3.18],
    e4_direct: [12.50, 10.85, 7.52, 5.81, 2.18],
  };
  const EXP_COLOR = {
    baseline: "var(--ink)", e1_mse: "var(--accent-3)", e1_si_snr: "var(--accent-2)", e2_no_skip: "var(--muted)",
    e3_depth3: "var(--line-2)", e3_depth5: "var(--accent)", e4_direct: "color-mix(in oklab, var(--accent-2) 45%, var(--accent-3))",
  };
  const METRIC_DOMAIN = { si_snr: [0, 15], pesq: [1, 2.3], stoi: [.8, .91] };
  const METRIC_DEC = { si_snr: 2, pesq: 3, stoi: 3 };
  const DEFAULT_MODELS = [{ id: "e3_depth5", local: false }, { id: "e1_si_snr", local: false }];

  // ------------------------------------------------------------------ durum
  const app = {
    server: { state: "checking", device: null, models: DEFAULT_MODELS },
    studio: { input: null, output: null, model: "e3_depth5", side: "b", split: 50, pendingSample: null },
    results: { split: "test", metric: "si_snr" },
    route: null,
    ready: Promise.resolve(),
    cleanups: [],
    onLang: null,
    onTheme: null,
  };

  // ------------------------------------------------------------------ i18n
  function applyI18n(root = document, animate = false) {
    $$("[data-i18n]", root).forEach((el) => { el.textContent = t(el.dataset.i18n); if (animate) swap(el); });
    $$("[data-i18n-html]", root).forEach((el) => { el.innerHTML = t(el.dataset.i18nHtml); if (animate) swap(el); });
    $$("[data-i18n-aria]", root).forEach((el) => el.setAttribute("aria-label", t(el.dataset.i18nAria)));
    if (root === document) document.title = t("meta.title");
  }
  function swap(el) {
    if (reduced || !el.offsetParent) return;
    el.classList.remove("i18n-swap"); void el.offsetWidth; el.classList.add("i18n-swap");
  }
  function setLang(l) {
    if (l === lang()) return;
    html.lang = l;
    store.set("st-lang", l);
    applyI18n(document, true);
    langSeg.set(l);
    app.onLang?.();
    updateStatusPill();
    requestAnimationFrame(positionNavPill);
  }

  // ------------------------------------------------------------------ segment anahtar
  function seg(el, attr, value, onChange) {
    const thumb = $(".seg-thumb", el);
    const btns = $$("button", el);
    const set = (v, quiet) => {
      btns.forEach((b) => b.classList.toggle("on", b.dataset[attr] === v));
      const on = btns.find((b) => b.dataset[attr] === v);
      if (on && thumb) { thumb.style.width = `${on.offsetWidth}px`; thumb.style.transform = `translateX(${on.offsetLeft}px)`; }
      if (!quiet) onChange?.(v);
    };
    btns.forEach((b) => b.addEventListener("click", () => set(b.dataset[attr])));
    set(value, true);
    const ro = new ResizeObserver(() => set(btns.find((b) => b.classList.contains("on"))?.dataset[attr], true));
    ro.observe(el);
    return { set: (v) => set(v, true), el };
  }

  // ------------------------------------------------------------------ tema
  function setTheme(next, origin) {
    const run = () => {
      html.dataset.theme = next;
      store.set("st-theme", next);
      app.onTheme?.();
    };
    if (!document.startViewTransition || reduced) return run();
    html.dataset.vt = "theme";
    const vt = document.startViewTransition(run);
    vt.ready.then(() => {
      const { x, y } = origin;
      const r = Math.hypot(Math.max(x, innerWidth - x), Math.max(y, innerHeight - y));
      html.animate({ clipPath: [`circle(0px at ${x}px ${y}px)`, `circle(${r}px at ${x}px ${y}px)`] },
        { duration: 750, easing: "cubic-bezier(.77,0,.175,1)", pseudoElement: "::view-transition-new(root)" });
    }).catch(() => {});
    vt.finished.catch(() => {}).finally(() => delete html.dataset.vt);
  }
  $("#themeBtn").addEventListener("click", (e) => {
    const r = e.currentTarget.getBoundingClientRect();
    setTheme(html.dataset.theme === "dark" ? "light" : "dark", { x: r.left + r.width / 2, y: r.top + r.height / 2 });
  });

  const langSeg = seg($("#langSeg"), "lang", lang(), setLang);

  // ------------------------------------------------------------------ gezinme çubuğu
  const nav = $("#nav"), navPill = $("#navPill"), menuBtn = $("#menuBtn");
  function positionNavPill() {
    const a = $("a.active", nav);
    if (!a || getComputedStyle(navPill).display === "none") { navPill.style.opacity = 0; return; }
    navPill.style.opacity = 1;
    navPill.style.width = `${a.offsetWidth}px`;
    navPill.style.transform = `translateX(${a.offsetLeft}px)`;
  }
  new ResizeObserver(positionNavPill).observe(nav);
  menuBtn.addEventListener("click", () => {
    const open = !nav.classList.contains("open");
    nav.classList.toggle("open", open);
    menuBtn.setAttribute("aria-expanded", open);
  });
  addEventListener("scroll", () => $("#topbar").classList.toggle("scrolled", scrollY > 8), { passive: true });
  $("#toTop").addEventListener("click", () => scrollTo({ top: 0, behavior: reduced ? "auto" : "smooth" }));

  // ------------------------------------------------------------------ mıknatıslı butonlar & kart ışığı
  // Giriş animasyonu bitince kalıcı `filter: blur(0)` katmanını bırak (görünüm aynı, her karede daha az iş)
  addEventListener("animationend", (e) => {
    if (e.animationName === "reveal" && e.target.hasAttribute?.("data-reveal")) e.target.classList.add("revealed");
  });
  let magnet = null;
  const ambient = $(".ambient");
  addEventListener("pointermove", (e) => {
    if (!reduced) { // arka plan ışığı imleci yavaşça izler; değişken yalnızca .ambient'e yazılır (ucuz)
      ambient.style.setProperty("--px", (e.clientX / innerWidth - 0.5).toFixed(3));
      ambient.style.setProperty("--py", (e.clientY / innerHeight - 0.5).toFixed(3));
    }
    const card = e.target.closest?.(".card");
    if (card) {
      const r = card.getBoundingClientRect();
      card.style.setProperty("--mx", `${e.clientX - r.left}px`);
      card.style.setProperty("--my", `${e.clientY - r.top}px`);
    }
    const btn = e.target.closest?.(".btn");
    if (btn) {
      const r = btn.getBoundingClientRect();
      btn.style.setProperty("--x", `${e.clientX - r.left}px`);
      btn.style.setProperty("--y", `${e.clientY - r.top}px`);
    }
    if (e.pointerType !== "mouse" || reduced) return;
    const m = e.target.closest?.("[data-magnetic]");
    if (magnet && magnet !== m) { magnet.style.transform = ""; magnet = null; }
    if (m && !m.disabled) {
      magnet = m;
      const r = m.getBoundingClientRect();
      const dx = e.clientX - (r.left + r.width / 2), dy = e.clientY - (r.top + r.height / 2);
      m.style.transform = `translate(${dx * 0.18}px, ${dy * 0.28}px)`;
    }
  }, { passive: true });
  document.addEventListener("pointerleave", () => { if (magnet) magnet.style.transform = ""; magnet = null; });

  document.addEventListener("click", async (e) => {
    const c = e.target.closest("[data-copy]");
    if (!c) return;
    try { await navigator.clipboard.writeText(c.dataset.copy); toast(t("toast.copied")); } catch { /* yoksay */ }
  });

  // ------------------------------------------------------------------ sunucu durumu
  const statusEl = $("#status");
  function updateStatusPill() {
    if (!statusEl) return; // durum noktası arayüzden kaldırıldı; sunucu kontrolü arka planda sürer
    const s = app.server.state;
    statusEl.dataset.state = s;
    const label = s === "online" ? `${t("status.online")}${app.server.device ? " · " + app.server.device : ""}` : t(`status.${s}`);
    $("span", statusEl).textContent = label; // görünmez; ekran okuyucu ve ipucu için
    statusEl.title = label;
  }
  async function checkServer() {
    const prev = app.server.state;
    try {
      const ctl = new AbortController();
      const timer = setTimeout(() => ctl.abort(), 20000);
      const res = await fetch("/api/status", { signal: ctl.signal, cache: "no-store" });
      clearTimeout(timer);
      if (!res.ok) throw new Error(res.status);
      const j = await res.json();
      app.server = { state: "online", device: j.device, models: j.models?.length ? j.models : DEFAULT_MODELS };
    } catch {
      app.server = { state: "offline", device: null, models: DEFAULT_MODELS };
    }
    updateStatusPill();
    if (prev !== app.server.state) app.onServer?.();
  }

  // ------------------------------------------------------------------ yönlendirici
  const ORDER = ["", "studio", "samples", "results", "method", "about"];
  const PAGES = { "": initHome, studio: initStudio, samples: initSamples, results: initResults, method: () => {}, about: () => {} };
  const view = $("#view");
  const parseRoute = () => {
    const r = location.hash.replace(/^#\/?/, "").split(/[/?]/)[0];
    return r in PAGES ? r : "";
  };

  function mount(route) {
    app.mounted = route;
    app.cleanups.forEach((f) => f()); app.cleanups = [];
    app.onLang = app.onTheme = app.onServer = null;
    view.classList.remove("in");
    view.replaceChildren($(`#tpl-${route || "home"}`).content.cloneNode(true));
    applyI18n(view);
    $$("[data-reveal]", view).forEach((el, i) => el.style.setProperty("--i", Math.min(i, 12)));
    $$("a[data-route]", nav).forEach((a) => a.classList.toggle("active", a.dataset.route === route));
    positionNavPill();
    nav.classList.remove("open"); menuBtn.setAttribute("aria-expanded", "false");
    scrollTo({ top: 0, behavior: "instant" });
    PAGES[route](view);
    app.ready.then(() => requestAnimationFrame(() => view.classList.add("in")));
  }

  async function navigate(initial = false) {
    const route = parseRoute();
    if (route === app.route && !initial) return;
    const dir = ORDER.indexOf(route) < ORDER.indexOf(app.route ?? "") ? "back" : "forward";
    app.route = route;
    if (initial || reduced) return mount(route);
    if (document.startViewTransition) {
      html.dataset.vt = "page"; html.dataset.dir = dir;
      const vt = document.startViewTransition(() => mount(route));
      vt.ready.catch(() => {});
      vt.finished.catch(() => {}).finally(() => {
        delete html.dataset.vt; delete html.dataset.dir;
        if (app.mounted !== app.route) mount(app.route); // geçiş iptal edildiyse sayfayı yine de kur
      });
    } else {
      view.classList.add("leaving");
      await wait(320);
      view.classList.remove("leaving");
      mount(route);
    }
  }
  addEventListener("hashchange", () => navigate());

  // ------------------------------------------------------------------ ANA SAYFA
  function initHome(root) {
    // sayaçlar
    const io = new IntersectionObserver((entries) => entries.forEach((en) => {
      if (!en.isIntersecting) return;
      io.unobserve(en.target);
      const el = en.target, to = +el.dataset.count, dec = +el.dataset.dec;
      const pre = el.dataset.prefix || "", suf = el.dataset.suffix || "";
      const t0 = performance.now(), dur = reduced ? 1 : 1600;
      const step = (now) => {
        const p = Math.min(1, (now - t0) / dur), e = 1 - Math.pow(1 - p, 4);
        el.textContent = pre + fmt(to * e, dec) + suf;
        if (p < 1) requestAnimationFrame(step);
      };
      requestAnimationFrame(step);
    }), { threshold: 0.4 });
    $$("[data-count]", root).forEach((el) => io.observe(el));
    app.cleanups.push(() => io.disconnect());
    app.onLang = () => $$("[data-count]", root).forEach((el) => {
      el.textContent = (el.dataset.prefix || "") + fmt(+el.dataset.count, +el.dataset.dec) + (el.dataset.suffix || "");
    });

    // kayan şerit
    const noises = ["dog", "rooster", "rain", "sea waves", "crackling fire", "crickets", "chirping birds", "water drops", "wind", "pouring water",
      "toilet flush", "thunderstorm", "crying baby", "sneezing", "clapping", "breathing", "coughing", "footsteps", "laughing", "brushing teeth",
      "snoring", "door knock", "mouse click", "keyboard typing", "door wood creaks", "can opening", "washing machine", "vacuum cleaner",
      "clock alarm", "clock tick", "glass breaking", "helicopter", "chainsaw", "siren", "car horn", "engine", "train", "church bells", "airplane", "fireworks", "hand saw"];
    const track = $("#marquee", root);
    const items = noises.map((n) => `<span>${n}</span>`).join("");
    track.innerHTML = items + items;

    // kahraman tuval: gürültülü ⟷ temiz
    const cv = $("#heroCanvas", root), vu = $$("#heroVu i", root);
    let split = 0.5, target = 0.5, pointer = false, raf = 0, visible = true, colors = null;
    const readColors = () => (colors = { a: cssVar("--accent"), n: cssVar("--ink-2"), l: cssVar("--line-2"), ink: cssVar("--ink"), bg: cssVar("--bg") });
    readColors();
    app.onTheme = () => requestAnimationFrame(readColors);
    const setFromEvent = (e) => { const r = cv.getBoundingClientRect(); target = Math.min(.96, Math.max(.04, (e.clientX - r.left) / r.width)); };
    cv.addEventListener("pointerdown", (e) => { pointer = true; cv.setPointerCapture(e.pointerId); setFromEvent(e); });
    cv.addEventListener("pointermove", (e) => { if (pointer || e.pointerType === "mouse") { pointer = pointer || e.pointerType === "mouse"; setFromEvent(e); } });
    cv.addEventListener("pointerup", () => { pointer = false; });
    cv.addEventListener("pointerleave", (e) => { if (e.pointerType === "mouse") pointer = false; });
    const vis = new IntersectionObserver(([en]) => { visible = en.isIntersecting; }, { threshold: 0 });
    vis.observe(cv);

    const hash = (x) => { const s = Math.sin(x * 127.1) * 43758.5453; return s - Math.floor(s); };
    const vnoise = (x) => { const i = Math.floor(x), f = x - i, u = f * f * (3 - 2 * f); return hash(i) * (1 - u) + hash(i + 1) * u; };
    const speech = (x) => { // kelime boşlukları + hece patlamaları
      const gate = Math.min(1, Math.max(0, (vnoise(x * 0.045) - 0.28) * 4));
      const syll = Math.pow(Math.abs(Math.sin(x * 0.33 + vnoise(x * 0.07) * 3)), 1.4);
      return gate * (0.25 + 0.75 * syll) * (0.55 + 0.45 * vnoise(x * 0.11 + 7)) * (0.85 + 0.15 * hash(Math.floor(x)));
    };
    const frame = (now) => {
      raf = requestAnimationFrame(frame);
      if (!visible) return;
      const { w, h, dpr, g } = fitCanvas(cv);
      const time = now / 1000;
      if (!pointer) target = 0.5 + 0.28 * Math.sin(time * 0.45);
      split += (target - split) * (pointer ? 0.25 : 0.05);
      g.clearRect(0, 0, w, h);
      const bw = 3 * dpr, gap = 2.2 * dpr, n = Math.floor(w / (bw + gap)), mid = h / 2;
      const scroll = time * 9, sx = split * w;
      let vuLevel = 0;
      for (let i = 0; i < n; i++) {
        const x = i * (bw + gap), p = i + scroll;
        const clean = speech(p);
        const noise = (0.25 + 0.3 * vnoise(p * 0.15 + time)) * (0.45 + 0.55 * hash(Math.floor(p) + Math.floor(time * 14)));
        const isClean = x > sx;
        const amp = isClean ? clean : Math.min(1, clean * 0.9 + noise * 0.5);
        const bh = Math.max(1.5 * dpr, amp * (h * 0.42));
        g.globalAlpha = isClean ? 1 : 0.55;
        g.fillStyle = isClean ? colors.a : colors.n;
        g.beginPath();
        g.roundRect ? g.roundRect(x, mid - bh, bw, bh * 2, bw / 2) : g.rect(x, mid - bh, bw, bh * 2);
        g.fill();
        if (Math.abs(x - sx) < bw + gap) vuLevel = clean;
      }
      g.globalAlpha = 1;
      // ayırıcı
      g.fillStyle = colors.ink;
      g.fillRect(sx - dpr * 0.75, 0, dpr * 1.5, h);
      g.beginPath(); g.arc(sx, mid, 13 * dpr, 0, Math.PI * 2); g.fill();
      g.fillStyle = colors.bg;
      g.font = `${11 * dpr}px "Martian Mono", monospace`; g.textAlign = "center"; g.textBaseline = "middle";
      g.fillText("⟷", sx, mid + dpr * 0.5);
      // VU
      const lit = Math.round(vuLevel * vu.length * (0.85 + 0.15 * Math.sin(time * 20)));
      vu.forEach((el, i) => el.classList.toggle("on", i < lit));
    };
    raf = requestAnimationFrame(frame);
    app.cleanups.push(() => { cancelAnimationFrame(raf); vis.disconnect(); });
  }

  // ------------------------------------------------------------------ A/B OYNATICI
  const player = {
    a: null, b: null, srcs: [], gains: null, playing: false, startAt: 0, offset: 0, side: "b", onEnd: null,
    get duration() { return this.a ? this.a.duration : 0; },
    load(a, b) {
      this.stop();
      const ctx = DSP.ctx();
      const mk = (arr) => { const buf = ctx.createBuffer(1, arr.length, DSP.SR); buf.copyToChannel(arr, 0); return buf; };
      this.a = mk(a); this.b = mk(b); this.offset = 0;
    },
    time() {
      if (!this.playing) return this.offset;
      return Math.min(this.duration, this.offset + DSP.ctx().currentTime - this.startAt);
    },
    play() {
      if (!this.a || this.playing) return;
      const ctx = DSP.ctx(); ctx.resume();
      if (this.offset >= this.duration - 0.02) this.offset = 0;
      const ga = ctx.createGain(), gb = ctx.createGain();
      ga.gain.value = this.side === "a" ? 1 : 0; gb.gain.value = this.side === "b" ? 1 : 0;
      ga.connect(ctx.destination); gb.connect(ctx.destination);
      this.srcs = [[this.a, ga], [this.b, gb]].map(([buf, g]) => {
        const s = ctx.createBufferSource(); s.buffer = buf; s.connect(g); return s;
      });
      const when = ctx.currentTime + 0.03;
      this.srcs.forEach((s) => s.start(when, this.offset));
      this.srcs[0].onended = () => { if (this.playing && this.time() >= this.duration - 0.05) { this.stop(); this.offset = 0; this.onEnd?.(); } };
      this.gains = [ga, gb]; this.startAt = when; this.playing = true;
    },
    pause() { if (!this.playing) return; this.offset = this.time(); this.stop(); },
    stop() {
      this.srcs.forEach((s) => { s.onended = null; try { s.stop(); } catch { /* zaten durdu */ } s.disconnect(); });
      this.gains?.forEach((g) => g.disconnect());
      this.srcs = []; this.gains = null; this.playing = false;
    },
    seek(sec) {
      const was = this.playing;
      if (was) this.stop();
      this.offset = Math.max(0, Math.min(this.duration, sec));
      if (was) this.play();
    },
    setSide(s) {
      this.side = s;
      if (!this.gains) return;
      const now = DSP.ctx().currentTime;
      this.gains[0].gain.setTargetAtTime(s === "a" ? 1 : 0, now, 0.012);
      this.gains[1].gain.setTargetAtTime(s === "b" ? 1 : 0, now, 0.012);
    },
  };

  // ------------------------------------------------------------------ çizim yardımcıları
  function drawWave(canvas, layers, progress = -1) {
    const { w, h, dpr, g } = fitCanvas(canvas);
    g.clearRect(0, 0, w, h);
    const bw = 2 * dpr, gap = 1.4 * dpr, n = Math.max(1, Math.floor(w / (bw + gap))), mid = h / 2;
    const px = progress >= 0 ? progress * w : -1;
    for (const L of layers) {
      const { values, max } = DSP.peaks(L.samples, n);
      const norm = L.norm ?? max;
      for (let i = 0; i < n; i++) {
        const x = i * (bw + gap);
        const bh = Math.max(dpr, (values[i] / norm) * (h * 0.46));
        const played = px >= 0 && x <= px;
        g.globalAlpha = L.alpha * (px >= 0 && !played ? 0.55 : 1);
        g.fillStyle = played && L.played ? L.played : L.color;
        g.fillRect(x, mid - bh, bw, bh * 2);
      }
    }
    g.globalAlpha = 1;
  }

  function specLut() {
    const stops = ["--spec-0", "--spec-1", "--spec-2", "--spec-3", "--spec-4"].map((v) => hexToRgb(cssVar(v)));
    const pos = [0, 0.35, 0.6, 0.83, 1];
    const lut = new Uint8ClampedArray(256 * 3);
    for (let i = 0; i < 256; i++) {
      const x = i / 255;
      let k = 0; while (k < pos.length - 2 && x > pos[k + 1]) k++;
      const f = (x - pos[k]) / (pos[k + 1] - pos[k]);
      for (let c = 0; c < 3; c++) lut[i * 3 + c] = stops[k][c] + (stops[k + 1][c] - stops[k][c]) * f;
    }
    return lut;
  }

  function drawSpec(canvas, spec, refMax) {
    const W = Math.min(spec.frames, 1400), H = spec.bins;
    canvas.width = W; canvas.height = H;
    const g = canvas.getContext("2d");
    const img = g.createImageData(W, H);
    const lut = specLut(), range = 80, top = refMax;
    const per = spec.frames / W;
    for (let x = 0; x < W; x++) {
      const f0 = Math.floor(x * per), f1 = Math.max(f0 + 1, Math.floor((x + 1) * per));
      for (let b = 0; b < H; b++) {
        let v = -Infinity;
        for (let f = f0; f < f1; f++) { const d = spec.data[f * H + b]; if (d > v) v = d; }
        const idx = Math.max(0, Math.min(255, Math.round(((v - (top - range)) / range) * 255)));
        const o = ((H - 1 - b) * W + x) * 4;
        img.data[o] = lut[idx * 3]; img.data[o + 1] = lut[idx * 3 + 1]; img.data[o + 2] = lut[idx * 3 + 2]; img.data[o + 3] = 255;
      }
    }
    g.putImageData(img, 0, 0);
  }

  // ------------------------------------------------------------------ STÜDYO
  function initStudio(root) {
    const S = app.studio;
    const runBtn = $("#runBtn", root);
    const els = {
      empty: $("#inputEmpty", root), loaded: $("#inputLoaded", root), drop: $("#drop", root), file: $("#fileInput", root),
      rec: $("#recBtn", root), recMeter: $("#recMeter", root), recCanvas: $("#recCanvas", root), recTime: $("#recTime", root),
      chips: $("#sampleChips", root), models: $("#models", root), outEmpty: $("#outEmpty", root), out: $("#out", root),
      banner: $("#offlineBanner", root),
    };

    // --- örnek çipleri
    const chip = (s) => `<button type="button" class="chip" data-id="${s.id}"><b>${fmt(s.snr, 0, true)} dB</b>${t("noise." + s.noise)}</button>`;
    const renderChips = () => {
      els.chips.innerHTML = SAMPLES.map(chip).join("");
      $("#trChipGroup", root).hidden = !TR;
      if (TR) $("#trChips", root).innerHTML = TR.items.map(chip).join("");
    };
    renderChips();
    trReady.then(renderChips);
    $(".sample-pick", root).addEventListener("click", (e) => { const b = e.target.closest(".chip"); if (b) loadSample(b.dataset.id); });

    // --- modeller
    const renderModels = () => {
      const list = app.server.models;
      if (!list.some((m) => m.id === S.model)) S.model = list[0].id;
      els.models.innerHTML = list.map((m) => `
        <button type="button" class="model" role="radio" data-id="${m.id}" aria-checked="${m.id === S.model}">
          <span class="radio"></span>
          <span><strong>${t("model." + m.id)}</strong><small>${t("model." + m.id + ".d")}</small></span>
          <span class="tag ${m.local ? "local" : ""}">${m.local ? t("model.local") : t("model.remote")}</span>
        </button>`).join("");
    };
    renderModels();
    els.models.addEventListener("click", (e) => {
      const b = e.target.closest(".model"); if (!b) return;
      S.model = b.dataset.id;
      $$(".model", els.models).forEach((m) => m.setAttribute("aria-checked", m === b));
    });

    const syncRun = () => {
      const busy = runBtn.classList.contains("busy");
      runBtn.disabled = !S.input || app.server.state !== "online" || busy;
      $(".run-label", runBtn).textContent = busy ? t("studio.running") : S.output ? t("studio.again") : t("studio.run");
      els.banner.hidden = app.server.state !== "offline";
    };

    // --- girdi
    const showInput = () => {
      els.empty.hidden = !!S.input;
      els.loaded.hidden = !S.input;
      if (S.input) {
        $("#fileName", root).textContent = S.input.name;
        $("#fileInfo", root).textContent = `${mmss(S.input.samples.length / DSP.SR)} · 16 kHz · mono${S.input.ref ? " · " + t("studio.ref") + " ✓" : ""}`;
        requestAnimationFrame(() => drawWave($("#inputWave", root), [{ samples: S.input.samples, color: cssVar("--ink-2"), alpha: 1 }]));
      }
      syncRun();
    };
    const setInput = (samples, name, ref = null) => {
      const max = 120 * DSP.SR;
      if (samples.length > max) { toast(t("toast.long")); samples = samples.slice(0, max); }
      S.input = { samples, name, ref };
      S.output = null;
      player.stop();
      showInput(); showOutput();
    };
    async function loadFile(file) {
      try { setInput(await DSP.decodeToMono16k(await file.arrayBuffer()), file.name); }
      catch { toast(t("toast.decode"), "err"); }
    }
    async function loadSample(id) {
      try {
        const buf = await (await fetch(sampleUrl(id, "noisy"))).arrayBuffer();
        setInput(await DSP.decodeToMono16k(buf), `${sampleName(id)}_noisy.wav`, id);
      } catch { toast(t("toast.decode"), "err"); }
    }
    els.file.addEventListener("change", () => { if (els.file.files[0]) loadFile(els.file.files[0]); els.file.value = ""; });
    ["dragenter", "dragover"].forEach((ev) => els.drop.addEventListener(ev, (e) => { e.preventDefault(); els.drop.classList.add("over"); }));
    ["dragleave", "drop"].forEach((ev) => els.drop.addEventListener(ev, (e) => { e.preventDefault(); els.drop.classList.remove("over"); }));
    els.drop.addEventListener("drop", (e) => { const f = e.dataTransfer.files[0]; if (f) loadFile(f); });
    $("#replaceBtn", root).addEventListener("click", () => { S.input = null; S.output = null; player.stop(); showInput(); showOutput(); });

    // --- mikrofon kaydı
    let rec = null;
    els.rec.addEventListener("click", async () => {
      if (rec) { rec.stop(); return; }
      let stream;
      try { stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false } }); }
      catch { toast(t("toast.recDenied"), "err"); return; }
      const ctx = DSP.ctx(); ctx.resume();
      const an = ctx.createAnalyser(); an.fftSize = 1024;
      const srcNode = ctx.createMediaStreamSource(stream); srcNode.connect(an);
      const mr = new MediaRecorder(stream), chunks = [], hist = [], t0 = performance.now();
      const buf = new Float32Array(an.fftSize);
      let raf = 0;
      const draw = () => {
        raf = requestAnimationFrame(draw);
        an.getFloatTimeDomainData(buf);
        hist.push(Math.min(1, DSP.rms(buf) * 5)); if (hist.length > 90) hist.shift();
        const { w, h, dpr, g } = fitCanvas(els.recCanvas);
        g.clearRect(0, 0, w, h); g.fillStyle = cssVar("--accent");
        const bw = w / 90;
        hist.forEach((v, i) => { const bh = Math.max(dpr, v * h * 0.9); g.fillRect(i * bw, (h - bh) / 2, bw * 0.6, bh); });
        const s = (performance.now() - t0) / 1000;
        els.recTime.textContent = `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
        if (s > 120) mr.stop();
      };
      mr.ondataavailable = (e) => e.data.size && chunks.push(e.data);
      mr.onstop = async () => {
        cancelAnimationFrame(raf); stream.getTracks().forEach((tr) => tr.stop()); srcNode.disconnect();
        rec = null; els.rec.classList.remove("recording"); $("span", els.rec).textContent = t("studio.record"); els.recMeter.hidden = true;
        const blob = new Blob(chunks, { type: mr.mimeType });
        const d = new Date(), stamp = `${String(d.getHours()).padStart(2, "0")}${String(d.getMinutes()).padStart(2, "0")}`;
        try { setInput(await DSP.decodeToMono16k(await blob.arrayBuffer()), `kayit_${stamp}.wav`); } catch { toast(t("toast.decode"), "err"); }
      };
      mr.start(); rec = mr; draw();
      els.rec.classList.add("recording"); $("span", els.rec).textContent = t("studio.stop"); els.recMeter.hidden = false;
    });
    app.cleanups.push(() => { if (rec) rec.stop(); });

    // --- çıktı
    const wave = $("#waveCanvas", root), waveWrap = $("#waveWrap", root), head = $("#playhead", root);
    const playBtn = $("#playBtn", root), timeLbl = $("#timeLbl", root), spec = $("#spec", root);
    let raf = 0;
    const abSeg = seg($("#abSeg", root), "side", S.side, (v) => { S.side = v; player.setSide(v); paintWave(); });

    const paintWave = () => {
      if (!S.output) return;
      const p = player.duration ? player.time() / player.duration : 0;
      const norm = DSP.peaks(S.input.samples, 400).max;
      const A = { samples: S.input.samples, color: cssVar("--line-2"), played: cssVar("--ink-2"), alpha: 1, norm };
      const B = { samples: S.output.samples, color: cssVar("--ink-2"), played: cssVar("--accent"), alpha: 1, norm };
      if (S.side === "a") { A.color = cssVar("--ink-2"); A.played = cssVar("--ink"); B.alpha = 0.35; }
      drawWave(wave, S.side === "a" ? [B, A] : [A, B], p);
      head.style.left = `${p * 100}%`;
      timeLbl.textContent = `${mmss(player.time())} / ${mmss(player.duration)}`;
    };
    const loop = () => { paintWave(); if (player.playing) raf = requestAnimationFrame(loop); };
    const setPlaying = (on) => { playBtn.classList.toggle("playing", on); if (on) { cancelAnimationFrame(raf); loop(); } };
    player.onEnd = () => { setPlaying(false); paintWave(); };
    const togglePlay = () => { if (player.playing) { player.pause(); setPlaying(false); } else { player.play(); setPlaying(true); } paintWave(); };
    playBtn.addEventListener("click", togglePlay);
    waveWrap.addEventListener("pointerdown", (e) => {
      const r = waveWrap.getBoundingClientRect();
      player.seek(((e.clientX - r.left) / r.width) * player.duration);
      paintWave();
    });

    // spektrogram ayırıcısı
    const setSplit = (v) => { S.split = Math.max(0, Math.min(100, v)); spec.style.setProperty("--split", `${S.split}%`); $("#specHandle", root).setAttribute("aria-valuenow", Math.round(S.split)); };
    setSplit(S.split);
    let dragging = false;
    const fromEv = (e) => { const r = spec.getBoundingClientRect(); setSplit(((e.clientX - r.left) / r.width) * 100); };
    spec.addEventListener("pointerdown", (e) => { dragging = true; spec.classList.add("dragging"); spec.setPointerCapture(e.pointerId); fromEv(e); });
    spec.addEventListener("pointermove", (e) => dragging && fromEv(e));
    spec.addEventListener("pointerup", () => { dragging = false; spec.classList.remove("dragging"); });
    $("#specHandle", root).addEventListener("keydown", (e) => {
      if (e.key === "ArrowLeft") { setSplit(S.split - 5); e.preventDefault(); e.stopPropagation(); }
      if (e.key === "ArrowRight") { setSplit(S.split + 5); e.preventDefault(); e.stopPropagation(); }
    });

    const paintSpecs = () => {
      if (!S.output) return;
      S.specA ??= DSP.spectrogram(S.input.samples);
      S.specB ??= DSP.spectrogram(S.output.samples);
      drawSpec($("#specA", root), S.specA, S.specA.max);
      drawSpec($("#specB", root), S.specB, S.specA.max);
    };

    const renderFacts = () => {
      const o = S.output; if (!o) return;
      const dur = S.input.samples.length / DSP.SR;
      const red = 20 * Math.log10(DSP.rms(S.input.samples) / Math.max(1e-9, DSP.rms(o.samples)));
      const facts = [
        ["studio.duration", `${fmt(dur, 1)} s`],
        ["studio.elapsed", o.elapsed != null ? `${fmt(o.elapsed, 2)} s` : "·"],
        ["studio.rtf", o.elapsed != null ? `${fmt(o.elapsed / dur, 3)}×` : "·"],
        ["studio.device", o.device || "·"],
        ["studio.reduction", `${fmt(red, 1)} dB`],
      ];
      $("#facts", root).innerHTML = facts.map(([k, v]) => `<div class="fact"><span>${t(k)}</span><b>${v}</b></div>`).join("");

      const m = o.metrics, box = $("#metrics", root);
      box.hidden = !m;
      if (!m) return;
      const defs = [["si_snr", "SI-SNR (dB)", 2, [-10, 25]], ["pesq", "PESQ", 2, [1, 4.5]], ["stoi", "STOI", 3, [0, 1]]];
      $("#metricGrid", root).innerHTML = defs.filter(([k]) => m.noisy?.[k] != null && m.enhanced?.[k] != null).map(([k, label, dec, [lo, hi]]) => {
        const b = m.noisy[k], a = m.enhanced[k], d = a - b;
        const pct = (v) => Math.max(2, Math.min(100, ((v - lo) / (hi - lo)) * 100));
        return `<div class="mcard"><span class="mname">${label}</span>
          <div class="mrow"><span class="mbefore">${fmt(b, dec)}</span><span class="mafter">${fmt(a, dec)}</span><span class="mdelta ${d < 0 ? "neg" : ""}">${fmt(d, dec, true)}</span></div>
          <div class="mbar"><i class="b" style="width:${pct(b)}%"></i></div><div class="mbar"><i class="a" style="width:${pct(a)}%"></i></div></div>`;
      }).join("");
    };

    function showOutput() {
      const has = !!S.output;
      els.outEmpty.hidden = has;
      els.out.hidden = !has;
      cancelAnimationFrame(raf);
      setPlaying(false);
      if (!has) return;
      player.load(S.input.samples, S.output.samples);
      player.setSide(S.side);
      abSeg.set(S.side);
      const dl = $("#dlBtn", root);
      dl.href = S.output.url;
      dl.download = S.input.name.replace(/\.[^.]+$/, "").replace(/_noisy$/, "") + `_temiz_${S.output.model}.wav`;
      requestAnimationFrame(() => { paintWave(); paintSpecs(); renderFacts(); });
    }

    // --- çalıştır
    runBtn.addEventListener("click", async () => {
      if (!S.input) return;
      runBtn.classList.add("busy"); syncRun();
      try {
        const q = new URLSearchParams({ model: S.model });
        if (S.input.ref) q.set("ref", S.input.ref);
        const res = await fetch(`/api/enhance?${q}`, { method: "POST", body: DSP.encodeWav(S.input.samples), headers: { "Content-Type": "audio/wav" } });
        if (!res.ok) throw new Error((await res.text()) || res.status);
        const blob = await res.blob();
        const samples = await DSP.decodeToMono16k(await blob.arrayBuffer());
        const n = S.input.samples.length;
        const out = samples.length === n ? samples : (() => { const a = new Float32Array(n); a.set(samples.subarray(0, n)); return a; })();
        let metrics = null;
        try { metrics = JSON.parse(res.headers.get("X-Metrics") || "null"); } catch { /* yoksay */ }
        if (S.output?.url) URL.revokeObjectURL(S.output.url);
        S.output = { samples: out, url: URL.createObjectURL(blob), model: S.model, elapsed: +res.headers.get("X-Elapsed") || null, device: res.headers.get("X-Device"), metrics };
        S.specA = S.specB = undefined;
        const used = app.server.models.find((m) => m.id === S.model);
        if (used && !used.local) { used.local = true; renderModels(); }
        S.side = "b";
        showOutput();
        if (matchMedia("(max-width: 920px)").matches) $(".studio-main", root).scrollIntoView({ behavior: "smooth", block: "start" });
      } catch (err) {
        toast(`${t("toast.server")}: ${String(err.message || err).slice(0, 140)}`, "err");
      } finally {
        runBtn.classList.remove("busy"); syncRun();
      }
    });

    // --- klavye
    const onKey = (e) => {
      if (!S.output || e.ctrlKey || e.metaKey || e.altKey || e.target.closest?.("input, textarea, [contenteditable]")) return;
      const k = e.key.toLowerCase();
      if (k === " ") { e.preventDefault(); togglePlay(); }
      else if (k === "a" || k === "b") { S.side = k; abSeg.set(k); player.setSide(k); paintWave(); }
      else if (e.key === "ArrowLeft") { player.seek(player.time() - 5); paintWave(); }
      else if (e.key === "ArrowRight") { player.seek(player.time() + 5); paintWave(); }
    };
    addEventListener("keydown", onKey);
    const onResize = () => { paintWave(); if (S.input) drawWave($("#inputWave", root), [{ samples: S.input.samples, color: cssVar("--ink-2"), alpha: 1 }]); };
    addEventListener("resize", onResize);
    app.cleanups.push(() => { removeEventListener("keydown", onKey); removeEventListener("resize", onResize); player.pause(); cancelAnimationFrame(raf); });

    app.onLang = () => { renderChips(); renderModels(); syncRun(); showInput(); renderFacts(); $("span", els.rec).textContent = rec ? t("studio.stop") : t("studio.record"); };
    app.onTheme = () => requestAnimationFrame(() => { S.input && showInput(); paintWave(); paintSpecs(); });
    app.onServer = () => { renderModels(); syncRun(); };

    showInput();
    showOutput();
    if (S.pendingSample) { const id = S.pendingSample; S.pendingSample = null; loadSample(id); }
  }

  // ------------------------------------------------------------------ ÖRNEKLER
  function initSamples(root) {
    const grid = root;
    const audio = new Audio();
    let active = null, raf = 0;
    const cache = {};

    const cards = (list, start) => list.map((s, i) => `
      <article class="card scard" data-reveal style="--i:${i + start}" data-id="${s.id}">
        <div class="scard-head"><span class="snr ${s.snr <= 0 ? "low" : ""}">${fmt(s.snr, 0, true)} dB SNR</span><span class="mono dim" data-i18n="samples.src.${s.lang || "ds"}">${t("samples.src." + (s.lang || "ds"))}</span></div>
        <h3 data-i18n="noise.${s.noise}">${t("noise." + s.noise)}</h3>
        ${s.text ? `<p class="quote">“${s.text}”</p>` : ""}
        <div class="swave"><canvas></canvas><span class="sprog"></span></div>
        <div class="sbtns">
          <button type="button" class="sbtn" data-kind="noisy"><svg viewBox="0 0 12 12"><path d="M2 1v10l9-5z"/></svg><span data-i18n="samples.noisy">${t("samples.noisy")}</span></button>
          <button type="button" class="sbtn clean" data-kind="clean"><svg viewBox="0 0 12 12"><path d="M2 1v10l9-5z"/></svg><span data-i18n="samples.clean">${t("samples.clean")}</span></button>
        </div>
        <a href="#/studio" class="btn btn-ghost" data-open data-magnetic><span data-i18n="samples.open">${t("samples.open")}</span><i class="arr">→</i></a>
      </article>`).join("");
    $("#sampleGrid", root).innerHTML = cards(SAMPLES, 6);

    const loadWaves = (scope) => $$(".scard", scope).forEach(async (card) => {
      try {
        const [n, c] = await Promise.all(["noisy", "clean"].map(async (k) =>
          DSP.decodeToMono16k(await (await fetch(sampleUrl(card.dataset.id, k))).arrayBuffer())));
        cache[card.dataset.id] = { noisy: n, clean: c, norm: DSP.peaks(n, 300).max };
        paint(card);
      } catch { /* file:// altında fetch çalışmaz; dalga formu boş kalır */ }
    });

    // Türkçe örnekler ve denemeler tablosu
    const renderTrials = () => {
      if (!TR) return;
      const names = TR.models, S0 = TR.summary;
      const best = names.reduce((a, b) => (S0[b].si_snr > S0[a].si_snr ? b : a));
      $("#trialSum", root).innerHTML =
        `<div class="fact"><span>${t("samples.avgNoisy")}</span><b>${fmt(S0.noisy.si_snr, 2)} dB</b></div>` +
        names.map((n) => `<div class="fact"><span>${t("model." + n)}</span><b>${fmt(S0[n].si_snr, 2)} dB<small>${fmt(S0[n].si_snr - S0.noisy.si_snr, 2, true)}</small></b></div>`).join("") +
        `<div class="fact"><span>STOI · ${t("model." + best)}</span><b>${fmt(S0.noisy.stoi, 3)} → ${fmt(S0[best].stoi, 3)}</b></div>`;
      const head = names.map((n) => `<th>${t("model." + n)} SI-SNR</th>`).join("");
      $("#trialTable", root).innerHTML = `
        <thead><tr><th>${t("samples.colSample")}</th><th>SNR</th><th>${t("samples.colNoise")}</th><th>${t("samples.colIn")}</th>${head}<th>STOI</th></tr></thead>
        <tbody>${TR.items.map((it) => {
          const m = it.metrics, top = names.reduce((a, b) => (m[b].si_snr > m[a].si_snr ? b : a));
          return `<tr><td class="dim">${it.text ? `“${it.text}”` : sampleName(it.id)}</td><td class="n">${fmt(it.snr, 0, true)} dB</td><td>${t("noise." + it.noise)}</td>
            <td class="n">${fmt(m.noisy.si_snr, 2)}</td>
            ${names.map((n) => `<td class="n ${n === top ? "best" : ""}">${fmt(m[n].si_snr, 2)} <span class="d pos">${fmt(m[n].si_snr - m.noisy.si_snr, 2, true)}</span></td>`).join("")}
            <td class="n">${fmt(m.noisy.stoi, 3)} → ${fmt(m[top].stoi, 3)}</td></tr>`;
        }).join("")}</tbody>`;
    };
    trReady.then(() => {
      if (!TR || !root.isConnected) return;
      $("#trSection", root).hidden = false;
      $("#trGrid", root).innerHTML = cards(TR.items, 3);
      renderTrials();
      loadWaves($("#trGrid", root));
    });

    const paint = (card) => {
      const d = cache[card.dataset.id]; if (!d) return;
      drawWave($("canvas", card), [
        { samples: d.noisy, color: cssVar("--line-2"), alpha: 1, norm: d.norm },
        { samples: d.clean, color: cssVar("--accent-2"), alpha: 0.9, norm: d.norm },
      ]);
    };
    loadWaves($("#sampleGrid", root));

    const tick = () => {
      if (active && audio.duration) $(".sprog", active.card).style.width = `${(audio.currentTime / audio.duration) * 100}%`;
      raf = requestAnimationFrame(tick);
    };
    const stopActive = () => {
      if (!active) return;
      audio.pause();
      $$(".sbtn", active.card).forEach((b) => b.classList.remove("on"));
      $(".sprog", active.card).style.width = "0";
      active = null;
    };
    audio.addEventListener("ended", stopActive);

    grid.addEventListener("click", (e) => {
      const card = e.target.closest(".scard"); if (!card) return;
      if (e.target.closest("[data-open]")) { app.studio.pendingSample = card.dataset.id; return; }
      const b = e.target.closest(".sbtn");
      const wv = e.target.closest(".swave");
      if (wv && active?.card === card && audio.duration) {
        const r = wv.getBoundingClientRect();
        audio.currentTime = ((e.clientX - r.left) / r.width) * audio.duration;
        return;
      }
      if (!b) return;
      const kind = b.dataset.kind;
      if (active?.card === card && active.kind === kind) { stopActive(); return; }
      const keep = active?.card === card ? audio.currentTime : 0;
      stopActive();
      audio.src = sampleUrl(card.dataset.id, kind);
      audio.currentTime = keep;
      audio.play().catch(() => {});
      active = { card, kind };
      b.classList.add("on");
    });
    raf = requestAnimationFrame(tick);
    const onResize = () => $$(".scard", grid).forEach(paint);
    addEventListener("resize", onResize);
    app.onTheme = () => requestAnimationFrame(onResize);
    app.onLang = renderTrials;
    app.cleanups.push(() => { stopActive(); cancelAnimationFrame(raf); removeEventListener("resize", onResize); audio.src = ""; });
  }

  // ------------------------------------------------------------------ SONUÇLAR
  function initResults(root) {
    const st = app.results;
    const bars = $("#bars", root);
    bars.innerHTML = `<div class="bars-area"><span class="bref"><span></span></span><span class="bbase"></span></div>` + EXPS.map((id) => `
      <div class="brow" data-id="${id}">
        <span class="bl"><span data-i18n="exp.${id}"></span><small data-i18n="chg.${id}"></small></span>
        <div class="btrack"><span class="bfill"></span><span class="berr"></span></div>
        <span class="bval"></span>
      </div>`).join("");
    applyI18n(bars);

    const updateBars = () => {
      const D = R[st.split], m = st.metric, [lo, hi] = METRIC_DOMAIN[m], dec = METRIC_DEC[m];
      const pos = (v) => Math.max(0, Math.min(1, (v - lo) / (hi - lo)));
      const best = EXPS.reduce((a, b) => (D[b][m][0] > D[a][m][0] ? b : a));
      const di = { si_snr: 0, pesq: 1, stoi: 2 }[m];
      $$(".brow", bars).forEach((row) => {
        const id = row.dataset.id, [mean, sd] = D[id][m];
        row.classList.toggle("best", id === best);
        row.classList.toggle("base", id === "baseline");
        $(".bfill", row).style.transform = `scaleX(${pos(mean)})`;
        const err = $(".berr", row);
        err.style.left = `${pos(mean - sd) * 100}%`; err.style.width = `${(pos(mean + sd) - pos(mean - sd)) * 100}%`;
        err.style.display = sd ? "" : "none";
        const d = D[id].d?.[di];
        $(".bval", row).innerHTML = `${fmt(mean, dec)}<small class="${d == null ? "" : d < 0 ? "neg" : "pos"}">${d == null ? "± " + fmt(sd, dec) : fmt(d, dec, true)}</small>`;
      });
      const area = $(".bars-area", bars);
      $(".bref", area).style.left = `${pos(D.e0[m]) * 100}%`;
      $(".bref span", area).textContent = `E0 ${fmt(D.e0[m], dec)}`;
      $(".bbase", area).style.left = `${pos(D.baseline[m][0]) * 100}%`;
    };
    // ilk girişte çubuklar sıfırdan büyüsün
    $$(".bfill", bars).forEach((f) => (f.style.transform = "scaleX(0)"));
    setTimeout(updateBars, reduced ? 0 : 450);

    const renderTable = () => {
      const D = R[st.split];
      const bestOf = (k) => EXPS.reduce((a, b) => (D[b][k][0] > D[a][k][0] ? b : a));
      const bs = { si_snr: bestOf("si_snr"), pesq: bestOf("pesq"), stoi: bestOf("stoi") };
      const dcell = (v, dec) => v == null ? "<td></td>" : `<td class="n d ${v < 0 ? "neg" : "pos"}">${fmt(v, dec, true)}</td>`;
      $("#rtable", root).innerHTML = `
        <thead><tr><th>${t("results.exp")}</th><th>${t("results.change")}</th><th>SI-SNR (dB)</th><th>PESQ</th><th>STOI</th><th>Δ SI-SNR</th><th>Δ PESQ</th><th>Δ STOI</th></tr></thead>
        <tbody>
          <tr class="e0"><td>${t("results.input")}</td><td></td><td class="n">${fmt(D.e0.si_snr, 2)}</td><td class="n">${fmt(D.e0.pesq, 3)}</td><td class="n">${fmt(D.e0.stoi, 3)}</td><td></td><td></td><td></td></tr>
          ${EXPS.map((id) => {
            const r = D[id], c = (k, dec) => `<td class="n ${bs[k] === id ? "best" : ""}">${fmt(r[k][0], dec)} <span class="dim">± ${fmt(r[k][1], dec)}</span></td>`;
            return `<tr><td><strong>${t("exp." + id)}</strong></td><td class="dim">${t("chg." + id)}</td>${c("si_snr", 2)}${c("pesq", 3)}${c("stoi", 3)}${dcell(r.d?.[0], 2)}${dcell(r.d?.[1], 3)}${dcell(r.d?.[2], 3)}</tr>`;
          }).join("")}
        </tbody>`;
    };
    renderTable();

    seg($("#splitSeg", root), "v", st.split, (v) => { st.split = v; updateBars(); renderTable(); });
    seg($("#metricSeg", root), "v", st.metric, (v) => { st.metric = v; updateBars(); });

    // --- E5 çizgi grafiği
    const svg = $("#lineChart", root), legend = $("#legend", root), wrap = $(".line-wrap", root);
    const tip = document.createElement("div"); tip.className = "ctip"; wrap.append(tip);
    let focus = null, drawn = false;
    const renderLine = () => {
      const W = Math.max(300, svg.clientWidth), H = 320, P = { l: 40, r: 16, t: 14, b: 32 };
      svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
      const x = (i) => P.l + (i / (SNRS.length - 1)) * (W - P.l - P.r);
      const y = (v) => P.t + (1 - v / 14) * (H - P.t - P.b);
      let s = `<g class="grid">${[0, 2, 4, 6, 8, 10, 12, 14].map((v) => `<line x1="${P.l}" x2="${W - P.r}" y1="${y(v)}" y2="${y(v)}"/>`).join("")}</g>`;
      s += `<g class="axis">${[0, 4, 8, 12].map((v) => `<text x="${P.l - 8}" y="${y(v) + 3}" text-anchor="end">+${v}</text>`).join("")}`;
      s += SNRS.map((v, i) => `<text x="${x(i)}" y="${H - 8}" text-anchor="middle">${fmt(v, 0, true)} ${t("results.inputDb")}</text>`).join("") + "</g>";
      EXPS.forEach((id) => {
        const d = E5[id].map((v, i) => `${i ? "L" : "M"}${x(i).toFixed(1)} ${y(v).toFixed(1)}`).join(" ");
        s += `<path class="ln ${drawn ? "" : "draw"}" data-id="${id}" d="${d}" style="stroke:${EXP_COLOR[id]}"/>`;
        s += E5[id].map((v, i) => `<circle class="pt" data-id="${id}" data-i="${i}" cx="${x(i)}" cy="${y(v)}" r="${id === "e3_depth5" || id === "e1_si_snr" ? 4 : 3}" style="fill:${EXP_COLOR[id]}"/>`).join("");
      });
      svg.innerHTML = s;
      $$(".ln.draw", svg).forEach((p, i) => { const L = p.getTotalLength(); p.style.setProperty("--len", L); p.style.strokeDasharray = L; p.style.animationDelay = `${i * 90}ms`; });
      drawn = true;
      applyFocus();
    };
    const applyFocus = () => {
      svg.classList.toggle("focus", !!focus);
      $$(".ln, .pt", svg).forEach((el) => el.classList.toggle("hot", el.dataset.id === focus));
      $$("button", legend).forEach((b) => b.classList.toggle("hot", b.dataset.id === focus));
    };
    const renderLegend = () => {
      legend.innerHTML = EXPS.map((id) => `<button type="button" data-id="${id}"><i style="background:${EXP_COLOR[id]}"></i>${t("exp." + id)}</button>`).join("");
    };
    renderLegend();
    legend.addEventListener("pointerover", (e) => { const b = e.target.closest("button"); if (b) { focus = b.dataset.id; applyFocus(); } });
    legend.addEventListener("pointerleave", () => { focus = null; applyFocus(); });
    svg.addEventListener("pointerover", (e) => {
      const p = e.target.closest(".pt, .ln"); if (!p) return;
      focus = p.dataset.id; applyFocus();
      if (p.classList.contains("pt")) {
        const i = +p.dataset.i, r = wrap.getBoundingClientRect(), c = p.getBoundingClientRect();
        tip.textContent = `${t("exp." + focus)} · ${fmt(SNRS[i], 0, true)} dB → ${fmt(E5[focus][i], 2, true)} dB`;
        tip.style.left = `${c.left - r.left + c.width / 2}px`; tip.style.top = `${c.top - r.top}px`;
        tip.classList.add("show");
      }
    });
    svg.addEventListener("pointerout", (e) => { if (!e.relatedTarget?.closest?.(".pt, .ln")) { focus = null; applyFocus(); tip.classList.remove("show"); } });

    // grafik görünür olunca çiz
    const io = new IntersectionObserver(([en]) => { if (en.isIntersecting) { io.disconnect(); renderLine(); } }, { threshold: 0.25 });
    io.observe(svg);
    const ro = new ResizeObserver(() => drawn && renderLine());
    ro.observe(svg);
    app.cleanups.push(() => { io.disconnect(); ro.disconnect(); });
    app.onLang = () => { updateBars(); renderTable(); renderLegend(); if (drawn) renderLine(); };
  }

  // ------------------------------------------------------------------ açılış
  function intro() {
    const el = $("#intro");
    let seen = false;
    try { seen = sessionStorage.getItem("st-intro") === "1"; sessionStorage.setItem("st-intro", "1"); } catch { /* yoksay */ }
    if (seen || reduced) { el.remove(); return Promise.resolve(); }
    const path = $("#introPath");
    const t0 = performance.now(), dur = 1300;
    return new Promise((resolve) => {
      const step = (now) => {
        const p = Math.min(1, (now - t0) / dur);
        const noise = Math.max(0, 1 - p * 1.4);
        let d = "M0 30";
        for (let x = 4; x <= 240; x += 4) {
          const env = Math.sin((x / 240) * Math.PI);
          const speech = Math.sin(x * 0.18 + now / 120) * 14 * env * Math.min(1, p * 2);
          const n = (Math.random() - 0.5) * 34 * noise * env;
          d += ` L${x} ${(30 + speech + n).toFixed(1)}`;
        }
        path.setAttribute("d", d);
        if (p < 1) requestAnimationFrame(step);
        else { el.classList.add("done"); setTimeout(() => { el.remove(); }, 1000); resolve(); }
      };
      requestAnimationFrame(step);
    });
  }

  // ------------------------------------------------------------------ başlat
  applyI18n(document);
  updateStatusPill();
  app.ready = intro();
  navigate(true);
  checkServer();
  setInterval(() => { if (app.server.state !== "online") checkServer(); }, 8000);
  if (document.fonts) document.fonts.ready.then(positionNavPill);
})();

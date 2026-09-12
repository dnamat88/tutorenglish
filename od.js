'use strict';
/* ============================================================
   English Tutor — on-device PWA (v0.2)
   brain: Gemma 4 E2B  (LiteRT-LM.js, WebGPU)
   ears : Whisper small (transformers.js, WebGPU→WASM)
   voice: Supertonic 3 F1 (onnxruntime-web, WebGPU→WASM)
   Everything runs in the browser. Models cached in Cache Storage.
   ============================================================ */

// ---------------- config ----------------
const GEMMA_URL = 'https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm/resolve/main/gemma-4-E2B-it-web.litertlm';
const SUP_HF    = 'https://huggingface.co/Supertone/supertonic-3/resolve/main';
const SUP_FILES = ['tts.json', 'unicode_indexer.json',
                   'duration_predictor.onnx', 'text_encoder.onnx',
                   'vector_estimator.onnx', 'vocoder.onnx'];
const VOICE = 'F1';
const TTS_SPEED = 1.05;
const CACHE_NAME = 'et-od-v3';
const VERSION = '27';

// ---------------- config: presets (localStorage) + URL overrides ----------------
// v27: config is chosen in the UI, not hidden in the URL. Two presets map to the two
// ways the app is actually used; URL params remain as an A/B override on top.
//   PRESET CASA     → brain=pc (Qwen 27B on the PC), ears=web (native, zero WASM)
//   PRESET MACCHINA → brain=device (Gemma 4 E2B on the phone), ears=web (native)
// `ears=web` (ADR-0001) is the safe default everywhere: the phone's own Web Speech
// API, ~1 s latency, no WASM heap, no PC required (needs mobile data, like a call).
// The old whisper knobs (ears=keep/pc, ?whisper=, ?tts=, ?asr=) are kept for A/B.
const URLP = new URLSearchParams(location.search);
const CFG_KEY = 'et-od-cfg-v1';
const PRESETS = {
  casa:     { brain: 'pc',     ears: 'web' },
  macchina: { brain: 'device', ears: 'web' },
};
function readStoredCfg() {
  try { return Object.assign({}, JSON.parse(localStorage.getItem(CFG_KEY) || 'null') || {}); } catch (_) { return {}; }
}
function resolveCfg() {
  const s = readStoredCfg();
  const preset = (PRESETS[s.preset] ? s.preset : null) || 'macchina';
  const base = PRESETS[preset];
  const cfg = {
    preset,
    pc: (s.pc || '').trim(),                       // Tailscale base URL of the PC API ('' = same-origin / dev)
    brain: s.brain  || base.brain,
    ears:  s.ears  || base.ears,
    steps: (s.steps || 5),
    tts:   s.tts   || 'wasm',
    asr:   s.asr   || 'wasm',
    recmax:(s.recmax || 30),
    whisper: s.whisper || 'Xenova/whisper-tiny.en',
  };
  // URL overrides (A/B testing / one-off experiments) — never written back to storage.
  if (URLP.get('brain')) cfg.brain = URLP.get('brain');
  if (URLP.get('ears')) cfg.ears = URLP.get('ears');
  if (URLP.get('steps')) cfg.steps = Math.min(12, Math.max(2, parseInt(URLP.get('steps'), 10) || 5));
  if (URLP.get('tts')) cfg.tts = URLP.get('tts');
  if (URLP.get('asr')) cfg.asr = URLP.get('asr');
  if (URLP.get('recmax')) cfg.recmax = Math.max(5, Math.min(60, parseInt(URLP.get('recmax'), 10) || 30));
  if (URLP.get('whisper')) cfg.whisper = URLP.get('whisper');
  if (URLP.get('pc')) cfg.pc = URLP.get('pc').trim().replace(/\/$/, '');
  return cfg;
}
function saveCfg(patch) {
  const s = readStoredCfg();
  const next = Object.assign(s, patch);
  if (patch.preset && PRESETS[patch.preset]) {           // picking a preset sets brain (ears only as a default)
    next.brain = PRESETS[patch.preset].brain;
    if (patch.ears == null) next.ears = PRESETS[patch.preset].ears;
  }
  try { localStorage.setItem(CFG_KEY, JSON.stringify(next)); } catch (_) {}
  CFG.pc = (next.pc || '').trim().replace(/\/$/, '');    // live-update derived consts if pc changed
  CFG.preset = next.preset; CFG.brain = next.brain; CFG.ears = next.ears;
  CFG.steps = next.steps || 5; CFG.recmax = next.recmax || 30;
}
const CFG = resolveCfg();
// PC API base: same-origin in dev (PC serves the site, pc=''); the Tailscale URL in prod.
function api(path) { return CFG.pc + path; }
let pcAlive = null;   // null=unknown, true/false after a probe
async function probePC(timeoutMs = 3000) {
  try { const r = await fetch(api('/health'), { signal: AbortSignal.timeout(timeoutMs) }); pcAlive = r.ok; }
  catch (_) { pcAlive = false; }
  return pcAlive;
}
// ---------------- derived runtime constants (old names, so the rest of od.js is unchanged) ----------------
const TTS_STEPS = Math.min(12, Math.max(2, CFG.steps | 0 || 5));
// v17: whisper-small OOM-killed Brave with brain+TTS resident; tiny.en ≈ 33 MB is the safe default.
const WHISPER = CFG.whisper;
// transformers.js v4: English-only models (id ending in ".en") REJECT task/language kwargs.
const WHISPER_EN_ONLY = WHISPER.endsWith('.en');
const REC_MAX = Math.max(5, Math.min(60, CFG.recmax | 0 || 30));
const TTS_EP = CFG.tts;   // 'wasm' | 'webgpu'
const ASR_EP = CFG.asr;   // 'wasm' | 'webgpu'
const BRAIN = CFG.brain;  // 'device' | 'pc'
const EARS = CFG.ears;    // 'web' | 'keep' | 'pc' | 'reload'

const SYSTEM_PROMPT = `You are "Coach", a friendly English conversation partner for an Italian learner (B2 level). The conversation is only in spoken English.

Every reply must have EXACTLY this structure:
[2-4 short spoken sentences that end with one follow-up question]
[blank line]
CORRECTIONS: [JSON array of at most 2 objects]

Each object must have exactly these 4 keys: "you_said" (the learner's wrong phrase), "better" (the corrected phrase), "why" (one short English reason), "it" (the corrected phrase translated into Italian). If the learner made no clear error, write: CORRECTIONS: []

Example of a full correct reply:

Nice! A restaurant is a great choice. What is the biggest problem your customers have?

CORRECTIONS: [{"you_said":"I like this app because is easy","better":"I like this app because it is easy","why":"missing subject","it":"mi piace questa app perché è facile"}]

Rules: never mention the corrections inside your spoken sentences. Keep the JSON on one line, with double quotes.`;

const FALLBACK_SCENARIO = {
  name: 'Company intro & vision',
  opening: 'Hi! I am Coach, your English conversation partner. Let us talk about your company. Tell me something about what you do.',
};

// ---------------- state ----------------
const state = {
  cache: null,
  engine: null, conv: null,          // LLM
  asr: null, asrDevice: null,        // ASR
  tts: null, ttsEp: null,            // {ort, cfgs, indexer, sessions, style}
  stage: 'boot',
  ready: false,
  rec: null, chunks: [], recording: false, busy: false,
  ttsChain: Promise.resolve(),
};
let turn = null; // per-turn metrics + buffers
let SCENARIO = FALLBACK_SCENARIO;

// ---------------- tiny helpers ----------------
const $ = s => document.querySelector(s);
const sleep = ms => new Promise(r => setTimeout(r, ms));
// fire-and-forget telemetry to the server (never throws, never blocks)
function tlog(stage, extra) {
  const msg = stage + (extra ? ' ' + extra : '');
  try { fetch(api('/odlog'), { method: 'POST', body: msg, keepalive: true }).catch(() => {}); } catch (_) {}
}
// surface JS errors on screen AND to the server (prototype: no invisible failures)
window.addEventListener('error', e => { try { hud('💥 JS error: ' + e.message); tlog('window-error', e.message); } catch (_) {} });
window.addEventListener('unhandledrejection', e => { try { const m = (e.reason && (e.reason.message || e.reason)) || 'unknown'; hud('💥 promise error: ' + m); tlog('unhandled-rejection', String(m).slice(0, 200)); } catch (_) {} });
// v11: crash telemetry — 5s heartbeat with the current stage (until ready),
// plus a pagehide beacon so a dead tab reports exactly where it died.
let hbTimer = null, hbN = 0;
function stopHeartbeat() { if (hbTimer) { clearInterval(hbTimer); hbTimer = null; } }
function memInfo() {
  const m = typeof performance !== 'undefined' && performance.memory;
  return m ? ' js=' + (m.usedJSHeapSize / 1048576).toFixed(0) + 'MB' : '';
}
function startHeartbeat() {
  stopHeartbeat();
  hbN = 0;
  hbTimer = setInterval(() => {
    // v16: never stops (also covers crashes during conversation); every beat = stage + JS heap
    tlog('hb', state.stage + memInfo());
  }, 10000);
}
// v16: SINGLE-TAB LOCK. Each tab pins ~3.5 GB (2GB brain + TTS + ASR); two tabs OOM-crash the phone.
// One localStorage claim, refreshed every 3 s by the owner. A fresh foreign claim => we yield:
// show a notice and wait. If the owner dies/is closed, the first poller takes over automatically.
const TAB_ID = Math.random().toString(36).slice(2, 8);
const CLAIM_KEY = 'et-od-owner-v16';
// v22: 7 s was too tight — during heavy sync WASM inference (brain denoise steps, 2 GB stream)
// the owner's refresh interval can be starved and a second tab snatches the claim (v17 crash:
// two tabs each loading a brain). 20 s window, 2 s refresh. A dead tab still clears in ~22 s.
const CLAIM_FRESH_MS = 20000;
let iAmOwner = false, claimTimer = null;
function claimInfo() { try { return JSON.parse(localStorage.getItem(CLAIM_KEY) || 'null'); } catch (e) { return null; } }
function writeClaim() { try { localStorage.setItem(CLAIM_KEY, JSON.stringify({ id: TAB_ID, ts: Date.now() })); } catch (e) {} }
function claimIsForeignFresh() { const c = claimInfo(); return !!(c && c.id !== TAB_ID && Date.now() - c.ts < CLAIM_FRESH_MS); }
function releaseClaim() {
  if (claimTimer) { clearInterval(claimTimer); claimTimer = null; }
  const c = claimInfo();
  if (c && c.id === TAB_ID) { try { localStorage.removeItem(CLAIM_KEY); } catch (e) {} }
  iAmOwner = false;
}
async function claimOrYield() {
  if (claimIsForeignFresh()) return false;
  writeClaim();
  await new Promise(r => setTimeout(r, 900)); // give a concurrent tab a moment to take over (tie-break)
  if (claimIsForeignFresh()) return false;
  iAmOwner = true;
  claimTimer = setInterval(writeClaim, 2000);
  return true;
}
async function waitOwnership() {
  if (await claimOrYield()) return;
  tlog('tab-yield', 'waiting for other tabs to close (me=' + TAB_ID + ')');
  hud('⏳ another tab of English Tutor is active — close it (or this tab). This tab will start automatically when one closes.');
  setBar('llm', 2, 'waiting…', false); setBar('tts', 2, 'waiting…', false); setBar('asr', 2, 'waiting…', false);
  while (!(await claimOrYield())) await new Promise(r => setTimeout(r, 2000));
  tlog('tab-claimed', 'ownership taken (me=' + TAB_ID + ')');
}
window.addEventListener('pagehide', () => {
  tlog('pagehide', 'stage=' + state.stage + memInfo());
  releaseClaim();
});
window.addEventListener('pageshow', () => {
  // bfcache restore: timers were paused, claim may be gone — re-take it.
  if (iAmOwner && !claimIsForeignFresh()) { writeClaim(); if (!claimTimer) claimTimer = setInterval(writeClaim, 2000); }
});
// secondary signal: let the owner tab log second-tab appearances (claim is authoritative)
try {
  const tabCh = new BroadcastChannel('et-od-tabs');
  tabCh.onmessage = e => {
    const d = e.data || {};
    if (d.type === 'ping' && d.id !== TAB_ID && iAmOwner) tlog('multi-tab', 'second tab ' + d.id + ' detected (me=' + TAB_ID + ')');
  };
  tabCh.postMessage({ type: 'ping', id: TAB_ID });
} catch (e) {}
function setBar(id, pct, label, done) {
  const f = $('#f-' + id);
  f.style.width = Math.min(100, Math.max(0, pct)).toFixed(1) + '%';
  if (done) f.classList.add('done');
  const s = $('#s-' + id);
  if (s && label) s.textContent = label;
}
function fmtBytes(n) {
  if (n >= 1e9) return (n / 1e9).toFixed(2) + ' GB';
  if (n >= 1e6) return (n / 1e6).toFixed(0) + ' MB';
  return (n / 1e3).toFixed(0) + ' KB';
}
function hud(t) { $('#hud').textContent = t; }

// ---------------- cache with progress + resume ----------------
// Partial downloads are persisted to Cache Storage every PARTIAL_EVERY bytes,
// so a killed tab (phone standby) can resume with a Range request.
const PARTIAL_EVERY = 200 * 1024 * 1024;
async function cachedBlob(url, onProgress) {
  // Small model files (voice). No resume: if not in cache, fetch fresh and store.
  // (The big brain model uses gemmaModelStream() instead, to stay out of JS heap.)
  const fname = url.split('/').pop();
  let cached = null;
  try { cached = await state.cache.match(url); } catch (e) { /* ignore */ }
  if (cached) {
    const b = await cached.blob();
    tlog('dl-cached', fname + ' ' + b.size);
    onProgress && onProgress({ loaded: b.size, total: b.size, cached: true });
    return b;
  }
  const res = await fetch(url);
  if (!res.ok) throw new Error('download failed: ' + url + ' → HTTP ' + res.status);
  const total = +(res.headers.get('content-length') || 0);
  tlog('dl-start', fname + ' total=' + total);
  const buf = await res.arrayBuffer();
  const blob = new Blob([buf]);
  try { await state.cache.put(url, new Response(blob)); } catch (e) { console.warn('cache put failed', e); }
  tlog('dl-done', fname + ' ' + blob.size);
  onProgress && onProgress({ loaded: blob.size, total: total || blob.size, cached: false });
  return blob;
}
const cachedArrayBuffer = (url, cb) => cachedBlob(url, cb).then(b => b.arrayBuffer());
const cachedJSON = (url, cb) => cachedBlob(url, cb).then(b => b.text()).then(JSON.parse);

// ---------------- brain cache (v12): disk-backed IndexedDB chunks ----------------
// v8/v9 lesson: Cache Storage buffered a SECOND 2 GB copy in RAM (Brave OOM).
// IndexedDB stores blobs on disk; 256 MB chunks keep only one chunk in JS heap.
// First load: download + cache simultaneously. Next loads: stream from disk (~30s).
const BRAIN_DB = 'et-od-brain';
const BRAIN_KEY = 'gemma-4-E2B-it-web';
const BRAIN_CHUNK = 256 * 1024 * 1024;
const BRAIN_TOTAL = 2008432640;
let brainDB = null;
function brainIDB() {
  if (brainDB) return Promise.resolve(brainDB);
  return new Promise((res, rej) => {
    const rq = indexedDB.open(BRAIN_DB, 1);
    rq.onupgradeneeded = () => rq.result.createObjectStore('files');
    rq.onsuccess = () => { brainDB = rq.result; res(brainDB); };
    rq.onerror = () => rej(rq.error);
  });
}
function brainPut(i, buf) {
  return brainIDB().then(db => new Promise((res, rej) => {
    const rq = db.transaction('files', 'readwrite').objectStore('files').put(buf, BRAIN_KEY + ':' + i);
    rq.onsuccess = () => res(); rq.onerror = () => rej(rq.error);
  }));
}
function brainGet(i) {
  return brainIDB().then(db => new Promise((res, rej) => {
    const rq = db.transaction('files', 'readonly').objectStore('files').get(BRAIN_KEY + ':' + i);
    rq.onsuccess = () => res(rq.result || null); rq.onerror = () => rej(rq.error);
  }));
}
async function brainCachedStream() {
  // null unless ALL chunks are present; else a ReadableStream yielding one chunk at a time
  const db = await brainIDB();
  const keys = await new Promise((res, rej) => {
    const rq = db.transaction('files', 'readonly').objectStore('files').getAllKeys();
    rq.onsuccess = () => res(rq.result); rq.onerror = () => rej(rq.error);
  });
  const have = new Set(keys);
  const n = Math.ceil(BRAIN_TOTAL / BRAIN_CHUNK);
  for (let i = 0; i < n; i++) if (!have.has(BRAIN_KEY + ':' + i)) return null;
  const lastExp = BRAIN_TOTAL - (n - 1) * BRAIN_CHUNK; // guard against model file changing
  const lastChunk = await brainGet(n - 1);
  if (!lastChunk || lastChunk.byteLength !== lastExp) { tlog('llm:cache-miss', 'last chunk size ' + (lastChunk ? lastChunk.byteLength : 'null') + ' exp ' + lastExp); return null; }
  tlog('llm:cache-hit', n + ' chunks');
  // v15: classic pull-based ReadableStream — ReadableStream.from() is NOT available on
  // Brave Android ("is not a function", Chrome 122+ API). pull() gives backpressure:
  // a chunk is read from IDB only when the engine asks for the next one.
  let ci = 0;
  const stream = new ReadableStream({
    async pull(ctrl) {
      if (ci >= n) { ctrl.close(); return; }
      const i = ci++;
      const c = await brainGet(i);
      if (!c) { tlog('llm:cache-chunk-fail', i); ctrl.error(new Error('brain chunk ' + i + ' missing from IDB')); return; }
      tlog('llm:cache-chunk', i + '/' + n + ' ' + c.byteLength);
      ctrl.enqueue(new Uint8Array(c));
    },
  });
  tlog('llm:cache-ready');
  return stream;
}

// ---------------- LLM (Gemma 4 via LiteRT-LM) ----------------
// v25: RESUMABLE brain download. The brain is stored as 8×256 MB chunks in IndexedDB.
// The old path was all-or-nothing: a refresh/crash mid-download voided the whole cache,
// so EVERY reload re-downloaded 2 GB (visible in odlog: 'llm:stream fresh' after each
// pagehide mid-download). Now each missing chunk is fetched with HTTP Range (PC disk
// first — /models/brain.litertlm supports Range — then HF CDN fallback) and persisted
// individually: a refresh resumes where it stopped, and the download survives the tab.
const BRAIN_N = Math.ceil(BRAIN_TOTAL / BRAIN_CHUNK);
function brainChunkLen(i) {
  return i === BRAIN_N - 1 ? BRAIN_TOTAL - (BRAIN_N - 1) * BRAIN_CHUNK : BRAIN_CHUNK;
}
async function brainHasChunk(i) {
  const db = await brainIDB();
  return new Promise((res, rej) => {
    const rq = db.transaction('files', 'readonly').objectStore('files').get(BRAIN_KEY + ':' + i);
    rq.onsuccess = () => res(!!rq.result && rq.result.byteLength === brainChunkLen(i));
    rq.onerror = () => rej(rq.error);
  });
}
async function brainDownloadMissing(onProgress) {
  // Source order: prefer the PC disk (fast, same tailnet) if it is alive; otherwise
  // lead with the HF CDN so an OFF PC never stalls the download behind TCP timeouts.
  if (pcAlive === null) await probePC(3000);
  const pcSrc = api('/models/brain.litertlm');
  const order = pcAlive ? [pcSrc, GEMMA_URL] : [GEMMA_URL, pcSrc];
  for (let i = 0; i < BRAIN_N; i++) {
    if (await brainHasChunk(i)) { tlog('llm:chunk-cached', i + '/' + BRAIN_N); continue; }
    const start = i * BRAIN_CHUNK, len = brainChunkLen(i), end = start + len - 1;
    tlog('llm:chunk-start', i + '/' + BRAIN_N + ' order=' + (pcAlive ? 'pc,cdn' : 'cdn,pc'));
    for (const src of order) {
      try {
        const res = await fetch(src, { headers: { Range: 'bytes=' + start + '-' + end } });
        if (res.status !== 206) throw new Error('no range: HTTP ' + res.status);
        const reader = res.body.getReader();
        const buf = new Uint8Array(len);
        let off = 0, lastT = performance.now();
        while (off < len) {
          const { done, value } = await reader.read();
          if (done) break;
          buf.set(value, off); off += value.byteLength;
          const now = performance.now();
          if (now - lastT > 700) { lastT = now; onProgress && onProgress(start + off, BRAIN_TOTAL); }
        }
        if (off !== len) throw new Error('short read ' + off + '/' + len);
        await brainPut(i, buf.buffer); // own buffer (reader chunks were copied via set)
        tlog('llm:chunk-done', i + '/' + BRAIN_N);
        onProgress && onProgress(start + len, BRAIN_TOTAL);
        break;
      } catch (e) {
        tlog('llm:chunk-fail', i + ' ' + src + ' ' + String((e && e.message) || e).slice(0, 120));
      }
    }
    if (!(await brainHasChunk(i))) throw new Error('chunk ' + i + ': tutte le fonti fallite — ri-refresha, riprende da qui');
  }
}
async function gemmaModelStream(onProgress, { cache = false } = {}) {
  // v25: legacy whole-file stream (kept as last-resort fallback; the Range path above
  // is the normal one). Streams the 2 GB brain DIRECTLY to the engine while writing
  // 256 MB chunks to IndexedDB.
  const res = await fetch(GEMMA_URL);
  if (!res.ok) throw new Error('brain download failed: HTTP ' + res.status);
  const total = +(res.headers.get('content-length') || 0);
  tlog('llm:stream', 'fresh total=' + total);
  let got = 0, last = 0, acc = null, off = 0, idx = 0;
  const counter = new TransformStream({
    transform(chunk, ctl) {
      ctl.enqueue(chunk); // engine first, always
      if (cache) {
        if (!acc) acc = new ArrayBuffer(BRAIN_CHUNK);
        if (off + chunk.byteLength > acc.byteLength) {
          const full = acc; acc = new ArrayBuffer(BRAIN_CHUNK); off = 0;
          brainPut(idx++, full).catch(e => console.warn('brain cache put failed', e));
        }
        new Uint8Array(acc, off, chunk.byteLength).set(chunk);
        off += chunk.byteLength;
      }
      got += chunk.byteLength;
      const now = performance.now();
      if (now - last > 700) {
        last = now;
        tlog('llm:dl', got + '/' + total + ' ' + (total ? (got / total * 100).toFixed(0) : 0) + '%');
        onProgress && onProgress(got, total);
      }
    },
    flush() {
      if (cache && acc && off > 0) brainPut(idx, acc.slice(0, off)).catch(e => console.warn('brain cache put failed', e));
    }
  });
  return res.body.pipeThrough(counter);
}
async function loadLLM() {
  setBar('llm', 1, 'importing…');
  tlog('llm:import-start');
  const litert = await import('https://cdn.jsdelivr.net/npm/@litert-lm/core@0.17.0/+esm');
  tlog('llm:import-done');
  let got = 0, total = 0;
  let stream;
  const cached = await brainCachedStream().catch(e => { tlog('brain-cache-fail', String((e && e.message) || e).slice(0, 200)); return null; });
  if (cached) {
    setBar('llm', 3, 'loading brain from cache…');
    tlog('llm:from-cache');
    state.llmFromCache = true;
    stream = cached;
  } else {
    // v25: resume-capable chunked download (PC disk → HF CDN), then stream from IDB.
    setBar('llm', 3, 'scaricando cervello (≈2 GB, riprendibile)…');
    tlog('llm:resume-start');
    try {
      await brainDownloadMissing((g, t) => { got = g; total = t; });
    } catch (e) {
      // last-resort: legacy whole-file stream (rebuilds cache from byte 0)
      tlog('llm:resume-fallback', String((e && e.message) || e).slice(0, 150));
      stream = await gemmaModelStream((g, t) => { got = g; total = t; }, { cache: true });
      await new Promise(r => setTimeout(r, 500)); // let the final brainPut land
    }
    if (!stream) {
      tlog('llm:resume-complete');
      const fromIDB = await brainCachedStream().catch(() => null);
      if (!fromIDB) throw new Error('cache incompleta dopo il download');
      stream = fromIDB;
    }
  }
  tlog('llm:engine-create-start');
  const t0 = performance.now();
  const tick = setInterval(() => {
    const secs = Math.round((performance.now() - t0) / 1000);
    if (total > 0) {
      const pct = Math.min(95, (got / total) * 95);
      setBar('llm', 5 + pct, 'brain ' + fmtBytes(got) + '/' + fmtBytes(total) + ' ' + (got / total * 100).toFixed(0) + '%  (' + secs + 's)');
    } else {
      setBar('llm', 5, 'loading brain… ' + secs + 's');
    }
  }, 500);
  const engine = await litert.Engine.create({
    model: stream,
    mainExecutorSettings: { maxNumTokens: 8192 },
  });
  clearInterval(tick);
  tlog('llm:engine-create-done');
  await newConversation(engine);
  setBar('llm', 100, 'ready ✓', true);
  tlog('llm:ready');
  const c = $('#chip-llm'); c.textContent = '🧠 brain ✓'; c.classList.add('ok');
}
async function newConversation(engine) {
  if (state.conv && state.conv.close) { try { await state.conv.close(); } catch (e) { /* ignore */ } }
  const conv = await engine.createConversation({
    preface: {
      messages: [{ role: 'system', content: SYSTEM_PROMPT }],
      extra_context: { enable_thinking: false },
    },
    sessionConfig: { maxOutputTokens: 1024 },
  });
  state.conv = conv;
}
async function* streamReply(text) {
  if (state.engine) {
    const stream = state.conv.sendMessageStreaming(text);
    for await (const chunk of stream) {
      for (const item of chunk.content) {
        if (item.type === 'text' && item.text) yield item.text;
      }
    }
    return;
  }
  // v18 PC mode: /chat SSE (llama-casa on the PC, proxy avoids CORS)
  const res = await fetch(api('/chat'), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: TAB_ID, user_text: text }),
  });
  if (!res.ok) throw new Error('/chat HTTP ' + res.status);
  const rd = res.body.getReader();
  const dec = new TextDecoder();
  let buf = '';
  while (true) {
    const { done, value } = await rd.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let i;
    while ((i = buf.indexOf('\n\n')) >= 0) {
      const line = buf.slice(0, i); buf = buf.slice(i + 2);
      if (!line.startsWith('data: ')) continue;
      const ev = JSON.parse(line.slice(6));
      if (ev.type === 'speech') yield ev.text;
      else if (ev.type === 'done') {
        state.pcCorrections = Array.isArray(ev.corrections) ? ev.corrections
          : (ev.corrections && ev.corrections.corrections) || null;
      }
      else if (ev.type === 'error') throw new Error(ev.error || 'llm error');
    }
  }
}

// ---------------- ASR (Whisper small) ----------------
async function loadASR() {
  setBar('asr', 1, 'importing…');
  tlog('asr:import-start');
  const tf = await import('https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0/+esm');
  tlog('asr:import-done');
  tf.env.allowLocalModels = false;
  // jsdelivr +esm default wasmPaths = package root (404) — point it at dist/ explicitly
  tf.env.backends.onnx.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.26.0-dev.20260416-b7804b056c/dist/';
  tf.env.backends.onnx.wasm.numThreads = 1;
  let files = 0;
  const filesSeen = new Set();
  tf.env.progress = (task, name, file, progress) => {
    if (typeof name !== 'string' || !name.includes('whisper')) return;
    if (task === 'download') {
      if (!filesSeen.has(file)) { filesSeen.add(file); files++; tlog('asr:file', 'start ' + file);
      } else if ((progress || 0) >= 100) { tlog('asr:file', 'done ' + file); }
    }
    const pct = files > 1 ? ((files - 1 + (progress || 0) / 100) / files) * 100 : (progress || 0);
    setBar('asr', pct, file ? (file.replace('onnx/', '').replace('.onnx', '')) : '');
  };
  // v13: ORT v4's QDQ optimizer crashes on whisper q8 exports (transformers.js #1707):
  //   "TransposeDQWeightsForMatMulNBits Missing required scale" — the buggy fusion pass
  //   runs at the 'extended' optimization level, so 'basic' skips it (same end result as
  //   the upstream fix: plain DQ+MatMul). Self-correcting chain:
  //   q8/basic → q8/disabled → fp32/basic (fp32 has no DQ nodes → immune).
  const cands = ASR_EP === 'webgpu'
    ? [ ['q8', 'webgpu', 'basic'], ['q8', 'wasm', 'basic'], ['q8', 'wasm', 'disabled'], ['fp32', 'wasm', 'basic'] ]
    : [ ['q8', 'wasm', 'basic'], ['q8', 'wasm', 'disabled'], ['fp32', 'wasm', 'basic'] ];
  let asr = null, won = null;
  for (const [dt, ep, gopt] of cands) {
    tlog('asr:pipeline-start', dt + ' ' + ep + ' gopt=' + gopt);
    try {
      asr = await tf.pipeline('automatic-speech-recognition', WHISPER, {
        dtype: dt, device: ep,
        session_options: { graphOptimizationLevel: gopt },
      });
      tlog('asr:pipeline-done', dt + ' ' + ep + ' gopt=' + gopt);
      tlog('asr:smoke-start');
      const t0 = performance.now();
      await asr(new Float32Array(16000)); // smoke test (1s of silence)
      tlog('asr:smoke-done', (performance.now() - t0).toFixed(0) + 'ms');
      won = dt + '/' + ep + '/' + gopt;
      break;
    } catch (e) {
      console.warn('whisper ' + dt + ' ' + ep + ' gopt=' + gopt + ' failed:', e);
      tlog('asr:epfail', dt + ' ' + ep + ' gopt=' + gopt + ' ' + String((e && e.message) || e).slice(0, 150));
    }
  }
  if (!asr) throw new Error('all asr candidates failed — see odlog asr:epfail lines');
  state.asr = asr;
  state.asrDevice = won;
  setBar('asr', 100, 'ready ✓ ' + won, true);
  const c = $('#chip-asr'); c.textContent = '👂 ears ✓ ' + won; c.classList.add('ok');
}
async function decode16k(blob) {
  const ctx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
  const ab = await blob.arrayBuffer();
  const audio = await ctx.decodeAudioData(ab);
  ctx.close();
  return audio.getChannelData(0);
}
// v22: car mode (?brain=device) — between turns, release the whisper pipeline so its ~300-400 MB
// of WASM memory is reclaimable while the 2 GB brain stays resident. Re-loads on next mic tap.
// v25: PROVEN CRASH PATH — Brave/Android keep the disposed WASM heap, so the reload on the
// 2nd tap allocates a 2nd heap on top of the unreclaimed 1st → OOM. Default is now 'keep'.
function releaseASR() {
  try { if (state.asr && state.asr.dispose) state.asr.dispose(); }
  catch (e) { tlog('asr:release-err', e.message); }
  state.asr = null;
  tlog('asr:released', 'device-mode reclaim');
}
async function asrViaPC(blob) { // v25: ?ears=pc — faster-whisper on the PC, zero WASM in the tab
  const bytes = new Uint8Array(await blob.arrayBuffer());
  let bin = '';
  const CH = 0x8000;
  for (let i = 0; i < bytes.length; i += CH) bin += String.fromCharCode.apply(null, bytes.subarray(i, i + CH));
  tlog('asr:pc-start', (bytes.length / 1024).toFixed(0) + 'KB');
  const res = await fetch(api('/asr'), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ audio_b64: btoa(bin) }),
  });
  if (!res.ok) throw new Error('/asr HTTP ' + res.status);
  const j = await res.json();
  tlog('asr:pc-done', j.ms + 'ms');
  return (j.text || '').replace(/^[\s.]+/, '').trim();
}
// v26: ?ears=web — the phone's own speech recognition (Android/Google), zero WASM
// memory in the tab. Needs mobile data (like any phone call), never the PC.
function webAsrStart() {
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) throw new Error('SpeechRecognition non disponibile su questo browser');
  const rec = new SR();
  rec.lang = 'en-US';
  rec.continuous = true;
  rec.interimResults = true;
  state.webAsr = rec;
  state.webAsrText = '';
  state.webAsrInterim = '';
  rec.onresult = e => {
    let interim = '';
    for (let i = e.resultIndex; i < e.results.length; i++) {
      const r = e.results[i];
      if (r.isFinal) state.webAsrText = (state.webAsrText + ' ' + r[0].transcript).trim();
      else interim += r[0].transcript;
    }
    if (interim) {
      state.webAsrInterim = interim.trim();
      hud('👂 ' + state.webAsrInterim.slice(0, 60) + '…');
    }
  };
  rec.onerror = e => tlog('asr:web-err', e.error || 'unknown');
  rec.onend = () => { // Web Speech auto-stops after ~8 s of silence — restart while recording
    if (state.recording && state.webAsr === rec) {
      try { rec.start(); } catch (_) { /* already running */ }
    }
  };
  rec.start();
}
function webAsrStop() {
  const rec = state.webAsr;
  if (!rec) return Promise.resolve('');
  state.webAsr = null;
  return new Promise(resolve => {
    let done = false;
    const finish = () => { if (!done) { done = true; resolve(state.webAsrText || ''); } };
    rec.onend = finish;
    setTimeout(finish, 3000); // don't hang the turn if onend never lands
    try { rec.stop(); } catch (_) { finish(); }
  });
}
// v27: file-based ASR for sample/test mode. `ears=web` only listens to the live mic
// and cannot take a file, so sample mode always uses this path instead: prefer the PC
// (zero phone memory), else in-app whisper. Works regardless of the active EARS.
async function doASRFile(blob) {
  if (pcAlive === null) await probePC(2500);
  if (pcAlive) { tlog('asr:file-pc'); return asrViaPC(blob); }
  const f = await decode16k(blob);
  if (!state.asr) { hud('👂 scaldando whisper (solo ora, qualche secondo)…'); tlog('asr:file-lazy'); await loadASR(); }
  tlog('asr:file-whisper', (f.length / 16000).toFixed(1) + 's audio');
  const out = await state.asr(f, WHISPER_EN_ONLY ? {} : { language: 'en', task: 'transcribe' });
  return (out.text || '').replace(/^[\s.]+/, '').trim();
}
async function doASR(blob) {
  if (state.sampleMode) return doASRFile(blob); // v27: sample = file-based, any EARS
  if (EARS === 'pc') return asrViaPC(blob); // v25
  if (EARS === 'web') { // v26: transcript was collected live during recording
    tlog('asr:web-done', 'live transcript');
    return (await webAsrStop()).replace(/[.\s]+$/, '').trim();
  }
  const f = await decode16k(blob);
  if (!state.asr) { // v16: lazy load on first mic tap
    hud('👂 scaldando le orecchie (solo la prima volta, qualche secondo)…');
    tlog('asr:lazy-load-start');
    await loadASR();
    tlog('asr:lazy-load-done');
  }
  tlog('asr:infer-start', (f.length / 16000).toFixed(1) + 's audio');
  const t0 = performance.now();
  const out = await state.asr(f, WHISPER_EN_ONLY ? {} : { language: 'en', task: 'transcribe' });
  tlog('asr:infer-done', ((performance.now() - t0) / 1000).toFixed(1) + 's for ' + (f.length / 16000).toFixed(1) + 's audio');
  if (EARS === 'reload' && BRAIN === 'device') releaseASR(); // v22 legacy (A/B via ?ears=reload)
  return (out.text || '').replace(/^[\s.]+/, '').trim();
}

// ---------------- TTS (Supertonic 3) ----------------
function supPreprocess(text) {
  let t = text.normalize('NFKD');
  t = t.replace(/[\u{1F600}-\u{1F64F}\u{1F300}-\u{1F5FF}\u{1F680}-\u{1F6FF}\u{1F700}-\u{1F77F}\u{1F780}-\u{1F7FF}\u{1F800}-\u{1F8FF}\u{1F900}-\u{1F9FF}\u{1FA00}-\u{1FA6F}\u{1FA70}-\u{1FAFF}\u{2600}-\u{26FF}\u{2700}-\u{27BF}\u{1F1E6}-\u{1F1FF}]+/gu, '');
  const rep = { '–': '-', '‑': '-', '—': '-', '_': ' ', '\u201C': '"', '\u201D': '"',
                '\u2018': "'", '\u2019': "'", '´': "'", '`': "'", '[': ' ', ']': ' ',
                '|': ' ', '/': ' ', '#': ' ', '→': ' ', '←': ' ' };
  for (const [k, v] of Object.entries(rep)) t = t.replaceAll(k, v);
  t = t.replace(/[♥☆♡©\\]/g, '');
  t = t.replaceAll('@', ' at ').replaceAll('e.g.,', 'for example, ').replaceAll('i.e.,', 'that is, ');
  t = t.replace(/ ,/g, ',').replace(/ \./g, '.').replace(/ !/g, '!').replace(/ \?/g, '?').replace(/ ;/g, ';').replace(/ :/g, ':').replace(/ '/g, "'");
  while (t.includes('""')) t = t.replace('""', '"');
  while (t.includes("''")) t = t.replace("''", "'");
  while (t.includes('``')) t = t.replace('``', '`');
  t = t.replace(/\s+/g, ' ').trim();
  if (!/[.!?;:,'\"')\]}…。」』】〉》›»]$/.test(t)) t += '.';
  return `<en>${t}</en>`;
}
function supTextIds(p, indexer) {
  const ids = [];
  for (let j = 0; j < p.length; j++) {
    const cp = p.codePointAt(j);
    ids.push(cp < indexer.length ? indexer[cp] : -1);
  }
  return ids;
}
function supNoisyLatent(duration, sampleRate, baseChunkSize, chunkCompress, latentDim) {
  const maxDur = Math.max(...duration);
  const wavLenMax = Math.floor(maxDur * sampleRate);
  const chunkSize = baseChunkSize * chunkCompress;
  const latentLen = Math.floor((wavLenMax + chunkSize - 1) / chunkSize);
  const latentDimVal = latentDim * chunkCompress;
  const xt = [];
  for (let d = 0; d < latentDimVal; d++) {
    const row = [];
    for (let t = 0; t < latentLen; t++) {
      const u1 = Math.max(0.0001, Math.random()), u2 = Math.random();
      row.push(Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2));
    }
    xt.push(row);
  }
  const latentLengths = duration.map(d => Math.floor((Math.floor(d * sampleRate) + chunkSize - 1) / chunkSize));
  const latentMask = latentLengths.map(len => {
    const row = new Array(latentLen).fill(0.0);
    for (let t = 0; t < Math.min(len, latentLen); t++) row[t] = 1.0;
    return [row];
  });
  for (let d = 0; d < latentDimVal; d++)
    for (let t = 0; t < latentLen; t++) xt[d][t] *= latentMask[0][0][t]; // v24: latentMask is [nSpeakers][1][L] — 3 levels, not 2 (was: number *= array → all-NaN audio)
  return { xt, latentMask };
}
async function supInfer(text) {
  const { ort, cfgs, indexer, sessions } = state.tts;
  const p = supPreprocess(text);
  const ids = supTextIds(p, indexer);
  const maxLen = ids.length;
  const textIdsTensor = new ort.Tensor('int64', new BigInt64Array(ids.map(x => BigInt(x))), [1, maxLen]);
  const textMaskTensor = new ort.Tensor('float32', new Float32Array(maxLen).fill(1), [1, 1, maxLen]);
  const { style } = state.tts;

  const dpo = await sessions.dp.run({ text_ids: textIdsTensor, style_dp: style.dp, text_mask: textMaskTensor });
  const duration = Array.from(dpo.duration.data);
  for (let i = 0; i < duration.length; i++) duration[i] /= TTS_SPEED;

  const teo = await sessions.enc.run({ text_ids: textIdsTensor, style_ttl: style.ttl, text_mask: textMaskTensor });
  const textEmb = teo.text_emb;

  const sampleRate = cfgs.ae.sample_rate;
  let { xt, latentMask } = supNoisyLatent(duration, sampleRate, cfgs.ae.base_chunk_size,
                                          cfgs.ttl.chunk_compress_factor, cfgs.ttl.latent_dim); // v22: was const → "Assignment to constant variable"
  const latentMaskTensor = new ort.Tensor('float32', new Float32Array(latentMask.flat(2)), [1, 1, latentMask[0][0].length]);
  const totalStepTensor = new ort.Tensor('float32', new Float32Array([TTS_STEPS]), [1]);

  for (let step = 0; step < TTS_STEPS; step++) {
    const currentStepTensor = new ort.Tensor('float32', new Float32Array([step]), [1]);
    const xtTensor = new ort.Tensor('float32', new Float32Array(xt.flat(2)), [1, xt.length, xt[0].length]);
    const veo = await sessions.ve.run({
      noisy_latent: xtTensor, text_emb: textEmb, style_ttl: style.ttl,
      latent_mask: latentMaskTensor, text_mask: textMaskTensor,
      current_step: currentStepTensor, total_step: totalStepTensor,
    });
    const denoised = Array.from(veo.denoised_latent.data);
    const latentDim = xt.length, latentLen = xt[0].length; // v21: was swapped (xt[0][0].length === undefined)
    xt = [];
    let idx = 0;
    for (let d = 0; d < latentDim; d++) {
      const row = [];
      for (let t = 0; t < latentLen; t++) row.push(denoised[idx++]);
      xt.push(row);
    }
  }
  const finalTensor = new ort.Tensor('float32', new Float32Array(xt.flat(2)), [1, xt.length, xt[0].length]);
  const vo = await sessions.voc.run({ latent: finalTensor });
  return { wav: Float32Array.from(vo.wav_tts.data), duration: duration[0] };
}
async function loadTTS() {
  setBar('tts', 1, 'importing…');
  tlog('tts:import-start');
  const ort = await import('https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.0/+esm');
  tlog('tts:import-done');
  // jsdelivr +esm default wasmPaths = package root (404) — point it at dist/ explicitly
  ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.0/dist/';
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.simd = true;
  setBar('tts', 1, 'downloading…');
  const tiny = q => setBar('tts', 1, q.cached ? 'cached ✓' : fmtBytes(q.loaded));
  const cfgs = await cachedJSON(SUP_HF + '/onnx/tts.json', tiny);
  const indexer = await cachedJSON(SUP_HF + '/onnx/unicode_indexer.json', tiny);
  const onnxNames = SUP_FILES.slice(2);
  const bufs = {};
  for (let i = 0; i < onnxNames.length; i++) {
    const f = onnxNames[i];
    bufs[f] = await cachedArrayBuffer(SUP_HF + '/onnx/' + f, pp => {
      const pct = ((i + (pp.total ? pp.loaded / pp.total : 0)) / onnxNames.length) * 95;
      setBar('tts', pct, f.replace('.onnx', '') + ' · ' + fmtBytes(pp.total || pp.loaded) + (pp.cached ? ' ✓' : ''));
    });
  }
  const styleJson = await cachedJSON(SUP_HF + '/voice_styles/' + VOICE + '.json', q => setBar('tts', 96, VOICE + ' ✓'));
  setBar('tts', 96, 'compiling…');
  const opts = ep => ({ executionProviders: [ep], graphOptimizationLevel: 'all' });
  const make = async ep => ({
    dp:  await ort.InferenceSession.create(bufs['duration_predictor.onnx'], opts(ep)),
    enc: await ort.InferenceSession.create(bufs['text_encoder.onnx'], opts(ep)),
    ve:  await ort.InferenceSession.create(bufs['vector_estimator.onnx'], opts(ep)),
    voc: await ort.InferenceSession.create(bufs['vocoder.onnx'], opts(ep)),
  });
  // v11: EP is configurable (?tts=wasm|webgpu); never fall forward into webgpu
  const epOrder = [TTS_EP, TTS_EP === 'webgpu' ? 'wasm' : 'webgpu'];
  let sessions = null, ep = epOrder[0];
  for (let i = 0; i < epOrder.length; i++) {
    const cand = epOrder[i];
    try { sessions = await make(cand); ep = cand; break; }
    catch (e) {
      console.warn('supertonic ' + cand + ' failed:', e);
      tlog('tts:epfail', cand + ' ' + String((e && e.message) || e).slice(0, 120));
      if (epOrder[i + 1] !== 'wasm') throw e;
    }
  }
  tlog('tts:ep', ep);
  state.tts = {
    ort, cfgs, indexer, sessions, ep,
    style: {
      ttl: new ort.Tensor('float32', new Float32Array(styleJson.style_ttl.data.flat(Infinity)), styleJson.style_ttl.dims),
      dp:  new ort.Tensor('float32', new Float32Array(styleJson.style_dp.data.flat(Infinity)), styleJson.style_dp.dims),
    },
  };
  state.ttsEp = ep;
  setBar('tts', 100, 'ready ✓ (' + ep + ')', true);
  const c = $('#chip-tts'); c.textContent = '🗣️ voice ✓ ' + ep; c.classList.add('ok');
}

// ---------------- audio playback ----------------
let audioCtx = null;
function getAudioCtx() {
  if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  return audioCtx;
}
function playWav(wav, sampleRate) {
  return new Promise(resolve => {
    const ctx = getAudioCtx();
    // v20: await resume — on Android the context starts suspended and stays
    // muted until a resume() lands inside (or right after) a user gesture.
    const doStart = () => {
      const buf = ctx.createBuffer(1, wav.length, sampleRate);
      buf.copyToChannel(wav, 0);
      const src = ctx.createBufferSource();
      src.buffer = buf;
      src.connect(ctx.destination);
      src.onended = () => resolve();
      src.start();
    };
    if (ctx.state === 'suspended') {
      ctx.resume().then(doStart).catch(() => doStart());
    } else doStart();
  });
}
// v26: pipeline — start inferring sentence N+1 WHILE sentence N is playing.
// Phone WASM TTS is ~1.6× slower than realtime, so serial infer-then-play left
// audible gaps and a long tail; overlapping hides most of the infer cost.
function queueSpeak(text) {
  if (!state.ttsQ) state.ttsQ = [];
  state.ttsQ.push(text);
  if (state.ttsPumping) return;
  state.ttsPumping = true;
  state.ttsChain = state.ttsChain.then(async () => {
    let pendingPlay = null;
    try {
      while (state.ttsQ.length) {
        const t = state.ttsQ.shift();
        const t0 = performance.now();
        tlog('tts:infer-start', t.length + ' chars');
        const inferP = supInfer(t).then(r => {
          tlog('tts:infer-done', ((performance.now() - t0) / 1000).toFixed(1) + 's ' + r.wav.length + ' samples');
          return r.wav;
        });
        if (pendingPlay) { // previous sentence is still playing — wait, but don't block the new infer
          await pendingPlay.catch(() => null);
          tlog('tts:play-done', 'ctx=' + getAudioCtx().state);
          pendingPlay = null;
        }
        const wav = await inferP;
        if (turn && turn.firstAudioAt == null) turn.firstAudioAt = performance.now();
        pendingPlay = playWav(wav, state.tts.cfgs.ae.sample_rate);
        if (turn) turn.ttsMs = (turn.ttsMs || 0) + (performance.now() - t0);
      }
    } catch (e) {
      tlog('tts:error', String((e && e.message) || e).slice(0, 200));
      hud('TTS error: ' + (e && e.message || e));
    }
    if (pendingPlay) { await pendingPlay.catch(() => null); tlog('tts:play-done', 'ctx=' + getAudioCtx().state); }
    state.ttsPumping = false;
  }).catch(() => {});
}

// ---------------- corrections parsing (tolerant) ----------------
function parseCorrections(full) {
  const m = full.match(/CORRECTIONS\s*:\s*(\[[\s\S]*\])/i);
  if (!m) return null;
  try {
    const arr = JSON.parse(m[1]);
    return Array.isArray(arr) ? arr : null;
  } catch (e) {
    const objs = [];
    const re = /\{[^{}]*\}/g;
    let mm;
    while ((mm = re.exec(m[1]))) {
      try { objs.push(JSON.parse(mm[0])); } catch (e2) { /* skip bad object */ }
    }
    return objs.length ? objs : null;
  }
}
function visiblePart(full) {
  const i = full.search(/CORRECTIONS\s*:/i);
  return (i >= 0 ? full.slice(0, i) : full);
}

// ---------------- UI ----------------
function addBubble(role, text) {
  const el = document.createElement('div');
  el.className = 'bub ' + role;
  el.textContent = text;
  $('#log').appendChild(el);
  $('#log').scrollTop = $('#log').scrollHeight;
  return el;
}
function renderCorrections(corr) {
  const el = document.createElement('div');
  el.className = 'card';
  el.innerHTML = '<h3>Corrections</h3>';
  if (!corr || !corr.length) {
    el.insertAdjacentHTML('beforeend', '<div class="sys" style="align-self:flex-start">no clear error 👌</div>');
  } else {
    for (const c of corr) {
      const row = document.createElement('div');
      row.className = 'row';
      row.innerHTML = '<div><span class="bad"></span></div><div><span class="good"></span></div>';
      row.querySelector('.bad').textContent = c.you_said || '';
      row.querySelector('.good').textContent = c.better || '';
      if (c.why) { const w = document.createElement('div'); w.className = 'why'; w.textContent = c.why; row.appendChild(w); }
      if (c.it) { const i = document.createElement('div'); i.className = 'it'; i.textContent = c.it; row.appendChild(i); }
      el.appendChild(row);
    }
  }
  $('#log').appendChild(el);
  $('#log').scrollTop = $('#log').scrollHeight;
}

// ---------------- settings / setup (v27) ----------------
let selPreset = CFG.preset;
function $set(id) { return document.getElementById(id); }
function markPreset(p) {
  $set('p-casa').classList.toggle('sel', p === 'casa');
  $set('p-macchina').classList.toggle('sel', p === 'macchina');
}
function showSettings(firstTime) {
  selPreset = CFG.preset;
  $set('pcurl').value = CFG.pc || '';
  $set('f-ears').value = EARS;
  $set('f-steps').value = TTS_STEPS;
  $set('f-recmax').value = REC_MAX;
  markPreset(selPreset);
  $set('settings').classList.add('show');
  const dot = $set('pcdot'), st = $set('pcstate');
  st.textContent = 'verifica…'; dot.className = '';
  probePC(3000).then(ok => { dot.className = ok ? 'ok' : 'bad'; st.textContent = ok ? 'raggiungibile ✓' : 'non raggiungibile'; });
  if (firstTime) hud('scelta iniziale: scegli Casa o Macchina, poi «Salva e riavvia»');
}
function closeSettings() {
  $set('settings').classList.remove('show');
  if (!readStoredCfg().preset) saveCfg({ preset: selPreset }); // first launch: keep the default, don't re-prompt
}
function pickPreset(p) { selPreset = p; markPreset(p); }
function applySettings() {
  saveCfg({
    preset: selPreset,
    pc: $set('pcurl').value.trim(),
    ears: $set('f-ears').value,
    steps: parseInt($set('f-steps').value, 10) || 5,
    recmax: parseInt($set('f-recmax').value, 10) || 30,
  });
  location.reload();
}

// ---------------- sample test mode (v27) ----------------
// Runs a canned WAV through the FULL pipeline (ASR → LLM → TTS) with no mic. The
// ASR step is file-based (doASRFile) so it works in every EARS mode.
function bindSample() {
  const el = $set('sample');
  if (!el) return;
  el.addEventListener('change', function () {
    const v = el.value;
    if (!v) return;
    if (state.busy || !state.ready) { el.value = ''; hud('in attesa — riprova'); return; }
    (async () => {
      try { const c = getAudioCtx(); if (c.state !== 'running') c.resume().catch(() => {}); } catch (_) {}
      hud('🎧 sample → caricando wav…');
      const r = await fetch(v);
      if (!r.ok) { hud('sample: HTTP ' + r.status); el.value = ''; return; }
      const blob = await r.blob();
      el.value = '';
      state.busy = true; state.sampleMode = true;
      $('#mic').disabled = true;
      hud('🎧 sample → ASR…');
      doTurn(blob);
    })();
  });
}

// ---------------- diagnostics (v27) ----------------
// In-app full-screen diagnostics — deliberately NOT a separate tab (two tabs = OOM
// on this phone). Read-only: IDB is counted by keys (no 256 MB chunk read),
// network probes are tiny Range requests. The goal is a copy-paste report.
let lastDiag = '';
function showDiag() { $set('diag').classList.add('show'); runDiag(); }
function closeDiag() { $set('diag').classList.remove('show'); }
async function diagFetchTimed(url, opts) {
  const t0 = performance.now();
  try {
    const r = await fetch(url, opts);
    if (opts && opts.headers && opts.headers.Range) await r.arrayBuffer().catch(() => {});
    return { ok: r.ok, status: r.status, ms: Math.round(performance.now() - t0) };
  } catch (e) {
    return { ok: false, status: 0, ms: Math.round(performance.now() - t0), err: String((e && e.message) || e).slice(0, 80) };
  }
}
async function diagBrainKeys() {
  try {
    const db = await brainIDB();
    const keys = await new Promise((res, rej) => {
      const rq = db.transaction('files', 'readonly').objectStore('files').getAllKeys();
      rq.onsuccess = () => res(rq.result); rq.onerror = () => rej(rq.error);
    });
    const n = keys.filter(k => String(k).startsWith(BRAIN_KEY + ':')).length;
    return n + '/' + BRAIN_N + ' chunk in IDB (' + (n * BRAIN_CHUNK / 1073741824).toFixed(2) + ' GB)';
  } catch (e) { return 'err: ' + e.message; }
}
function diagMem() {
  const m = typeof performance !== 'undefined' && performance.memory;
  if (!m) return 'performance.memory non esposto qui — il heartbeat odlog manda js=…MB ogni 10 s';
  const g = x => (x / 1048576).toFixed(0) + ' MB';
  return 'usato ' + g(m.usedJSHeapSize) + ' / totale ' + g(m.totalJSHeapSize) + ' / limite ' + g(m.jsHeapSizeLimit);
}
async function runDiag() {
  const el = $set('diagbody');
  el.textContent = 'raccolta dati…';
  const t = [];
  const P = s => t.push(s);
  try {
    P('══ AppEnglishTutor · diagnostica v' + VERSION + ' ══');
    P('tab=' + TAB_ID + ' · ' + new Date().toLocaleString());
    P('url=' + location.href);
    P('ua=' + navigator.userAgent);
    P('');
    P('─ Dispositivo ─');
    P('RAM dichiarata: ' + (navigator.deviceMemory || '?') + ' GB · CPU logiche: ' + navigator.hardwareConcurrency);
    P('screen ' + screen.width + 'x' + screen.height + '@' + (window.devicePixelRatio || 1) + ' · secure=' + isSecureContext + ' · online=' + navigator.onLine);
    try { const c = getAudioCtx(); P('AudioContext: ' + (c.sampleRate || '?') + ' Hz'); } catch (e) { P('AudioContext: err ' + e.message); }
    try {
      const cv = document.createElement('canvas');
      const gl = cv.getContext('webgl2') || cv.getContext('webgl');
      P('WebGL: ' + (gl ? gl.getParameter(gl.VERSION) : 'no'));
    } catch (e) { P('WebGL: no (' + e.message + ')'); }
    try {
      if (!navigator.gpu) P('WebGPU: no (property assente)');
      else { const a = await navigator.gpu.requestAdapter(); P('WebGPU: ' + (a ? 'ok (adapter)' : 'no adapter')); }
    } catch (e) { P('WebGPU: err ' + e.message); }
    P('');
    P('─ Capacità ─');
    P('SpeechRecognition (orecchie web): ' + (!!(window.SpeechRecognition || window.webkitSpeechRecognition) ? 'sì' : 'NO')); 
    P('IndexedDB: ' + (!!window.indexedDB ? 'sì' : 'NO') + ' · Cache API: ' + (!!window.caches ? 'sì' : 'NO'));
    let sw = 'no SW';
    try {
      const reg = await navigator.serviceWorker.getRegistration();
      sw = reg ? 'attivo (' + (reg.active ? (reg.active.scriptURL || '').split('/').pop() : 'pending') + ', scope ' + reg.scope + ')' : 'non registrato';
    } catch (e) { sw = 'err ' + e.message; }
    P('Service worker: ' + sw);
    P('Cervello in IDB: ' + (await diagBrainKeys()));
    P('');
    P('─ Memoria (JS heap) ─');
    P(diagMem());
    P('uptime pagina: ' + (performance.now() / 1000).toFixed(0) + ' s');
    P('');
    P('─ Config (v27) ─');
    P('preset=' + CFG.preset + ' · brain=' + BRAIN + ' · ears=' + EARS + ' · steps=' + TTS_STEPS + ' · recmax=' + REC_MAX);
    P('tts_ep=' + TTS_EP + ' · asr_ep=' + ASR_EP + ' · whisper=' + WHISPER);
    P('pc_url="' + (CFG.pc || '(vuoto = stesso sito / dev)') + '"');
    P('stored=' + JSON.stringify(readStoredCfg()));
    const ov = [];
    for (const k of ['brain', 'ears', 'steps', 'tts', 'asr', 'recmax', 'whisper', 'pc']) if (URLP.get(k)) ov.push(k + '=' + URLP.get(k));
    P('url-override: ' + (ov.length ? ov.join(' ') : 'nessuno'));
    P('');
    P('─ Rete ─');
    const t0 = performance.now();
    try {
      const r = await fetch(api('/health'), { signal: AbortSignal.timeout(4000) });
      const j = await r.json().catch(() => null);
      P('PC /health: HTTP ' + r.status + ' in ' + Math.round(performance.now() - t0) + ' ms' + (j ? ' · ' + JSON.stringify(j).slice(0, 140) : ''));
    } catch (e) {
      P('PC /health: FALLITO in ' + Math.round(performance.now() - t0) + ' ms → ' + String((e && e.message) || e).slice(0, 60) + (CFG.pc ? ' (URL PC non raggiungibile)' : ' (nessun URL PC = previsto fuori casa)'));
    }
    const cdn = await diagFetchTimed(GEMMA_URL, { headers: { Range: 'bytes=0-0' } });
    P('HF CDN (fonte cervello): ' + (cdn.ok ? 'HTTP ' + cdn.status + ' in ' + cdn.ms + ' ms ✓' : 'FALLITO ' + (cdn.err || 'HTTP ' + cdn.status)));
    const sm = await diagFetchTimed('samples/supertonic_F1_en.wav', { headers: { Range: 'bytes=0-1023' } });
    P('Sample locale (pipeline test): ' + (sm.ok ? 'HTTP ' + sm.status + ' in ' + sm.ms + ' ms ✓' : 'FALLITO ' + (sm.err || 'HTTP ' + sm.status)));
    P('');
    P('─ Modelli (stato live) ─');
    P('stage=' + state.stage + ' · ready=' + state.ready + ' · busy=' + state.busy);
    P('LLM: ' + (BRAIN === 'pc' ? 'sul PC (via /chat, Qwen 27B)' : state.engine ? 'caricato in-app' + (state.llmFromCache ? ' [da IDB cache]' : '') : 'non caricato'));
    P('ASR: ' + (state.asr ? 'whisper carico (' + state.asrDevice + ')' : 'non caricato — lazy al 1° tap') + (EARS === 'web' ? ' · attivo: Web Speech (no whisper)' : ''));
    P('TTS: ' + (state.tts ? 'Supertonic carico (' + (state.ttsEp || '?') + ')' : 'non caricato'));
    P('');
    P('Gate test: UNA sola tab · preset 🚗 Macchina · 10 turni + 2 sample 🎧 · nessun crash.');
    P('Se qualcosa va storto: invia questo report a coach (pulsante 📋 report).');
  } catch (e) {
    P('⚠️ errore durante la raccolta: ' + String((e && e.stack) || e).slice(0, 400));
  }
  lastDiag = t.join('\n');
  el.textContent = lastDiag;
}
async function copyDiag() {
  if (!lastDiag) { hud('prima ↻ ripeti'); await runDiag(); }
  try {
    await navigator.clipboard.writeText(lastDiag);
    hud('report copiato ✓ — incollalo qui');
  } catch (e) {
    const el = $set('diagbody');
    try { el.focus(); el.select && el.select(); } catch (_) {}
    hud('copia automatica non riuscita — long-press sul testo, Copia, incolla qui');
  }
}

// ---------------- recording ----------------
function pickMime() {
  const c = ['audio/webm;codecs=opus', 'audio/webm', 'audio/mp4', 'audio/ogg;codecs=opus'];
  for (const m of c) if (window.MediaRecorder && MediaRecorder.isTypeSupported(m)) return m;
  return '';
}
async function startRec() {
  if (EARS === 'web' && !((window.SpeechRecognition || window.webkitSpeechRecognition)))
    throw new Error('SpeechRecognition non disponibile su questo browser — usa ?ears=pc');
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: { echoCancellation: true, noiseSuppression: true }
  });
  const mime = pickMime();
  const rec = new MediaRecorder(stream, mime ? { mimeType: mime } : undefined);
  state.chunks = [];
  rec.ondataavailable = e => { if (e.data.size) state.chunks.push(e.data); };
  state.stream = stream;
  state.rec = rec;
  state.recording = true;
  $('#mic').classList.add('rec');
  rec.start();
  if (EARS === 'web') webAsrStart(); // v26: recognize live in parallel (transcript, not the webm)
}
function stopRec() {
  const rec = state.rec;
  const stream = state.stream;
  state.recording = false;
  $('#mic').classList.remove('rec');
  rec.stop();
  if (EARS === 'web' && state.webAsr) { try { state.webAsr.onend = () => {}; } catch (_) {} } // v26: no auto-restart
  return new Promise(resolve => {
    rec.onstop = () => {
      stream.getTracks().forEach(t => t.stop());
      state.stream = null;
      resolve(new Blob(state.chunks, { type: rec.mimeType || 'audio/webm' }));
    };
  });
}

// ---------------- turn flow ----------------
function startTurn(tStop) {
  turn = {
    tStop, asrMs: null, llmFirstAt: null, firstAudioAt: null,
    full: '', pending: '', liveEl: null,
  };
}
async function doTurn(blob) {
  const tStop = performance.now();
  startTurn(tStop);
  hud('👂 listening to you…');
  let text = '';
  try {
    text = await doASR(blob);
  } catch (e) {
    console.error(e);
    tlog('asr:error', String((e && e.message) || e).slice(0, 200));
    hud('ASR error: ' + e.message);
    state.busy = false;
    $('#mic').disabled = false;
    return;
  }
  turn.asrMs = performance.now() - tStop;
  if (!text) {
    addBubble('sys', 'didn’t catch that — tap the mic and try again');
    state.busy = false;
    $('#mic').disabled = false;
    return;
  }
  addBubble('you', text);
  turn.liveEl = addBubble('coach', '…');

  try {
    const gen = streamReply(text);
    for await (const piece of gen) {
      if (!turn.llmFirstAt) turn.llmFirstAt = performance.now();
      turn.full += piece;
      turn.liveEl.textContent = visiblePart(turn.full).trim() || '…';
      // flush all complete sentences to TTS as they arrive
      turn.pending += piece;
      let m;
      while ((m = turn.pending.match(/^[\s\S]*?[.!?](?:["')\]]+)?\s+/))) {
        const s = m[0].trim();
        turn.pending = turn.pending.slice(m[0].length);
        if (s.length > 1) queueSpeak(s);
      }
      $('#log').scrollTop = $('#log').scrollHeight;
    }
  } catch (e) {
    console.error('llm error', e);
    turn.liveEl.textContent = visiblePart(turn.full).trim() + '\n⚠️ (generation stopped: ' + e.message + ')';
  }
  // final flush (strip CORRECTIONS part)
  let tail = turn.pending.replace(/CORRECTIONS[\s\S]*$/i, '').trim();
  if (tail.length > 1) queueSpeak(tail);
  turn.liveEl.textContent = visiblePart(turn.full).trim() || '…';

  await state.ttsChain;
  const corr = state.engine ? parseCorrections(turn.full) : state.pcCorrections;
  state.pcCorrections = null;
  renderCorrections(corr);
  hud('');
  hud('ASR ' + (turn.asrMs / 1000).toFixed(1) + 's · LLM ' +
      (turn.llmFirstAt ? (turn.llmFirstAt - turn.tStop) / 1000 : 0).toFixed(1) + 's · ' +
      'audio ' + (turn.firstAudioAt ? (turn.firstAudioAt - turn.tStop) / 1000 : 0).toFixed(1) + 's · ' +
      'TTS ' + ((turn.ttsMs || 0) / 1000).toFixed(1) + 's');
  state.busy = false;
  $('#mic').disabled = false;
}

// ---------------- mic button ----------------
$('#mic').addEventListener('click', async () => {
  if (state.busy || !state.ready) return;
  state.sampleMode = false; // mic turn: clear any sample flag
  // v20: unlock audio playback INSIDE the user gesture (Android autoplay policy).
  try { const c = getAudioCtx(); if (c.state !== 'running') c.resume().catch(() => {}); } catch (_) {}
  if (!state.recording) {
    try {
      await startRec();
      hud('🔴 registrazione — tocca per fermare (max ' + REC_MAX + ' s)');
      const t0 = Date.now();
      recTimer = setInterval(() => {
        const left = REC_MAX - Math.floor((Date.now() - t0) / 1000);
        if (left <= 0) { hud('⏱ ' + REC_MAX + ' s — elaboro…'); stopAndProcess(); }
        else hud('🔴 ' + left + 's — tocca per fermare' + (EARS === 'web' && state.webAsrInterim ? ' · 👂 ' + state.webAsrInterim.slice(0, 40) : ''));
      }, 1000);
    } catch (e) {
      hud('mic error: ' + e.message);
    }
  } else {
    stopAndProcess();
  }
});
// v17: 15 s auto-stop — caps ASR activation memory and keeps the conversation snappy.
let recTimer = null;
function stopAndProcess() {
  if (!state.recording) return;
  if (recTimer) { clearInterval(recTimer); recTimer = null; }
  (async () => {
    const blob = await stopRec();
    if (blob.size < 16000) { hud('too short'); return; }
    state.busy = true;
    $('#mic').disabled = true;
    doTurn(blob);
  })();
}

// ---------------- new call ----------------
window.newCall = async function () {
  if (!state.ready || state.busy) return;
  try { const c = getAudioCtx(); if (c.state !== 'running') c.resume().catch(() => {}); } catch (_) {}
  $('#log').innerHTML = '';
  if (state.engine) await newConversation(state.engine);
  else fetch(api('/reset'), { method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: TAB_ID }) }).catch(() => {});
  addBubble('coach', SCENARIO.opening);
  queueSpeak(SCENARIO.opening);
  hud('new call');
};

// ---------------- init ----------------
async function init() {
  tlog('init-start', 'v' + VERSION + ' tab=' + TAB_ID + ' brain=' + BRAIN + ' ears=' + EARS + ' ua=' + navigator.userAgent.slice(0, 80) + ' tts=' + TTS_EP + ' asr=' + ASR_EP);
  state.cache = await caches.open(CACHE_NAME);
  state.stage = 'init';
  startHeartbeat();
  bindSample(); // v27: canned-WAV pipeline test (no mic)
  if (!readStoredCfg().preset) showSettings(true); // v27: first-launch setup (Casa/Macchina)
  await waitOwnership(); // v16: exactly one tab loads the models
  state.stage = 'llm';
  hud('od.js v' + VERSION + ' — ' + (BRAIN === 'pc' ? 'cervello in PC, carico istantaneo…' : 'importando il cervello (2 GB, la prima volta)…') + ' ');
  // v27: model status panel — where each model runs + its source
  const ml = document.querySelectorAll('#loadbox .mlab');
  if (ml[0]) ml[0].textContent = '🧠 Coach — ' + (BRAIN === 'pc' ? 'Qwen 27B (PC)' : 'Gemma 4 E2B (telefono)');
  if (ml[1]) ml[1].textContent = '🗣️ Voice — Supertonic F1 (telefono)';
  if (ml[2]) ml[2].textContent = '👂 Ears — ' + (EARS === 'web' ? 'native (Web Speech)' : EARS === 'pc' ? 'whisper (PC)' : 'whisper ' + WHISPER.split('/').pop() + ' (telefono)');
  // scenario: sync name/opening from server; keep built-in system prompt (server has none for this protocol)
  fetch(api('/scenario')).then(r => r.ok ? r.json() : null)
    .then(s => {
      if (!s) return;
      if (s.name) { SCENARIO.name = s.name; $('#scenario').textContent = s.name; }
      if (s.opening) SCENARIO.opening = s.opening;
    })
    .catch(() => {});
  $('#scenario').textContent = SCENARIO.name;

  const errors = [];
  // v18: brain on the PC by default (no 2 GB in the phone). TTS on device always.
  // v16: ASR is DEFERRED to the first mic tap.
  const loads = [];
  if (BRAIN === 'device') {
    loads.push(['llm', loadLLM]);
  } else {
    tlog('llm:pc-mode', 'llama-casa via /chat');
    setBar('llm', 100, 'ready ✓ (PC llama-casa)', true);
    const c = $('#chip-llm'); c.textContent = '🧠 brain ✓ (PC)'; c.classList.add('ok');
  }
  loads.push(['tts', loadTTS]);
  for (const [k, fn] of loads) {
    state.stage = k;
    try { await fn(); tlog('stage-ok', k); }
    catch (e) {
      const m = (e && (e.message || e.reason)) || String(e); tlog(k + ':fail', String(m).slice(0, 300)); errors.push([k, e]);
      if (k === 'llm' && state.llmFromCache) { tlog('llm:cache-invalidate'); brainDeleteAll().catch(() => {}); state.llmFromCache = false; } // v14
    }
  }
  tlog('asr:deferred', 'loads on first mic tap');
  setBar('asr', 0, 'standby — si scalda al 1° tap del microfono', false);
  tlog('init-settled', 'errors=' + (errors.length ? errors.map(e => e[0]).join('|') : 'none'));
  if (errors.length) {
    for (const [k, e] of errors) console.error(k, e);
    state.stage = 'init-errors';
    hud('❌ ' + errors.map(e => e[0]).join(', ') + ' non si sono caricati — leggi la riga con ❌ ("Failed to fetch" = Wi-Fi/internet; in tal caso ricarica la pagina).');
    return;
  }
  state.ready = true;
  state.stage = 'ready';
  $('#loadbox').style.display = 'none';
  $('#mic').disabled = false;
  const net = $('#chip-net');
  net.textContent = '📶 ' + (navigator.onLine ? 'online' : 'offline');
  net.classList.add('ok');
  window.addEventListener('online', () => { net.textContent = '📶 online'; });
  window.addEventListener('offline', () => { net.textContent = '📶 offline (still works)'; });
  addBubble('coach', SCENARIO.opening);
  queueSpeak(SCENARIO.opening);
  hud('ready — tap the mic and say something in English');
}
init();

// v23: server-side notice — the PC can push a message to the phone. Shown in the log
// on every page load (this is how you hear about new builds / test instructions).
(async () => {
  try {
    const r = await fetch(api('/notice'));
    const j = await r.json();
    const t = (j && j.text || '').trim();
    if (t) {
      tlog('notice', t.slice(0, 300));
      const el = document.getElementById('log');
      if (el) el.insertAdjacentHTML('beforeend', '<div class="sys">📩 ' + t.replace(/</g, '&lt;') + '</div>');
    }
  } catch (_) {}
})();

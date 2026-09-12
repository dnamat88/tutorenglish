/* English Tutor on-device — service worker (v23)
 *
 * Purpose: make the app work with ZERO network (car mode).
 *
 * Strategy:
 *  - App shell (/od.html, /od.js): NETWORK-FIRST with cache fallback.
 *    → updates flow while online; offline serves the last known version.
 *  - Pinned CDN assets (jsdelivr: onnxruntime-web@1.17.0 + the pinned dev build
 *    used by transformers.js, transformers@4.2.0, @litert-lm/core@0.17.0 —
 *    ESM modules + wasm binaries): CACHE-FIRST.
 *    → URLs are version-pinned, files are large, and they never change.
 *  - Hugging Face models: NOT cached here. The app already caches them in its
 *    own Cache Storage ('et-od-v3') and the brain in IndexedDB. The SW just
 *    passes those requests through; offline the app's own caches are hit.
 *  - /notice: network-first (it is the ops messaging channel — must stay fresh
 *    while online), cache fallback when offline.
 *  - All POSTs (/odlog, /chat, /asr, /reset): passthrough, never cached.
 */
const SHELL_CACHE = 'et-od-shell-v2';;
const CDN_CACHE = 'et-od-cdn-v1';

self.addEventListener('install', (e) => {
  e.waitUntil((async () => {
    const c = await caches.open(SHELL_CACHE);
    for (const u of ['/od.html', '/od.js', '/sw.js']) {
      try { await c.add(new Request(u, { mode: 'same-origin' })); } catch (_) {}
    }
    self.skipWaiting();
  })());
});

self.addEventListener('activate', (e) => {
  e.waitUntil((async () => {
    const keep = new Set([SHELL_CACHE, CDN_CACHE]);
    for (const name of await caches.keys()) if (!keep.has(name)) await caches.delete(name);
    await self.clients.claim();
  })());
});

function isPinnedCdn(u) {
  if (u.hostname !== 'cdn.jsdelivr.net') return false;
  return u.pathname.startsWith('/npm/onnxruntime-web@1.17.0') ||
         u.pathname.startsWith('/npm/onnxruntime-web@1.26.0-dev.20260416-b7804b056c') ||
         u.pathname.startsWith('/npm/@huggingface/transformers@4.2.0') ||
         u.pathname.startsWith('/npm/@litert-lm/core@0.17.0');
}

self.addEventListener('fetch', (e) => {
  const req = e.request;
  if (req.method !== 'GET') return;
  const url = new URL(req.url);

  // 1) Pinned CDN: cache-first.
  if (isPinnedCdn(url)) {
    e.respondWith((async () => {
      const c = await caches.open(CDN_CACHE);
      const hit = await c.match(req);
      if (hit) return hit;
      const res = await fetch(req);
      if (res.ok) c.put(req, res.clone());
      return res;
    })());
    return;
  }

  // 2) Shell + notice: network-first, cache fallback (strip ?v= on the
  //    fallback so an older cached build still boots offline).
  const p = url.pathname;
  if (url.origin === self.location.origin &&
      (p === '/od.html' || p === '/od.js' || p === '/notice')) {
    e.respondWith((async () => {
      const c = await caches.open(SHELL_CACHE);
      try {
        const res = await fetch(req);
        if (res.ok) {
          // v2.1: cache under the EXACT url AND the bare pathname, so a
          // partial install (e.g. /od.html cached but /od.js not) repairs
          // itself on the next online load. Both keys stay in sync.
          const bare = new Request(url.pathname, { mode: 'same-origin' });
          const clone = res.clone();
          c.put(req, clone.clone());
          c.put(bare, clone);
        }
        return res;
      } catch (_) {
        let hit = await c.match(req);
        if (!hit) hit = await c.match(new Request(url.pathname, { mode: 'same-origin' }));
        if (hit) return hit;
        return new Response('offline', { status: 503 });
      }
    })());
  }
  // everything else (HF models, /models/brain.litertlm, POSTs, ...): passthrough.
});

// service-worker.js
const APP_VERSION = 'ndpwj-v1.2.0';
const APP_SHELL = [
  './',
  './index.html',
  './dist/styles.css',
  './manifest.webmanifest',
  './icons/icon-192.png',
  './icons/icon-512.png',
  './icons/maskable-192.png',
  './icons/maskable-512.png'
];

self.addEventListener('install', (event) => {
  event.waitUntil(caches.open(APP_VERSION).then((cache) => cache.addAll(APP_SHELL)));
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keys) => Promise.all(keys.map((k) => (k !== APP_VERSION ? caches.delete(k) : null))))
  );
  self.clients.claim();
});

self.addEventListener('fetch', (event) => {
  const req = event.request;

  if (req.mode === 'navigate') {
    event.respondWith((async () => {
      try { return await fetch(req); }
      catch {
        const cache = await caches.open(APP_VERSION);
        return (await cache.match('./index.html')) || Response.error();
      }
    })());
    return;
  }

  event.respondWith((async () => {
    const cache = await caches.open(APP_VERSION);
    const cached = await cache.match(req);
    const fetched = fetch(req).then((res) => {
      if (res && res.status === 200 && (res.type === 'basic' || res.type === 'opaque')) {
        cache.put(req, res.clone());
      }
      return res;
    }).catch(() => null);

    return cached || fetched || Response.error();
  })());
});

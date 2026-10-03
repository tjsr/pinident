/// <reference lib="webworker" />

const sw = self as unknown as ServiceWorkerGlobalScope;

const CACHE = 'pinident-mobile-v2';
const SHELL = ['/', '/index.html', '/styles.css', '/app.js', '/library.js',
  '/scanner.js', '/catalog.js', '/manifest.webmanifest', '/icon.svg',
  '/icon-192.png', '/icon-512.png'];

sw.addEventListener('install', event => {
  event.waitUntil(caches.open(CACHE).then(cache => cache.addAll(SHELL)));
  void sw.skipWaiting();
});

sw.addEventListener('activate', event => {
  event.waitUntil(Promise.all([
    caches.keys().then(keys => Promise.all(keys.filter(key => key.startsWith('pinident-mobile-') && key !== CACHE)
      .map(key => caches.delete(key)))),
    sw.clients.claim()
  ]));
});

sw.addEventListener('fetch', event => {
  const url = new URL(event.request.url);
  if (event.request.method !== 'GET' || url.origin !== sw.location.origin ||
      url.pathname === '/catalog.json') return;
  event.respondWith(fetch(event.request).then(response => {
    if (response.ok) {
      const copy = response.clone();
      void caches.open(CACHE).then(cache => cache.put(event.request, copy));
    }
    return response;
  }).catch(async () => {
    const cached = await caches.match(event.request);
    return cached ?? (event.request.mode === 'navigate'
      ? await caches.match('/index.html') ?? Response.error() : Response.error());
  }));
});

/// <reference lib="webworker" />

function isServiceWorker(scope: WorkerGlobalScope): scope is ServiceWorkerGlobalScope {
  return 'clients' in scope && 'skipWaiting' in scope &&
    typeof scope.skipWaiting === 'function';
}
if (!isServiceWorker(self)) throw new Error('This script requires a service worker.');
const sw: ServiceWorkerGlobalScope = self;

const CACHE = 'pinident-mobile-v15';
const SHELL = ['/', '/index.html', '/styles.css', '/app.js', '/library.js',
  '/scanner.js', '/stage_geometry.js', '/scan_session.js', '/model.js', '/identity.js',
  '/catalog.js', '/reviews.js', '/json_input.js',
  '/manifest.webmanifest', '/icon.svg',
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
      url.pathname === '/catalog.json' || url.pathname.startsWith('/api/')) return;
  event.respondWith(fetch(event.request).then(response => {
    if (response.ok && (!['/model.json', '/identity-index.json'].includes(url.pathname) ||
        response.headers.get('content-type')?.includes('application/json'))) {
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

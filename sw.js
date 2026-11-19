// Offline support for the UVS scanner.
// App files and card data: network first (so updates arrive), cached copy when offline.
// Text-reading library and its language data (from CDNs): cache first, fetched once.
// Card pictures: cache first, filled as they are viewed.
const VERSION = "uvs-20261119-134058";
const SHELL = ["uvs_scanner.html", "fingerprint.js", "manifest.webmanifest", "icons/icon-192.png", "icons/icon-512.png",
  "data/catalog.json", "data/fingerprints.json"];

self.addEventListener("install", (e) => {
  e.waitUntil(caches.open(VERSION).then((c) => c.addAll(SHELL)).then(() => self.skipWaiting()));
});
self.addEventListener("activate", (e) => {
  e.waitUntil(caches.keys().then((keys) => Promise.all(keys.filter((k) => k !== VERSION).map((k) => caches.delete(k))))
    .then(() => self.clients.claim()));
});

const isLib = (u) => /cdn\.jsdelivr\.net|unpkg\.com|tessdata|projectnaptha/.test(u.host + u.pathname);
const isPicture = (u) => /\.(webp|jpe?g|png)$/i.test(u.pathname) && !u.pathname.includes("/icons/");

self.addEventListener("fetch", (e) => {
  const req = e.request;
  if (req.method !== "GET") return;
  const u = new URL(req.url);
  if (isLib(u) || isPicture(u)) {
    e.respondWith(caches.open(VERSION).then(async (c) => {
      const hit = await c.match(req);
      if (hit) return hit;
      const res = await fetch(req);
      if (res.ok || res.type === "opaque") c.put(req, res.clone());
      return res;
    }));
    return;
  }
  if (u.origin !== location.origin) return;
  e.respondWith(fetch(req).then((res) => {
    if (res.ok) caches.open(VERSION).then((c) => c.put(req, res.clone()));
    return res;
  }).catch(() => caches.match(req, { ignoreSearch: true })));
});

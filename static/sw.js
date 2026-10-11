/* Service Worker: офлайн-оболочка с корня сайта (scope /) */
const CACHE = "photone-shell-v1";
const PRECACHE = [
  "/offline",
  "/static/offline.css",
  "/static/offline.js",
  "/static/favicon.svg",
  "/static/favicon.ico",
];

async function putOpaqueOk(cache, url, res) {
  const headers = new Headers(res.headers);
  headers.delete("set-cookie");
  headers.set("Cache-Control", "max-age=86400");
  headers.set("Content-Type", headers.get("Content-Type") || "text/html; charset=utf-8");
  await cache.put(url, new Response(await res.blob(), { status: 200, statusText: "OK", headers }));
}

self.addEventListener("install", (event) => {
  event.waitUntil(
    (async () => {
      const cache = await caches.open(CACHE);
      for (const url of PRECACHE) {
        const res = await fetch(url, { cache: "reload" });
        if (!res.ok) throw new Error("precache failed " + url);
        await putOpaqueOk(cache, url, res);
      }
      await self.skipWaiting();
    })()
  );
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k)))
    ).then(() => self.clients.claim())
  );
});

self.addEventListener("fetch", (event) => {
  const req = event.request;
  if (req.method !== "GET") return;
  const url = new URL(req.url);
  if (url.origin !== self.location.origin) return;

  if (url.pathname.startsWith("/api/")) {
    event.respondWith(
      fetch(req).catch(
        () =>
          new Response(JSON.stringify({ ok: false, offline: true }), {
            status: 503,
            headers: { "Content-Type": "application/json" },
          })
      )
    );
    return;
  }

  if (req.mode === "navigate") {
    event.respondWith(
      fetch(req)
        .then((res) => res)
        .catch(async () => {
          const cached = await caches.match("/offline", { ignoreSearch: true });
          return (
            cached ||
            new Response(
              "<!DOCTYPE html><meta charset=utf-8><title>Photone</title><h1>Офлайн</h1><p>Откройте launcher.html в папке проекта или start-tray.bat</p>",
              { headers: { "Content-Type": "text/html; charset=utf-8" } }
            )
          );
        })
    );
    return;
  }

  event.respondWith(
    caches.match(req, { ignoreSearch: true }).then(
      (hit) =>
        hit ||
        fetch(req)
          .then((res) => {
            if (res.ok && url.pathname.startsWith("/static/")) {
              const copy = res.clone();
              caches.open(CACHE).then((c) => c.put(req, copy));
            }
            return res;
          })
          .catch(() => caches.match("/offline", { ignoreSearch: true }))
    )
  );
});

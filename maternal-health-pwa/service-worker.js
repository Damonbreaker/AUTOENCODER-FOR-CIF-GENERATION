const CACHE_NAME = "maternal-health-pwa-v3";
const ASSETS_TO_CACHE = [
  "./",
  "./index.html",
  "./app.js",
  "./manifest.json",
  "./assets/icon.svg",
  "./assets/icon-16.png",
  "./assets/icon-32.png",
  "./assets/icon-72.png",
  "./assets/icon-96.png",
  "./assets/icon-128.png",
  "./assets/icon-144.png",
  "./assets/icon-152.png",
  "./assets/icon-180.png",
  "./assets/icon-192.png",
  "./assets/icon-384.png",
  "./assets/icon-512.png"
];

// Install event: fault-tolerant pre-caching so one missing asset never breaks installation
self.addEventListener("install", (event) => {
  self.skipWaiting();
  event.waitUntil(
    caches.open(CACHE_NAME).then(async (cache) => {
      await Promise.allSettled(
        ASSETS_TO_CACHE.map((url) =>
          cache.add(url).catch((err) => {
            console.warn("Non-fatal cache skip for:", url, err);
          })
        )
      );
    })
  );
});

// Activate event: clean up old caches and claim clients immediately
self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((cacheNames) =>
        Promise.all(
          cacheNames
            .filter((name) => name !== CACHE_NAME)
            .map((name) => caches.delete(name))
        )
      )
      .then(() => self.clients.claim())
  );
});

// Complete fetch event listener (Required by Chrome for PWA installability)
self.addEventListener("fetch", (event) => {
  if (event.request.method !== "GET") return;

  const requestUrl = new URL(event.request.url);
  if (requestUrl.origin !== self.location.origin) return;

  event.respondWith(
    caches.match(event.request).then((cachedResponse) => {
      const networkFetch = fetch(event.request)
        .then((networkResponse) => {
          if (
            networkResponse &&
            networkResponse.status === 200 &&
            networkResponse.type === "basic"
          ) {
            const responseClone = networkResponse.clone();
            caches.open(CACHE_NAME).then((cache) => {
              cache.put(event.request, responseClone);
            });
          }
          return networkResponse;
        })
        .catch(() => {
          if (event.request.mode === "navigate") {
            return caches.match("./index.html");
          }
          return cachedResponse;
        });

      return cachedResponse || networkFetch;
    })
  );
});

// Push event listener: displays a browser notification when a push event is triggered
self.addEventListener("push", (event) => {
  let payload = {
    title: "Maternal Health App Reminder",
    body: "Time to check today's medicines, water intake, and meal schedule for Maa.",
    icon: "assets/icon-192.png",
    badge: "assets/icon-96.png"
  };

  if (event.data) {
    try {
      const parsed = event.data.json();
      payload = { ...payload, ...parsed };
    } catch (e) {
      payload.body = event.data.text();
    }
  }

  const options = {
    body: payload.body,
    icon: payload.icon || "assets/icon-192.png",
    badge: payload.badge || "assets/icon-96.png",
    vibrate: [200, 100, 200],
    data: {
      url: "./index.html"
    }
  };

  event.waitUntil(
    self.registration.showNotification(payload.title, options)
  );
});

// Notification click handler: focuses or opens the PWA
self.addEventListener("notificationclick", (event) => {
  event.notification.close();
  event.waitUntil(
    clients
      .matchAll({ type: "window", includeUncontrolled: true })
      .then((windowClients) => {
        for (const client of windowClients) {
          if ("focus" in client) {
            return client.focus();
          }
        }
        if (clients.openWindow) {
          return clients.openWindow(
            event.notification.data?.url || "./index.html"
          );
        }
      })
  );
});

const CACHE_NAME = "maternal-health-pwa-v5";
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

function xmlEscape(text) {
  return String(text).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function profileResponse() {
  const page = new URL("./", self.registration.scope).href;
  return fetch(new URL("./assets/icon-180.png", self.registration.scope))
    .then((iconRes) => (iconRes.ok ? iconRes.arrayBuffer() : Promise.reject()))
    .then((buffer) => {
      const bytes = new Uint8Array(buffer);
      let binary = "";
      for (let i = 0; i < bytes.length; i++) binary += String.fromCharCode(bytes[i]);
      return btoa(binary);
    })
    .catch(() => "")
    .then((icon) => {
      const iconXml = icon ? "<key>Icon</key>\n<data>" + icon + "</data>\n" : "";
      const xml = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">',
        '<plist version="1.0"><dict>',
        "<key>PayloadContent</key><array><dict>",
        "<key>FullScreen</key><true/>",
        "<key>IsRemovable</key><true/>",
        "<key>Label</key><string>Maa Care</string>",
        "<key>PayloadIdentifier</key><string>com.maacare.webclip</string>",
        "<key>PayloadType</key><string>com.apple.webClip.managed</string>",
        "<key>PayloadUUID</key><string>4E2B8A11-6C70-4D55-9A18-0F3C7B2E91AA</string>",
        "<key>PayloadVersion</key><integer>1</integer>",
        "<key>Precomposed</key><true/>",
        "<key>URL</key><string>" + xmlEscape(page) + "</string>",
        iconXml.trim(),
        "</dict></array>",
        "<key>PayloadDisplayName</key><string>Maa Care</string>",
        "<key>PayloadIdentifier</key><string>com.maacare.profile</string>",
        "<key>PayloadType</key><string>Configuration</string>",
        "<key>PayloadUUID</key><string>91D4C0E7-2A58-4B63-8F10-6C5A9E3D47B2</string>",
        "<key>PayloadVersion</key><integer>1</integer>",
        "</dict></plist>"
      ].filter(Boolean).join("\n");
      return new Response(xml, {
        status: 200,
        headers: {
          "Content-Type": "application/x-apple-aspen-config",
          "Content-Disposition": 'attachment; filename="MaaCare.mobileconfig"',
          "Cache-Control": "no-store"
        }
      });
    });
}

// Complete fetch event listener (Required by Chrome for PWA installability)
self.addEventListener("fetch", (event) => {
  if (event.request.method !== "GET") return;

  const requestUrl = new URL(event.request.url);
  if (requestUrl.origin !== self.location.origin) return;
  if (requestUrl.pathname.endsWith("/maa-care.mobileconfig")) {
    event.respondWith(profileResponse());
    return;
  }

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

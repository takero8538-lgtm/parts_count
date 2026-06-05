const CACHE_NAME = 'pwa-cache-v2'; // バージョンを更新
const urlsToCache = [
  '/',
  '/index.html',
  '/css/style.css',
  '/js/app.js',
  '/js/tf.min.js',
  // 必要なファイルがあれば追加してください
];

// インストールイベント：キャッシュ登録および即時アクティブ化
self.addEventListener('install', event => {
  self.skipWaiting(); // 新しいSWを直ちに有効化
  event.waitUntil(
    caches.open(CACHE_NAME).then(cache => cache.addAll(urlsToCache))
  );
});

// アクティベートイベント：古いキャッシュを削除しクライアントを即制御
self.addEventListener('activate', event => {
  event.waitUntil(
    caches.keys().then(keyList =>
      Promise.all(
        keyList.map(key => {
          if (key !== CACHE_NAME) {
            return caches.delete(key);
          }
        })
      )
    ).then(() => self.clients.claim())
  );
});

// フェッチイベント：キャッシュ優先でレスポンス、無ければネットワークフェッチ
self.addEventListener('fetch', event => {
  event.respondWith(
    caches.match(event.request).then(response => {
      return response || fetch(event.request);
    })
  );
});

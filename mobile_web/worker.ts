/// <reference lib="webworker" />

export default {
  async fetch(request: Request): Promise<Response> {
    const url = new URL(request.url);
    if (request.method !== 'GET' || url.pathname !== '/catalog.json') {
      return new Response('Not found', { status: 404 });
    }
    try {
      const upstream = await fetch('https://pinpanion.com/pins.json', {
        headers: { Accept: 'application/json', 'User-Agent': 'Pinident/0.2 (+public catalog lookup)' },
        signal: AbortSignal.timeout(15000)
      });
      if (!upstream.ok) throw new Error('Catalog unavailable');
      const declaredSize = Number(upstream.headers.get('content-length'));
      if (declaredSize > 5_000_000) throw new Error('Catalog too large');
      if (!upstream.body) throw new Error('Catalog body missing');
      const reader = upstream.body.getReader();
      const chunks: Uint8Array[] = [];
      let size = 0;
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        size += value.byteLength;
        if (size > 5_000_000) {
          await reader.cancel();
          throw new Error('Catalog too large');
        }
        chunks.push(value);
      }
      const body = new Uint8Array(size);
      let offset = 0;
      for (const chunk of chunks) { body.set(chunk, offset); offset += chunk.byteLength; }
      const data: unknown = JSON.parse(new TextDecoder().decode(body));
      if (typeof data !== 'object' || data === null || !('pins' in data) ||
          !Array.isArray(data.pins)) throw new Error('Invalid catalog');
      return new Response(body, { headers: {
        'Content-Type': 'application/json; charset=utf-8',
        'Cache-Control': 'public, max-age=3600',
        'X-Content-Type-Options': 'nosniff'
      } });
    } catch {
      return Response.json({ error: 'Catalog temporarily unavailable' }, { status: 503 });
    }
  }
};

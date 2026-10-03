import assert from 'node:assert/strict';
import test from 'node:test';
import worker from './worker.ts';

test('Cloudflare catalog route forwards the exact Pinpanion JSON feed', async () => {
  const originalFetch = globalThis.fetch;
  const upstream: string[] = [];
  globalThis.fetch = async (input: RequestInfo | URL): Promise<Response> => {
    upstream.push(String(input));
    return new Response(JSON.stringify({ pins: [{ id: 7, name: 'Test Pin' }], categories: [] }),
      { status: 200, headers: { 'Content-Type': 'application/json' } });
  };
  try {
    const response = await worker.fetch(new Request('https://pinident.example/catalog.json'));
    assert.equal(response.status, 200);
    assert.deepEqual(upstream, ['https://pinpanion.com/pins.json']);
    assert.equal((await response.json()).pins[0].id, 7);
  } finally { globalThis.fetch = originalFetch; }
});

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

test('missing model returns 404 when static assets fall through to SPA HTML', async () => {
  const env = {
    DB: { prepare: () => ({ first: async () => null }) },
    ASSETS: { fetch: async () => new Response('<html>app</html>', {
      headers: { 'Content-Type': 'text/html' }
    }) }
  } as unknown as Parameters<typeof worker.fetch>[1];
  const response = await worker.fetch(new Request('https://pinident.example/model.json'), env);
  assert.equal(response.status, 404);
});

test('shared reviews are idempotent, ordered, and desktop-only to read', async () => {
  const events: Array<{ revision: number; review_id: string; submitted_at: string; payload: string }> = [];
  const models: Array<{ version: number; published_at: string; payload: string }> = [];
  const db = {
    prepare(sql: string) {
      let values: unknown[] = [];
      return {
        bind(...args: unknown[]) { values = args; return this; },
        async first() {
          if (sql.includes('FROM training_events'))
            return [...events].reverse().find(row => row.review_id === values[0]) ?? null;
          if (sql.includes('FROM model_versions'))
            return models.at(-1) ?? null;
          return null;
        },
        async all() {
          return { results: events.filter(row => row.revision > Number(values[0])).slice(0, 25) };
        },
        async run() {
          if (sql.includes('INSERT INTO training_events')) {
            const revision = events.length + 1;
            events.push({ revision, review_id: String(values[0]),
              submitted_at: String(values[1]), payload: String(values[2]) });
            return { meta: { changes: 1, last_row_id: revision } };
          }
          if (Number(values[2]) !== models.length)
            return { meta: { changes: 0, last_row_id: 0 } };
          const version = models.length + 1;
          models.push({ version, published_at: String(values[0]), payload: String(values[1]) });
          return { meta: { changes: 1, last_row_id: version } };
        }
      };
    }
  };
  const env = { DB: db, SYNC_TOKEN: 'desktop-secret' } as unknown as Parameters<typeof worker.fetch>[1];
  const origin = 'https://pinident.example';
  const review = { kind: 'training', id: 'review-123', created_at: new Date().toISOString(),
    source: 'photo.jpg', rights_status: 'unverified', time_seconds: null,
    coords: [0, 0, 10, 10], image_data_url: 'data:image/jpeg;base64,AAAA',
    label: 'pin_unknown', pin_id: null, name: null };
  const post = () => worker.fetch(new Request(origin + '/api/training-reviews', {
    method: 'POST', headers: { Origin: origin, 'Content-Type': 'application/json' },
    body: JSON.stringify(review)
  }), env);
  assert.equal((await post()).status, 201);
  assert.equal((await post()).status, 200);
  assert.equal(events.length, 1);
  const annotation = { kind: 'annotation', id: 'frame-12345', created_at: new Date().toISOString(),
    source: 'video.mp4', rights_status: 'unverified', time_seconds: 1.5,
    width: 64, height: 48, image_data_url: 'data:image/jpeg;base64,AAAA',
    boxes: [{ coords: [5, 4, 24, 24], track_id: 'pin-track',
      review_state: 'confirmed', pin_id: 'pinpanion:7', name: 'Pin' }] };
  const postAnnotation = () => worker.fetch(new Request(origin + '/api/training-reviews', {
    method: 'POST', headers: { Origin: origin, 'Content-Type': 'application/json' },
    body: JSON.stringify(annotation)
  }), env);
  assert.equal((await postAnnotation()).status, 201);
  assert.equal((await postAnnotation()).status, 200);
  assert.equal(events.length, 2);
  const unauthorized = await worker.fetch(new Request(origin + '/api/training-reviews'), env);
  assert.equal(unauthorized.status, 401);
  const fetched = await worker.fetch(new Request(origin + '/api/training-reviews?after=0', {
    headers: { Authorization: 'Bearer desktop-secret' }
  }), env);
  assert.equal(fetched.status, 200);
  const page = await fetched.json();
  assert.equal(page.reviews[0].review.id, review.id);
  assert.equal(page.reviews[1].review.image_data_url, annotation.image_data_url);

  const model = { format: 'pinident-linear-svm', version: 1, feature_version: 2,
    training_digest: 'a'.repeat(64), sample_counts: { positive: 1, negative: 1 },
    weights: Array(258).fill(0), bias: 1 };
  const put = (base_version: number, token: string) => worker.fetch(
    new Request(origin + '/api/model', { method: 'PUT',
      headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' },
      body: JSON.stringify({ base_version, model }) }), env);
  assert.equal((await put(0, 'wrong')).status, 401);
  assert.equal((await put(0, 'desktop-secret')).status, 201);
  assert.equal((await put(0, 'desktop-secret')).status, 409);
  const published = await worker.fetch(new Request(origin + '/model.json'), env);
  assert.equal(published.headers.get('X-Model-Version'), '1');
  assert.equal((await published.json()).training_digest, 'a'.repeat(64));
});

test('difficulty marks are shared for ranking but stay out of training events', async () => {
  const marks: Array<{ mark_id: string; signature: string }> = [];
  const db = { prepare(sql: string) {
    let values: unknown[] = [];
    return {
      bind(...args: unknown[]) { values = args; return this; },
      async all() {
        assert.match(sql, /FROM difficulty_marks/);
        return { results: [...marks].reverse() };
      },
      async run() {
        assert.match(sql, /INSERT OR IGNORE INTO difficulty_marks/);
        if (marks.some(mark => mark.mark_id === values[0]))
          return { meta: { changes: 0, last_row_id: 0 } };
        marks.push({ mark_id: String(values[0]), signature: String(values[2]) });
        return { meta: { changes: 1, last_row_id: marks.length } };
      }
    };
  } };
  const env = { DB: db } as unknown as Parameters<typeof worker.fetch>[1];
  const origin = 'https://pinident.example';
  const signature = Array.from({ length: 36 }, (_, index) => index % 2 ? 0.2 : -0.2);
  const post = (body: unknown, requestOrigin = origin) => worker.fetch(
    new Request(origin + '/api/difficulty-marks', { method: 'POST',
      headers: { Origin: requestOrigin, 'Content-Type': 'application/json' },
      body: JSON.stringify(body) }), env);
  const mark = { id: 'mark-12345', signature };
  assert.equal((await post(mark)).status, 201);
  assert.equal((await post(mark)).status, 200);
  assert.equal(marks.length, 1);
  assert.equal((await post({ ...mark, signature: [1] })).status, 400);
  assert.equal((await post(mark, 'https://other.example')).status, 403);
  const response = await worker.fetch(new Request(origin + '/api/difficulty-marks'), env);
  assert.equal(response.status, 200);
  assert.deepEqual((await response.json()).marks, [mark]);
});

/// <reference lib="webworker" />

interface Statement {
  bind(...values: (string | number)[]): Statement;
  first<T>(): Promise<T | null>;
  all<T>(): Promise<{ results: T[] }>;
  run(): Promise<{ meta: { changes: number; last_row_id: number } }>;
}
interface Database { prepare(sql: string): Statement }
interface Env { DB?: Database; SYNC_TOKEN?: string; ASSETS?: { fetch(request: Request): Promise<Response> } }
type ReviewRow = { revision: number; submitted_at: string; payload: string };
type ModelRow = { version: number; published_at: string; payload: string };
type DifficultyRow = { mark_id: string; signature: string };
const HEADERS = { 'Content-Type': 'application/json; charset=utf-8',
  'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff' };

function authorized(request: Request, env: Env): boolean {
  return !!env.SYNC_TOKEN && request.headers.get('Authorization') === `Bearer ${env.SYNC_TOKEN}`;
}
async function limitedJson(request: Request, limit: number): Promise<unknown> {
  if (!request.headers.get('Content-Type')?.startsWith('application/json') ||
      Number(request.headers.get('Content-Length')) > limit)
    throw new Error('Invalid or oversized JSON.');
  const body = await request.text();
  if (new TextEncoder().encode(body).byteLength > limit) throw new Error('Invalid or oversized JSON.');
  return JSON.parse(body) as unknown;
}
function record(value: unknown): Record<string, unknown> | null {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
    ? value as Record<string, unknown> : null;
}
function validReview(value: unknown): boolean {
  const row = record(value);
  const coords = row?.coords;
  return !!row && row.kind === 'training' && typeof row.id === 'string' &&
    /^[a-zA-Z0-9:_-]{8,100}$/.test(row.id) &&
    typeof row.created_at === 'string' && !Number.isNaN(Date.parse(row.created_at)) &&
    typeof row.source === 'string' && row.source.length <= 250 &&
    row.rights_status === 'unverified' &&
    (row.time_seconds === null || typeof row.time_seconds === 'number') &&
    Array.isArray(coords) && coords.length === 4 &&
    coords.every(item => Number.isInteger(item) && item >= 0) &&
    coords[2] > 0 && coords[3] > 0 &&
    typeof row.image_data_url === 'string' &&
    /^data:image\/(jpeg|png);base64,[A-Za-z0-9+/=]+$/.test(row.image_data_url) &&
    row.image_data_url.length < 500_000 &&
    ['not_pin', 'pin_unknown', 'pin_identified'].includes(String(row.label)) &&
    (row.pin_id === null || typeof row.pin_id === 'string') &&
    (row.name === null || typeof row.name === 'string') &&
    (row.label !== 'pin_identified' ||
      (typeof row.pin_id === 'string' && row.pin_id.length > 0));
}
function validAnnotation(value: unknown): boolean {
  const row = record(value);
  if (!row || row.kind !== 'annotation' || typeof row.id !== 'string' ||
      !/^[a-zA-Z0-9:_-]{8,100}$/.test(row.id) ||
      typeof row.created_at !== 'string' || Number.isNaN(Date.parse(row.created_at)) ||
      typeof row.source !== 'string' || row.source.length > 250 ||
      row.rights_status !== 'unverified' ||
      !(row.time_seconds === null || typeof row.time_seconds === 'number' &&
        Number.isFinite(row.time_seconds)) ||
      !Number.isInteger(row.width) || Number(row.width) < 1 || Number(row.width) > 1024 ||
      !Number.isInteger(row.height) || Number(row.height) < 1 || Number(row.height) > 1024 ||
      typeof row.image_data_url !== 'string' ||
      !/^data:image\/(jpeg|png);base64,[A-Za-z0-9+/=]+$/.test(row.image_data_url) ||
      row.image_data_url.length >= 900_000 ||
      !Array.isArray(row.boxes) || row.boxes.length > 100) return false;
  return row.boxes.every(value => {
    const box = record(value);
    const coords = box?.coords;
    return !!box && typeof box.track_id === 'string' &&
      /^[a-zA-Z0-9:_-]{1,100}$/.test(box.track_id) &&
      ['unconfirmed', 'confirmed', 'inherited', 'rejected'].includes(String(box.review_state)) &&
      (box.pin_id === null || typeof box.pin_id === 'string' && box.pin_id.length <= 200) &&
      (box.name === null || typeof box.name === 'string' && box.name.length <= 200) &&
      Array.isArray(coords) && coords.length === 4 &&
      coords.every(item => Number.isInteger(item) && item >= 0) &&
      coords[2] > 0 && coords[3] > 0 &&
      coords[0] + coords[2] <= Number(row.width) &&
      coords[1] + coords[3] <= Number(row.height);
  });
}
function validDifficultyMark(value: unknown): value is { id: string; signature: number[] } {
  const row = record(value);
  return !!row && typeof row.id === 'string' &&
    /^[a-zA-Z0-9:_-]{8,100}$/.test(row.id) &&
    Array.isArray(row.signature) && row.signature.length === 36 &&
    row.signature.every(item => typeof item === 'number' && Number.isFinite(item) &&
      item >= -1 && item <= 1);
}
function validModel(value: unknown): boolean {
  const row = record(value), counts = record(row?.sample_counts);
  return !!row && row.format === 'pinident-linear-svm' && row.version === 1 &&
    row.feature_version === 2 && typeof row.training_digest === 'string' &&
    /^[a-f0-9]{64}$/.test(row.training_digest) &&
    Number.isInteger(counts?.positive) && Number(counts?.positive) > 0 &&
    Number.isInteger(counts?.negative) && Number(counts?.negative) > 0 &&
    Array.isArray(row.weights) && row.weights.length === 258 &&
    row.weights.every(item => typeof item === 'number' && Number.isFinite(item)) &&
    typeof row.bias === 'number' && Number.isFinite(row.bias);
}
async function catalog(): Promise<Response> {
  try {
    const upstream = await fetch('https://pinpanion.com/pins.json', {
      headers: { Accept: 'application/json', 'User-Agent': 'Pinident/0.2 (+public catalog lookup)' },
      signal: AbortSignal.timeout(15000)
    });
    if (!upstream.ok) throw new Error('Catalog unavailable');
    const declaredSize = Number(upstream.headers.get('content-length'));
    if (declaredSize > 5_000_000 || !upstream.body) throw new Error('Catalog too large');
    const reader = upstream.body.getReader();
    const chunks: Uint8Array[] = [];
    let size = 0;
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      size += value.byteLength;
      if (size > 5_000_000) { await reader.cancel(); throw new Error('Catalog too large'); }
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
      'Cache-Control': 'public, max-age=3600', 'X-Content-Type-Options': 'nosniff'
    } });
  } catch {
    return Response.json({ error: 'Catalog temporarily unavailable' }, { status: 503 });
  }
}

async function staticModel(request: Request, env: Env): Promise<Response> {
  const response = await env.ASSETS?.fetch(request);
  return response?.ok && response.headers.get('Content-Type')?.startsWith('application/json')
    ? response : new Response('Not found', { status: 404 });
}

export default {
  async fetch(request: Request, env: Env = {}): Promise<Response> {
    const url = new URL(request.url);
    if (request.method === 'GET' && url.pathname === '/catalog.json') return catalog();
    if (!['/api/training-reviews', '/api/difficulty-marks',
      '/api/model', '/model.json'].includes(url.pathname))
      return new Response('Not found', { status: 404 });
    if (!env.DB) {
      if (url.pathname === '/model.json') return staticModel(request, env);
      return Response.json({ error: 'Shared training storage is not configured.' },
        { status: 503, headers: HEADERS });
    }
    try {
      if (url.pathname === '/api/difficulty-marks' && request.method === 'POST') {
        if (request.headers.get('Origin') !== url.origin)
          return new Response('Forbidden origin', { status: 403 });
        const value = await limitedJson(request, 2_000);
        if (!validDifficultyMark(value))
          return Response.json({ error: 'Invalid difficulty mark.' }, { status: 400 });
        const result = await env.DB.prepare(
          'INSERT OR IGNORE INTO difficulty_marks (mark_id, marked_at, signature) VALUES (?, ?, ?)'
        ).bind(value.id, new Date().toISOString(), JSON.stringify(value.signature)).run();
        return Response.json({ saved: true },
          { status: result.meta.changes ? 201 : 200, headers: HEADERS });
      }
      if (url.pathname === '/api/difficulty-marks' && request.method === 'GET') {
        const result = await env.DB.prepare(
          'SELECT mark_id, signature FROM difficulty_marks ORDER BY id DESC LIMIT 500'
        ).all<DifficultyRow>();
        return Response.json({ marks: result.results.map(row => ({
          id: row.mark_id, signature: JSON.parse(row.signature) as number[]
        })) }, { headers: HEADERS });
      }
      if (url.pathname === '/api/training-reviews' && request.method === 'POST') {
        if (request.headers.get('Origin') !== url.origin)
          return new Response('Forbidden origin', { status: 403 });
        const value = await limitedJson(request, 1_000_000);
        if (!validReview(value) && !validAnnotation(value))
          return Response.json({ error: 'Invalid review or annotation.' }, { status: 400 });
        const review = value as { id: string };
        const payload = JSON.stringify(value);
        const previous = await env.DB.prepare(
          'SELECT revision, payload FROM training_events WHERE review_id = ? ORDER BY revision DESC LIMIT 1'
        ).bind(review.id).first<{ revision: number; payload: string }>();
        if (previous?.payload === payload)
          return Response.json({ revision: previous.revision, saved: true }, { headers: HEADERS });
        const result = await env.DB.prepare(
          'INSERT INTO training_events (review_id, submitted_at, payload) VALUES (?, ?, ?)'
        ).bind(review.id, new Date().toISOString(), payload).run();
        return Response.json({ revision: result.meta.last_row_id, saved: true },
          { status: 201, headers: HEADERS });
      }
      if (url.pathname === '/api/training-reviews' && request.method === 'GET') {
        if (!authorized(request, env)) return new Response('Unauthorized', { status: 401 });
        const after = Number(url.searchParams.get('after') ?? '0');
        if (!Number.isSafeInteger(after) || after < 0)
          return Response.json({ error: 'Invalid cursor.' }, { status: 400 });
        const result = await env.DB.prepare(
          'SELECT revision, submitted_at, payload FROM training_events WHERE revision > ? ORDER BY revision LIMIT 5'
        ).bind(after).all<ReviewRow>();
        const reviews = result.results.map(row => ({
          revision: row.revision, submitted_at: row.submitted_at,
          review: JSON.parse(row.payload) as unknown
        }));
        return Response.json({ reviews, next: reviews.at(-1)?.revision ?? after }, { headers: HEADERS });
      }
      if (url.pathname === '/model.json' && request.method === 'GET') {
        const latest = await env.DB.prepare(
          'SELECT version, published_at, payload FROM model_versions ORDER BY version DESC LIMIT 1'
        ).first<ModelRow>();
        if (!latest) return staticModel(request, env);
        return new Response(latest.payload, { headers: { ...HEADERS,
          ETag: `"${latest.version}"`, 'X-Model-Version': String(latest.version) } });
      }
      if (url.pathname === '/api/model' && request.method === 'GET') {
        if (!authorized(request, env)) return new Response('Unauthorized', { status: 401 });
        const latest = await env.DB.prepare(
          'SELECT version, published_at FROM model_versions ORDER BY version DESC LIMIT 1'
        ).first<{ version: number; published_at: string }>();
        return Response.json({ version: latest?.version ?? 0,
          published_at: latest?.published_at ?? null }, { headers: HEADERS });
      }
      if (url.pathname === '/api/model' && request.method === 'PUT') {
        if (!authorized(request, env)) return new Response('Unauthorized', { status: 401 });
        const value = record(await limitedJson(request, 100_000));
        if (!value || !validModel(value.model) || !Number.isSafeInteger(value.base_version) ||
            Number(value.base_version) < 0)
          return Response.json({ error: 'Invalid model upload.' }, { status: 400 });
        const result = await env.DB.prepare(`INSERT INTO model_versions (published_at, payload)
          SELECT ?, ? WHERE (SELECT COALESCE(MAX(version), 0) FROM model_versions) = ?`)
          .bind(new Date().toISOString(), JSON.stringify(value.model), Number(value.base_version)).run();
        if (result.meta.changes !== 1)
          return Response.json({ error: 'A newer model was published. Refresh before uploading.' },
            { status: 409 });
        return Response.json({ version: result.meta.last_row_id }, { status: 201, headers: HEADERS });
      }
      return new Response('Method not allowed', { status: 405 });
    } catch (error) {
      if (error instanceof SyntaxError ||
          error instanceof Error && /JSON body|Invalid or oversized/.test(error.message))
        return Response.json({ error: 'Invalid or oversized JSON.' }, { status: 400 });
      return Response.json({ error: 'Shared training storage is temporarily unavailable.' },
        { status: 503 });
    }
  }
};

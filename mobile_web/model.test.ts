import assert from 'node:assert/strict';
import test from 'node:test';
import { classifyProposal, loadMobileModel, parseMobileModel, proposalFeature } from './dist/model.js';
import { detectCandidates, scanFrame, type PixelFrame } from './dist/scanner.js';
import { referenceFeature, type IdentityIndex } from './dist/identity.js';

function model(bias: number) {
  return parseMobileModel({ format: 'pinident-linear-svm', version: 1, feature_version: 2,
    training_digest: 'a'.repeat(64), sample_counts: { positive: 4, negative: 4 },
    weights: Array(258).fill(0), bias });
}

test('browser RGB feature matches the desktop reference', () => {
  const width = 32, height = 32, data = new Uint8ClampedArray(width * height * 4);
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
    const i = (y * width + x) * 4;
    data[i] = (x * 7 + y * 3) % 256;
    data[i + 1] = (x * 2 + y * 11) % 256;
    data[i + 2] = (x * 13 + y * 5) % 256;
    data[i + 3] = 255;
  }
  const feature = proposalFeature({ width, height, data }, [0, 0, width, height]);
  assert.ok(feature);
  for (let i = 0; i < 4; i++)
    assert.ok(Math.abs(feature[i]! - [-0.15275587, -0.14011656, -0.12747726, -0.11483795][i]!) < 1e-6);
  assert.ok(Math.abs(feature[256]! - 0.0125) < 1e-6);
  assert.ok(Math.abs(feature[257]! - 0.01044118) < 1e-6);
});

test('model rejects new proposals but keeps an already tracked area', () => {
  const width = 160, height = 120, data = new Uint8ClampedArray(width * height * 4);
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
    const pin = x >= 24 && x < 52 && y >= 28 && y < 56;
    const i = (y * width + x) * 4;
    data[i] = pin ? 30 : 245; data[i + 1] = pin ? 60 : 245;
    data[i + 2] = pin ? 100 : 245; data[i + 3] = 255;
  }
  const image: PixelFrame = { width, height, data };
  const feature = referenceFeature(image, detectCandidates(image)[0]!);
  assert.ok(feature);
  const index: IdentityIndex = { format: 'pinident-catalog-reference-index', version: 1,
    catalog_state: [1, 1, 1], catalog_rights_unverified: 0,
    references: [{ pin_id: 'pinpanion:7', name: 'Test Pin', source: 'fixture',
      rights: 'fixture', feature }] };
  const baseline = scanFrame(image, null, null, undefined, index);
  assert.ok(baseline.boxes.length > 0);
  assert.equal(classifyProposal(model(-1), image, baseline.boxes[0]!.coords), 'negative');
  assert.equal(scanFrame(image, null, model(-1), undefined, index).boxes.length, 0);
  assert.ok(scanFrame(image, baseline, model(-1), undefined, index).boxes.some(box =>
    box.track_id === baseline.boxes[0]!.track_id));
});

test('unsupported model data is rejected', () => {
  assert.throws(() => parseMobileModel({ ...model(1), feature_version: 1 }), /Unsupported/);
});

test('model loader accepts JSON and treats SPA HTML fallback as no model', async () => {
  const originalFetch = globalThis.fetch;
  try {
    globalThis.fetch = async () => new Response('<html></html>', {
      headers: { 'Content-Type': 'text/html' } });
    assert.equal(await loadMobileModel(), null);
    globalThis.fetch = async () => Response.json(model(1));
    assert.equal((await loadMobileModel())?.feature_version, 2);
  } finally { globalThis.fetch = originalFetch; }
});

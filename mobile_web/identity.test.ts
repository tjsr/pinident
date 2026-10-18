import assert from 'node:assert/strict';
import test from 'node:test';
import { combineIdentityIndex, learnNegative, learnPositive, loadIdentityIndex,
  loadLearnedExamples, matchProposal, matchReference, parseIdentityIndex, referenceFeature,
  saveLearnedExamples,
  type IdentityIndex } from './dist/identity.js';
import type { PixelFrame } from './dist/scanner.js';

function patternedFrame(): PixelFrame {
  const width = 32, height = 32;
  const data = new Uint8ClampedArray(width * height * 4);
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
    const index = (y * width + x) * 4;
    data[index] = (x * 13 + y * 7) % 256;
    data[index + 1] = (x * 3 + y * 17) % 256;
    data[index + 2] = (x * 19 + y * 5) % 256;
    data[index + 3] = 255;
  }
  return { width, height, data };
}

test('a detailed catalog crop yields a provisional identity and match strength', () => {
  const feature = referenceFeature(patternedFrame(), [0, 0, 32, 32]);
  assert.ok(feature);
  // The same pattern encoded as PNG and read by build_identity_index.py.
  for (let i = 0; i < 4; i++)
    assert.ok(Math.abs(feature[i]! - [-0.2068826, -0.1284002, -0.0499178, -0.0447151][i]!) < 1e-6);
  for (let i = 0; i < 3; i++)
    assert.ok(Math.abs(feature[64 + i]! - [0.5019608, 0.4705882, 0.5019608][i]!) < 1e-6);
  const index: IdentityIndex = {
    format: 'pinident-catalog-reference-index', version: 1,
    catalog_state: [1, 1, 1], catalog_rights_unverified: 0,
    references: [{ pin_id: 'pinpanion:12', name: 'Pattern Pin',
      source: 'fixture', rights: 'fixture', feature }]
  };
  assert.deepEqual(matchReference(index, feature), {
    pin_id: 'pinpanion:12', name: 'Pattern Pin', strength: 1
  });
  assert.equal(referenceFeature(patternedFrame(), [-1, 0, 32, 32]), null);
  assert.throws(() => parseIdentityIndex({ ...index, references: [
    { ...index.references[0], feature: [0] }
  ] }), /Unsupported/);
});

test('direct corrections train a local positive example and rejected crops veto it', () => {
  const feature = referenceFeature(patternedFrame(), [0, 0, 32, 32]);
  assert.ok(feature);
  const saved = new Map<string, string>();
  const storage = {
    getItem: (key: string) => saved.get(key) ?? null,
    setItem: (key: string, value: string) => { saved.set(key, value); }
  };
  const examples = loadLearnedExamples(storage);
  learnPositive(examples, 'pinpanion:12', 'Corrected Pin', feature);
  saveLearnedExamples(examples, storage);
  const restored = loadLearnedExamples(storage);
  const index = combineIdentityIndex(null, restored);
  assert.ok(index);
  assert.equal(matchReference(index, feature)?.name, 'Corrected Pin');
  learnNegative(restored, feature);
  assert.equal(matchReference(index, feature, restored.negative), null);
  assert.equal(matchProposal(index, patternedFrame(), [0, 0, 32, 32], restored.negative), null);
  assert.equal(combineIdentityIndex(null, { positive: [], negative: restored.negative }), null);
});

test('flat crops cannot acquire a catalog name', () => {
  const frame: PixelFrame = { width: 16, height: 16,
    data: new Uint8ClampedArray(16 * 16 * 4).fill(200) };
  assert.equal(referenceFeature(frame, [0, 0, 16, 16]), null);
});

test('matching tolerates detector padding around a catalog image', () => {
  const image = patternedFrame();
  const padded: PixelFrame = { width: 40, height: 40,
    data: new Uint8ClampedArray(40 * 40 * 4).fill(255) };
  for (let y = 0; y < 32; y++) for (let x = 0; x < 32; x++) {
    const source = (y * 32 + x) * 4;
    const target = ((y + 4) * 40 + x + 4) * 4;
    padded.data.set(image.data.subarray(source, source + 4), target);
  }
  const feature = referenceFeature(image, [0, 0, 32, 32]);
  assert.ok(feature);
  const index: IdentityIndex = { format: 'pinident-catalog-reference-index', version: 1,
    catalog_state: [1, 1, 1], catalog_rights_unverified: 0,
    references: [{ pin_id: 'pinpanion:12', name: 'Pattern Pin',
      source: 'fixture', rights: 'fixture', feature }] };
  const match = matchProposal(index, padded, [0, 0, 40, 40]);
  assert.equal(match?.pin_id, 'pinpanion:12');
  assert.ok((match?.strength ?? 0) >= 0.5);
});

test('an absent reference index does not fall through to the app HTML', async () => {
  const oldFetch = globalThis.fetch;
  try {
    globalThis.fetch = async () => new Response('Not found', { status: 404 });
    assert.equal(await loadIdentityIndex(), null);
    globalThis.fetch = async () => new Response('<html></html>', {
      headers: { 'Content-Type': 'text/html' } });
    assert.equal(await loadIdentityIndex(), null);
  } finally { globalThis.fetch = oldFetch; }
});

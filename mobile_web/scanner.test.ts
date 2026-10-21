import assert from 'node:assert/strict';
import test from 'node:test';
import { areaAtPoint, deleteAreas, detectCandidates, rejectTrack, resolveAreaDrag,
  scanFrame, selectAreasInRectangle, similarSize, strongestMatchSizeReference,
  tagTrack, type Coords, type PixelFrame, type ScanDiagnostics,
  type ScanState } from './dist/scanner.js';
import { referenceFeature, type IdentityIndex } from './dist/identity.js';
import { containFrame, framePoint } from './dist/stage_geometry.js';

function frame(rectangles: Array<[number, number, number, number]>): PixelFrame {
  const width = 160, height = 120;
  const data = new Uint8ClampedArray(width * height * 4);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const pin = rectangles.some(([left, top, w, h]) => x >= left && x < left + w && y >= top && y < top + h);
      const index = (y * width + x) * 4;
      data[index] = pin ? 30 : 245;
      data[index + 1] = pin ? 60 : 245;
      data[index + 2] = pin ? 100 : 245;
      data[index + 3] = 255;
    }
  }
  return { width, height, data };
}

function indexFor(image: PixelFrame): IdentityIndex {
  const coords = detectCandidates(image)[0]!;
  const feature = referenceFeature(image, coords);
  assert.ok(feature);
  return { format: 'pinident-catalog-reference-index', version: 1,
    catalog_state: [1, 1, 1], catalog_rights_unverified: 0,
    references: [{ pin_id: 'pinpanion:7', name: 'Test Pin', source: 'test',
      rights: 'test fixture', feature }] };
}

test('on-device detector proposes an isolated object', () => {
  const boxes = detectCandidates(frame([[24, 28, 28, 28]]));
  assert.ok(boxes.some(([x, y, w, h]) => x <= 24 && y <= 28 && x + w >= 52 && y + h >= 56));
});

test('scan diagnostics retain proposal coordinates and model rejections', () => {
  const image = frame([[24, 28, 28, 28]]);
  const diagnostics: ScanDiagnostics = { proposals: [], modelRejected: [], tracked: 0 };
  const model = { format: 'pinident-linear-svm' as const, version: 1 as const,
    feature_version: 2 as const, training_digest: '0'.repeat(64),
    sample_counts: { positive: 1, negative: 1 }, weights: Array(258).fill(0), bias: -1 };
  const result = scanFrame(image, null, model, diagnostics, indexFor(image));
  assert.ok(diagnostics.proposals.length > 0);
  assert.deepEqual(diagnostics.modelRejected, diagnostics.proposals);
  assert.equal(diagnostics.tracked, 0);
  assert.equal(result.boxes.length, 0);
});

test('close-up pin is proposed as one full area instead of its small details', () => {
  const image = frame([[20, 12, 120, 96]]);
  // Bright marks inside the dark body mimic lettering or highlights on a pin.
  for (const [left, top] of [[42, 34], [88, 34], [65, 70]] as Array<[number, number]>) {
    for (let y = top; y < top + 9; y++) {
      for (let x = left; x < left + 9; x++) {
        const p = (y * image.width + x) * 4;
        image.data[p] = image.data[p + 1] = image.data[p + 2] = 235;
      }
    }
  }
  const boxes = detectCandidates(image);
  const whole = boxes.find(([x, y, w, h]) => x <= 20 && y <= 12 && x + w >= 140 && y + h >= 108);
  assert.ok(whole, `Expected a full-pin proposal, got ${JSON.stringify(boxes)}`);
  assert.ok(boxes.every(box => box === whole ||
    !(box[0] >= whole[0] && box[1] >= whole[1] &&
      box[0] + box[2] <= whole[0] + whole[2] && box[1] + box[3] <= whole[1] + whole[3])),
  `Small interior details should not be separate proposals: ${JSON.stringify(boxes)}`);
  assert.ok(scanFrame(image, null, null, undefined, indexFor(image)).boxes.some(({ coords: [x, y, w, h] }) =>
    x <= 20 && y <= 12 && x + w >= 140 && y + h >= 108));
});

test('confirmed and rejected tracks carry to later frames', () => {
  const image = frame([[24, 28, 28, 28]]);
  const index = indexFor(image);
  const first = scanFrame(image, null, null, undefined, index);
  assert.ok(first.boxes.length > 0);
  const track = first.boxes[0]!.track_id;
  assert.equal(tagTrack(first, track, { pin_id: 'pinpanion:7', name: 'Test Pin' }), true);
  const second = scanFrame(image, first, null, undefined, index);
  const inherited = second.boxes.find(box => box.track_id === track);
  assert.equal(inherited?.review_state, 'inherited');
  assert.equal(inherited?.pin_id, 'pinpanion:7');
  assert.equal(rejectTrack(second, track), true);
  const third = scanFrame(image, second, null, undefined, index);
  assert.equal(third.boxes.find(box => box.track_id === track)?.review_state, 'rejected');
});

test('automatic areas wait for a 50% catalog match and show a provisional name', () => {
  const image = frame([[24, 28, 28, 28]]);
  assert.equal(scanFrame(image).boxes.length, 0);
  const accepted = scanFrame(image, null, null, undefined, indexFor(image));
  assert.equal(accepted.boxes.length, 1);
  assert.equal(accepted.boxes[0]?.suggested_name, 'Test Pin');
  assert.ok((accepted.boxes[0]?.match_strength ?? 0) >= 0.5);
  assert.equal(accepted.boxes[0]?.pin_id, null);
});

test('the strongest later match replaces the size reference', () => {
  const candidates: Array<{ coords: Coords; strength: number }> = [
    { coords: [8, 8, 12, 12], strength: 0.62 },
    { coords: [45, 8, 28, 28], strength: 0.91 },
    { coords: [90, 8, 27, 29], strength: 0.7 },
  ];
  const anchor = strongestMatchSizeReference(candidates);
  assert.deepEqual(anchor, [45, 8, 28, 28]);
  assert.equal(similarSize(candidates[0]!.coords, anchor!, 160, 120), false);
  assert.equal(similarSize(candidates[2]!.coords, anchor!, 160, 120), true);
  assert.equal(similarSize([8, 8, 14, 42], anchor!, 160, 120), true);
  assert.equal(similarSize([8, 8, 43, 28], anchor!, 160, 120), false);
  assert.equal(similarSize([0, 8, 18, 28], anchor!, 160, 120), true);
});

test('marquee selects only areas covered by more than half of their own rectangle', () => {
  const state: ScanState = { width: 100, height: 100, boxes: [
    { coords: [0, 0, 20, 20], track_id: 'more', review_state: 'unconfirmed',
      pin_id: null, name: null, signature: [] },
    { coords: [40, 0, 20, 20], track_id: 'exact-half', review_state: 'confirmed',
      pin_id: null, name: null, signature: [] },
    { coords: [70, 0, 20, 20], track_id: 'less', review_state: 'unconfirmed',
      pin_id: null, name: null, signature: [] },
    { coords: [0, 0, 20, 20], track_id: 'hidden', review_state: 'rejected',
      pin_id: null, name: null, signature: [] },
  ] };
  assert.deepEqual(selectAreasInRectangle(state, [0, 0, 70, 11]), ['more', 'exact-half']);
  assert.deepEqual(selectAreasInRectangle(state, [0, 0, 70, 10]), []);
  assert.deepEqual(selectAreasInRectangle(state, [0, 0, 70, 9]), []);
});

test('default paused drag selects covered areas and draws only when none are covered', () => {
  const state: ScanState = { width: 100, height: 100, boxes: [
    { coords: [10, 10, 20, 20], track_id: 'existing', review_state: 'unconfirmed',
      pin_id: null, name: null, signature: [] },
  ] };
  assert.deepEqual(resolveAreaDrag(state, [5, 5, 30, 30], 'auto'),
    { action: 'select', trackIds: ['existing'] });
  assert.deepEqual(resolveAreaDrag(state, [50, 50, 20, 20], 'auto'), { action: 'draw' });
  assert.deepEqual(resolveAreaDrag(state, [5, 5, 30, 30], 'draw'), { action: 'draw' });
  assert.deepEqual(resolveAreaDrag(state, [50, 50, 20, 20], 'select'),
    { action: 'select', trackIds: [] });
});

test('portrait preview keeps its bottom edge visible and draggable inside a capped stage', () => {
  const placement = containFrame(358, 548, 236, 420);
  assert.ok(placement);
  assert.equal(placement.top, 0);
  assert.ok(placement.left > 0);
  assert.ok(placement.left + placement.width < 358);
  assert.deepEqual(framePoint(placement, [placement.left, 0], 236, 420), [0, 0]);
  const bottom = framePoint(placement, [placement.left + placement.width / 2, 547], 236, 420);
  assert.ok(bottom && bottom[1] > 419 && bottom[1] < 420);
  assert.equal(framePoint(placement, [0, 547], 236, 420), null);
});

test('deleting selected areas removes annotations without rejecting them', () => {
  const state: ScanState = { width: 100, height: 100, boxes: [
    { coords: [0, 0, 20, 20], track_id: 'first', review_state: 'confirmed',
      pin_id: 'pinpanion:1', name: 'Pin', signature: [] },
    { coords: [40, 0, 20, 20], track_id: 'second', review_state: 'unconfirmed',
      pin_id: null, name: null, signature: [] },
  ] };
  assert.equal(deleteAreas(state, new Set(['first', 'missing'])), 1);
  assert.deepEqual(state.boxes.map(box => box.track_id), ['second']);
  assert.equal(deleteAreas(state, new Set(['missing'])), 0);
  assert.equal(areaAtPoint(state, [45, 5]), 'second');
  assert.equal(areaAtPoint(state, [5, 5]), null);
});

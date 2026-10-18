import assert from 'node:assert/strict';
import test from 'node:test';
import { completeScan, scanControls } from './dist/scan_session.js';
import { addManualBox, confirmUnknown, scanFrame, tagTrack, type PixelFrame } from './dist/scanner.js';
import { clampDrawnBox, difficultyPriority, loadSharedDifficultyHints,
  makeDifficultyMark, makeFrameAnnotation, makeTrainingReview,
  postDifficultyMark, postTrainingReview, postFrameAnnotation } from './dist/reviews.js';

function frame(): PixelFrame {
  const width = 80, height = 60;
  const data = new Uint8ClampedArray(width * height * 4);
  data.fill(255);
  return { width, height, data };
}

test('active scan replaces source choices with Pause and OK, then restores review actions', () => {
  const idle = scanControls(false, false, false, false);
  assert.equal(idle.sourceChoices, true);
  assert.equal(idle.freeze, false);
  assert.equal(idle.finish, false);
  const running = scanControls(true, true, false, false);
  assert.equal(running.sourceChoices, false);
  assert.equal(running.running, true);
  assert.equal(running.freeze, true);
  assert.equal(running.finish, true);
  assert.equal(running.stop, false);
  assert.equal(running.stillActions, false);
  const paused = scanControls(true, true, true, false);
  assert.equal(paused.sourceChoices, false);
  assert.equal(paused.running, false);
  assert.equal(paused.resume, true);
  assert.equal(paused.finish, true);
  assert.equal(paused.stop, true);
  assert.equal(paused.stillActions, true);
  const photo = scanControls(true, true, true, true);
  assert.equal(photo.resume, false);
  assert.equal(photo.finish, true);
  assert.equal(scanControls(true, false, true, false).stillActions, false);
});

test('drawn boxes are clipped and reject tiny drags', () => {
  assert.deepEqual(clampDrawnBox([50, 40], [-10, 10], 80, 60), [0, 10, 50, 30]);
  assert.equal(clampDrawnBox([3, 3], [5, 5], 80, 60), null);
});

test('manual annotation and unknown review stay distinct from identity confirmation', () => {
  const image = frame();
  const scan = scanFrame(image);
  const box = addManualBox(scan, image, [5, 6, 20, 21]);
  assert.ok(box);
  assert.equal(addManualBox(scan, image, [78, 3, 10, 10]), null);
  assert.equal(confirmUnknown(scan, box.track_id), true);
  const review = makeTrainingReview('photo.jpg', null, box, 'data:image/jpeg;base64,abc', 'pin_unknown');
  assert.equal(review.label, 'pin_unknown');
  assert.equal(review.pin_id, null);
  assert.equal(review.suggested_pin_id, null);
  const annotation = makeFrameAnnotation('photo.jpg', null, scan.width, scan.height,
    'data:image/jpeg;base64,frame', scan.boxes);
  assert.ok(annotation.boxes.some(saved => saved.track_id === box.track_id));
  assert.equal(annotation.kind, 'annotation');
});

test('training review preserves original suggestion when identity is corrected', () => {
  const image = frame();
  const scan = scanFrame(image);
  const box = addManualBox(scan, image, [5, 6, 20, 21]);
  assert.ok(box);
  tagTrack(scan, box.track_id, { pin_id: 'pinpanion:1', name: 'First' });
  box.suggested_pin_id = 'pinpanion:1';
  box.suggested_name = 'First';
  const review = makeTrainingReview('video.mp4', 1.25, box, 'data:image/jpeg;base64,abc',
    'pin_identified', { pin_id: 'pinpanion:2', name: 'Second' });
  assert.equal(review.suggested_pin_id, 'pinpanion:1');
  assert.equal(review.pin_id, 'pinpanion:2');
  assert.equal(review.time_seconds, 1.25);
});

test('saved training review posts its crop and label without local sync metadata', async () => {
  const image = frame(), scan = scanFrame(image);
  const box = addManualBox(scan, image, [5, 6, 20, 21]);
  assert.ok(box);
  const review = { ...makeTrainingReview('photo.jpg', null, box,
    'data:image/jpeg;base64,AAAA', 'pin_unknown'), remote_revision: 2 };
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async (input, init) => {
    assert.equal(input, '/api/training-reviews');
    assert.equal(init?.method, 'POST');
    const posted = JSON.parse(String(init?.body));
    assert.equal(posted.label, 'pin_unknown');
    assert.equal(posted.image_data_url, review.image_data_url);
    assert.equal(posted.remote_revision, undefined);
    return Response.json({ revision: 3 });
  };
  try { assert.equal(await postTrainingReview(review), 3); }
  finally { globalThis.fetch = originalFetch; }
});

test('Unknown stores a difficulty mark, not a pin or non-pin training label', () => {
  const image = frame(), scan = scanFrame(image);
  const box = addManualBox(scan, image, [5, 6, 20, 21]);
  assert.ok(box);
  const mark = makeDifficultyMark('photo.jpg', null, box, 'data:image/jpeg;base64,AAAA');
  assert.equal(mark.kind, 'difficulty');
  assert.equal('label' in mark, false);
  assert.equal(mark.image_data_url, 'data:image/jpeg;base64,AAAA');
  assert.deepEqual(mark.signature, box.signature);
  assert.equal(mark.kind, 'difficulty');
});

test('saved frame annotation posts bounded image and boxes without local sync metadata', async () => {
  const image = frame(), scan = scanFrame(image);
  const box = addManualBox(scan, image, [5, 6, 20, 21]);
  assert.ok(box);
  const annotation = { ...makeFrameAnnotation('video.mp4', 1.5, scan.width, scan.height,
    'data:image/jpeg;base64,AAAA', scan.boxes), remote_revision: 4 };
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async (input, init) => {
    assert.equal(input, '/api/training-reviews');
    assert.equal(init?.method, 'POST');
    const posted = JSON.parse(String(init?.body));
    assert.equal(posted.kind, 'annotation');
    assert.equal(posted.image_data_url, annotation.image_data_url);
    assert.equal(posted.boxes[0].review_state, 'confirmed');
    assert.equal(posted.remote_revision, undefined);
    return Response.json({ revision: 5 });
  };
  try { assert.equal(await postFrameAnnotation(annotation), 5); }
  finally { globalThis.fetch = originalFetch; }
  const changed = makeFrameAnnotation('video.mp4', 1.5, scan.width, scan.height,
    'data:image/jpeg;base64,BBBB', [], annotation);
  assert.equal(changed.id, annotation.id);
  assert.equal(changed.created_at, annotation.created_at);
  assert.equal(changed.remote_revision, undefined);
  assert.deepEqual(changed.boxes, []);
});

test('OK commits tagged boxes before clearing the scan, and keeps it on save failure', async () => {
  const image = frame(), scan = scanFrame(image);
  const box = addManualBox(scan, image, [5, 6, 20, 21]);
  assert.ok(box);
  tagTrack(scan, box.track_id, { pin_id: 'pinpanion:7', name: 'Reviewed pin' });
  let currentSession = 'first', cleared = 0;
  let saved = makeFrameAnnotation('photo.jpg', null, scan.width, scan.height,
    'data:image/jpeg;base64,AAAA', scan.boxes);
  const save = async (): Promise<boolean> => {
    saved = makeFrameAnnotation('photo.jpg', null, scan.width, scan.height,
      'data:image/jpeg;base64,AAAA', scan.boxes, saved);
    return true;
  };
  assert.equal(await completeScan('first', () => currentSession, save, () => { cleared++; }), true);
  assert.equal(saved.boxes[0]?.pin_id, 'pinpanion:7');
  assert.equal(cleared, 1);
  assert.equal(await completeScan('first', () => currentSession, async () => false,
    () => { cleared++; }), false);
  assert.equal(cleared, 1);
  assert.equal(await completeScan('first', () => currentSession, async () => {
    currentSession = 'second'; return true;
  }, () => { cleared++; }), false);
  assert.equal(cleared, 1);
});

test('visually similar difficult crops rank behind clear candidates', () => {
  const pattern = Array.from({ length: 36 }, (_, index) => index % 2 ? 0.2 : -0.2);
  const unrelated = Array.from({ length: 36 }, (_, index) => index < 18 ? 0.2 : -0.2);
  const hints = [{ id: 'hard-one', signature: pattern }];
  assert.equal(difficultyPriority(pattern, hints), 1);
  assert.equal(difficultyPriority(unrelated, hints), 0);
  assert.equal(difficultyPriority(Array(36).fill(0), hints), 0);
  assert.ok(difficultyPriority(pattern, hints) > difficultyPriority(unrelated, hints));
});

test('difficulty sync shares only a signature, never the crop or a training label', async () => {
  const image = frame(), scan = scanFrame(image);
  const box = addManualBox(scan, image, [5, 6, 20, 21]);
  assert.ok(box);
  const mark = makeDifficultyMark('photo.jpg', null, box, 'data:image/jpeg;base64,AAAA');
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async (input, init) => {
    assert.equal(input, '/api/difficulty-marks');
    assert.equal(init?.method, 'POST');
    assert.deepEqual(JSON.parse(String(init?.body)), { id: mark.id, signature: mark.signature });
    return Response.json({ saved: true }, { status: 201 });
  };
  try { await postDifficultyMark(mark); }
  finally { globalThis.fetch = originalFetch; }
  globalThis.fetch = async (input) => {
    assert.equal(input, '/api/difficulty-marks');
    return Response.json({ marks: [{ id: mark.id, signature: mark.signature }] });
  };
  try { assert.deepEqual(await loadSharedDifficultyHints(), [{ id: mark.id, signature: mark.signature }]); }
  finally { globalThis.fetch = originalFetch; }
});

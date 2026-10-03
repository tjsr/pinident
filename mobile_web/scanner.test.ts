import assert from 'node:assert/strict';
import test from 'node:test';
import { detectCandidates, rejectTrack, scanFrame, tagTrack, type PixelFrame } from './dist/scanner.js';

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

test('on-device detector proposes an isolated object', () => {
  const boxes = detectCandidates(frame([[24, 28, 28, 28]]));
  assert.ok(boxes.some(([x, y, w, h]) => x <= 24 && y <= 28 && x + w >= 52 && y + h >= 56));
});

test('confirmed and rejected tracks carry to later frames', () => {
  const image = frame([[24, 28, 28, 28]]);
  const first = scanFrame(image);
  assert.ok(first.boxes.length > 0);
  const track = first.boxes[0]!.track_id;
  assert.equal(tagTrack(first, track, { pin_id: 'pinpanion:7', name: 'Test Pin' }), true);
  const second = scanFrame(image, first);
  const inherited = second.boxes.find(box => box.track_id === track);
  assert.equal(inherited?.review_state, 'inherited');
  assert.equal(inherited?.pin_id, 'pinpanion:7');
  assert.equal(rejectTrack(second, track), true);
  const third = scanFrame(image, second);
  assert.equal(third.boxes.find(box => box.track_id === track)?.review_state, 'rejected');
});

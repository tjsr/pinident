import { makeId, type PinIdentity } from './library.js';

export type Coords = [number, number, number, number];
export type ReviewState = 'unconfirmed' | 'confirmed' | 'inherited' | 'rejected';

export interface ScanBox {
  coords: Coords;
  track_id: string;
  review_state: ReviewState;
  pin_id: string | null;
  name: string | null;
  signature: number[];
}

export interface ScanState {
  width: number;
  height: number;
  boxes: ScanBox[];
}

export interface PixelFrame {
  width: number;
  height: number;
  data: Uint8ClampedArray;
}

function overlap(a: Coords, b: Coords): number {
  const area = Math.max(0, Math.min(a[0] + a[2], b[0] + b[2]) - Math.max(a[0], b[0])) *
    Math.max(0, Math.min(a[1] + a[3], b[1] + b[3]) - Math.max(a[1], b[1]));
  const union = a[2] * a[3] + b[2] * b[3] - area;
  return union > 0 ? area / union : 0;
}

function signature(frame: PixelFrame, box: Coords): number[] {
  const [x, y, width, height] = box;
  const values: number[] = [];
  for (let gy = 0; gy < 6; gy++) {
    for (let gx = 0; gx < 6; gx++) {
      const px = Math.min(frame.width - 1, Math.max(0, Math.floor(x + (gx + 0.5) * width / 6)));
      const py = Math.min(frame.height - 1, Math.max(0, Math.floor(y + (gy + 0.5) * height / 6)));
      const index = (py * frame.width + px) * 4;
      values.push((frame.data[index]! * 0.299 + frame.data[index + 1]! * 0.587 + frame.data[index + 2]! * 0.114) / 255);
    }
  }
  const mean = values.reduce((total, value) => total + value, 0) / values.length;
  return values.map(value => value - mean);
}

function similarity(a: number[], b: number[]): number {
  if (a.length !== b.length) return 0;
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i]! * b[i]!;
    normA += a[i]! * a[i]!;
    normB += b[i]! * b[i]!;
  }
  return normA > 0.005 && normB > 0.005 ? Math.max(0, dot / Math.sqrt(normA * normB)) : 0;
}

function components(mask: Uint8Array, width: number, height: number): Coords[] {
  const visited = new Uint8Array(mask.length);
  const minSide = Math.max(8, Math.floor(Math.min(width, height) * 0.04));
  const maxSide = Math.floor(Math.max(width, height) * 0.45);
  const boxes: Coords[] = [];
  const queue = new Int32Array(mask.length);
  for (let start = 0; start < mask.length; start++) {
    if (!mask[start] || visited[start]) continue;
    let head = 0, tail = 1, minX = start % width, maxX = minX;
    let minY = Math.floor(start / width), maxY = minY;
    visited[start] = 1;
    queue[0] = start;
    while (head < tail) {
      const point = queue[head++]!;
      const x = point % width, y = Math.floor(point / width);
      minX = Math.min(minX, x); maxX = Math.max(maxX, x);
      minY = Math.min(minY, y); maxY = Math.max(maxY, y);
      const neighbors = [x > 0 ? point - 1 : -1, x + 1 < width ? point + 1 : -1,
        y > 0 ? point - width : -1, y + 1 < height ? point + width : -1];
      for (const next of neighbors) {
        if (next >= 0 && mask[next] && !visited[next]) {
          visited[next] = 1;
          queue[tail++] = next;
        }
      }
    }
    const w = maxX - minX + 1, h = maxY - minY + 1;
    if (Math.min(w, h) < minSide || Math.max(w, h) > maxSide ||
        Math.max(w, h) / Math.min(w, h) > 3 || tail < minSide * minSide * 0.3) continue;
    const pad = Math.max(3, Math.floor(Math.max(w, h) * 0.08));
    const left = Math.max(0, minX - pad), top = Math.max(0, minY - pad);
    boxes.push([left, top, Math.min(width, maxX + pad + 1) - left,
      Math.min(height, maxY + pad + 1) - top]);
  }
  return boxes;
}

/** Coarse, intentionally class-agnostic proposals. A clean background is needed for best results. */
export function detectCandidates(frame: PixelFrame): Coords[] {
  const { width, height, data } = frame;
  if (width < 24 || height < 24 || data.length !== width * height * 4) return [];
  const border: number[][] = [];
  const step = Math.max(1, Math.floor(Math.min(width, height) / 60));
  for (let x = 0; x < width; x += step) {
    for (const y of [0, height - 1]) {
      const i = (y * width + x) * 4;
      border.push([data[i]!, data[i + 1]!, data[i + 2]!]);
    }
  }
  for (let y = 0; y < height; y += step) {
    for (const x of [0, width - 1]) {
      const i = (y * width + x) * 4;
      border.push([data[i]!, data[i + 1]!, data[i + 2]!]);
    }
  }
  const background = [0, 1, 2].map(channel => {
    const sorted = border.map(pixel => pixel[channel]!).sort((a, b) => a - b);
    return sorted[Math.floor(sorted.length / 2)]!;
  });
  const borderVariation = border.reduce((sum, pixel) => sum + Math.hypot(
    pixel[0]! - background[0]!, pixel[1]! - background[1]!, pixel[2]! - background[2]!), 0) / border.length;
  const globalMask = new Uint8Array(width * height);
  const localMask = new Uint8Array(width * height);

  // Integral luminance allows a local contrast test without quadratic per-pixel work.
  const stride = width + 1;
  const integral = new Float64Array(stride * (height + 1));
  for (let y = 0; y < height; y++) {
    let row = 0;
    for (let x = 0; x < width; x++) {
      const p = (y * width + x) * 4;
      row += data[p]! * 0.299 + data[p + 1]! * 0.587 + data[p + 2]! * 0.114;
      integral[(y + 1) * stride + x + 1] = integral[y * stride + x + 1]! + row;
    }
  }
  const radius = Math.max(8, Math.floor(Math.min(width, height) * 0.035));
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const i = y * width + x, p = i * 4;
      if (borderVariation < 38 && Math.hypot(data[p]! - background[0]!,
        data[p + 1]! - background[1]!, data[p + 2]! - background[2]!) > 48) globalMask[i] = 1;
      const x0 = Math.max(0, x - radius), x1 = Math.min(width, x + radius + 1);
      const y0 = Math.max(0, y - radius), y1 = Math.min(height, y + radius + 1);
      const mean = (integral[y1 * stride + x1]! - integral[y0 * stride + x1]! -
        integral[y1 * stride + x0]! + integral[y0 * stride + x0]!) / ((x1 - x0) * (y1 - y0));
      const luminance = data[p]! * 0.299 + data[p + 1]! * 0.587 + data[p + 2]! * 0.114;
      if (Math.abs(luminance - mean) > 40) localMask[i] = 1;
    }
  }
  const proposed = [...components(globalMask, width, height), ...components(localMask, width, height)];
  proposed.sort((a, b) => b[2] * b[3] - a[2] * a[3]);
  const result: Coords[] = [];
  for (const box of proposed) {
    if (result.every(previous => overlap(box, previous) < 0.35)) result.push(box);
    if (result.length >= 40) break;
  }
  return result;
}

export function scanFrame(frame: PixelFrame, previous: ScanState | null = null): ScanState {
  const proposals = detectCandidates(frame);
  const candidates = proposals.map(coords => ({ coords, signature: signature(frame, coords) }));
  const used = new Set<string>();
  const boxes = candidates.map(({ coords, signature: appearance }): ScanBox => {
    let best: ScanBox | null = null, bestScore = 0;
    if (previous && previous.width === frame.width && previous.height === frame.height) {
      for (const old of previous.boxes) {
        if (used.has(old.track_id)) continue;
        const iou = overlap(coords, old.coords);
        if (iou < 0.20) continue;
        const visual = similarity(appearance, old.signature);
        if (visual < 0.58) continue;
        const score = iou * 0.6 + visual * 0.4;
        if (score > bestScore) { best = old; bestScore = score; }
      }
    }
    if (best) {
      used.add(best.track_id);
      return { coords, signature: appearance, track_id: best.track_id,
        review_state: best.review_state === 'rejected' ? 'rejected' :
          best.review_state === 'unconfirmed' ? 'unconfirmed' : 'inherited',
        pin_id: best.pin_id, name: best.name };
    }
    return { coords, signature: appearance, track_id: makeId(),
      review_state: 'unconfirmed', pin_id: null, name: null };
  });
  return { width: frame.width, height: frame.height, boxes };
}

export function tagTrack(state: ScanState, trackId: string, pin: PinIdentity): boolean {
  const box = state.boxes.find(candidate => candidate.track_id === trackId);
  if (!box || box.review_state === 'rejected') return false;
  box.pin_id = pin.pin_id;
  box.name = pin.name;
  box.review_state = 'confirmed';
  return true;
}

export function rejectTrack(state: ScanState, trackId: string): boolean {
  const box = state.boxes.find(candidate => candidate.track_id === trackId);
  if (!box) return false;
  box.pin_id = null;
  box.name = null;
  box.review_state = 'rejected';
  return true;
}

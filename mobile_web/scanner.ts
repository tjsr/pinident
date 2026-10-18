import { makeId, type PinIdentity } from './library.js';
import { classifyProposal, type MobileModel } from './model.js';
import { matchProposal, type IdentityIndex } from './identity.js';

export type Coords = [number, number, number, number];
export type ReviewState = 'unconfirmed' | 'confirmed' | 'inherited' | 'rejected';

export interface ScanBox {
  coords: Coords;
  track_id: string;
  review_state: ReviewState;
  pin_id: string | null;
  name: string | null;
  suggested_pin_id?: string | null;
  suggested_name?: string | null;
  match_strength?: number | null;
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

export interface ScanDiagnostics {
  proposals: Coords[];
  modelRejected: Coords[];
  tracked: number;
  belowMatchThreshold?: Coords[];
  sizeFiltered?: Coords[];
}

/** A marquee must cover strictly more than half of each visible box it selects. */
export function selectAreasInRectangle(state: ScanState, rectangle: Coords): string[] {
  const [left, top, width, height] = rectangle;
  if (width <= 0 || height <= 0) return [];
  return state.boxes.filter(box => {
    if (box.review_state === 'rejected') return false;
    const [x, y, w, h] = box.coords;
    const covered = Math.max(0, Math.min(left + width, x + w) - Math.max(left, x)) *
      Math.max(0, Math.min(top + height, y + h) - Math.max(top, y));
    return covered > w * h / 2;
  }).map(box => box.track_id);
}

export type AreaDragMode = 'auto' | 'draw' | 'select';

/** Default drags select covered areas; an empty marquee becomes a new area. */
export function resolveAreaDrag(state: ScanState, rectangle: Coords, mode: AreaDragMode):
  { action: 'select'; trackIds: string[] } | { action: 'draw' } {
  if (mode === 'draw') return { action: 'draw' };
  const trackIds = selectAreasInRectangle(state, rectangle);
  return mode === 'select' || trackIds.length ? { action: 'select', trackIds } : { action: 'draw' };
}

/** Delete annotations only; this does not create a non-pin review. */
export function deleteAreas(state: ScanState, trackIds: ReadonlySet<string>): number {
  const before = state.boxes.length;
  state.boxes = state.boxes.filter(box => !trackIds.has(box.track_id));
  return before - state.boxes.length;
}

export function areaAtPoint(state: ScanState, point: [number, number]): string | null {
  const [px, py] = point;
  return state.boxes.filter(box => {
    if (box.review_state === 'rejected') return false;
    const [x, y, width, height] = box.coords;
    return px >= x && py >= y && px <= x + width && py <= y + height;
  }).sort((left, right) => left.coords[2] * left.coords[3] -
    right.coords[2] * right.coords[3])[0]?.track_id ?? null;
}

function intersectionArea(a: Coords, b: Coords): number {
  return Math.max(0, Math.min(a[0] + a[2], b[0] + b[2]) - Math.max(a[0], b[0])) *
    Math.max(0, Math.min(a[1] + a[3], b[1] + b[3]) - Math.max(a[1], b[1]));
}

function overlap(a: Coords, b: Coords): number {
  const area = intersectionArea(a, b);
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
  // A nearby pin can occupy most of the camera frame. The old 45% cap discarded
  // its silhouette and left only small high-contrast artwork as candidates.
  const maxSide = Math.floor(Math.max(width, height) * 0.95);
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
    // IoU alone retains tiny artwork inside a close-up pin because its area is
    // small relative to the silhouette. Compare coverage of the smaller box.
    if (result.every(previous => overlap(box, previous) < 0.35 &&
        intersectionArea(box, previous) < box[2] * box[3] * 0.85)) result.push(box);
    if (result.length >= 40) break;
  }
  return result;
}

export function scanFrame(frame: PixelFrame, previous: ScanState | null = null,
                          model: MobileModel | null = null,
                          diagnostics?: ScanDiagnostics,
                          identityIndex: IdentityIndex | null = null,
                          negativeExamples: readonly number[][] = []): ScanState {
  const proposals = detectCandidates(frame);
  if (diagnostics) {
    diagnostics.proposals = proposals.map(coords => [...coords]);
    diagnostics.modelRejected = [];
    diagnostics.tracked = 0;
    diagnostics.belowMatchThreshold = [];
    diagnostics.sizeFiltered = [];
  }
  const candidates = proposals.map(coords => {
    const appearance = signature(frame, coords);
    const match = identityIndex
      ? matchProposal(identityIndex, frame, coords, negativeExamples) : null;
    const modelNegative = model && classifyProposal(model, frame, coords) === 'negative';
    return { coords, appearance, match, modelNegative };
  });
  const anchor = strongestMatchSizeReference(candidates.filter(candidate =>
    !candidate.modelNegative).map(candidate => ({
      coords: candidate.coords, strength: candidate.match?.strength ?? 0
    })));
  const used = new Set<string>();
  const boxes: ScanBox[] = [];
  for (const { coords, appearance, match, modelNegative } of candidates) {
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
      if (diagnostics) diagnostics.tracked++;
      used.add(best.track_id);
      const updatedSuggestion = !best.pin_id && !!match && match.strength >= 0.5;
      boxes.push({ coords, signature: appearance, track_id: best.track_id,
        review_state: best.review_state === 'rejected' ? 'rejected' :
          best.review_state === 'unconfirmed' ? 'unconfirmed' : 'inherited',
        pin_id: best.pin_id, name: best.name,
        suggested_pin_id: updatedSuggestion ? match!.pin_id : null,
        suggested_name: updatedSuggestion ? match!.name : null,
        match_strength: updatedSuggestion ? match!.strength : null });
      continue;
    }
    if (modelNegative) {
      if (diagnostics) diagnostics.modelRejected.push([...coords]);
      continue;
    }
    if (!match || match.strength < 0.5) {
      if (diagnostics) diagnostics.belowMatchThreshold!.push([...coords]);
      continue;
    }
    if (anchor && !similarSize(coords, anchor, frame.width, frame.height)) {
      if (diagnostics) diagnostics.sizeFiltered!.push([...coords]);
      continue;
    }
    boxes.push({ coords, signature: appearance, track_id: makeId(),
      review_state: 'unconfirmed', pin_id: null, name: null,
      suggested_pin_id: match.pin_id, suggested_name: match.name,
      match_strength: match.strength });
  }
  return { width: frame.width, height: frame.height, boxes };
}

export function strongestMatchSizeReference(
  candidates: ReadonlyArray<{ coords: Coords; strength: number }>): Coords | null {
  let best: { coords: Coords; strength: number } | null = null;
  for (const candidate of candidates) {
    if (candidate.strength >= 0.5 && (!best || candidate.strength > best.strength))
      best = candidate;
  }
  return best?.coords ?? null;
}

export function similarSize(coords: Coords, reference: Coords,
                            frameWidth: number, frameHeight: number): boolean {
  const clippedX = coords[0] === 0 || coords[0] + coords[2] === frameWidth;
  const clippedY = coords[1] === 0 || coords[1] + coords[3] === frameHeight;
  return (coords[2] <= reference[2] * 4 / 3 &&
    (clippedX || coords[2] >= reference[2] * 0.75)) &&
    (coords[3] <= reference[3] * 4 / 3 &&
    (clippedY || coords[3] >= reference[3] * 0.75));
}

export function tagTrack(state: ScanState, trackId: string, pin: PinIdentity): boolean {
  const box = state.boxes.find(candidate => candidate.track_id === trackId);
  if (!box || box.review_state === 'rejected') return false;
  box.pin_id = pin.pin_id;
  box.name = pin.name;
  box.suggested_pin_id = null;
  box.suggested_name = null;
  box.match_strength = null;
  box.review_state = 'confirmed';
  return true;
}

export function confirmUnknown(state: ScanState, trackId: string): boolean {
  const box = state.boxes.find(candidate => candidate.track_id === trackId);
  if (!box || box.review_state === 'rejected') return false;
  box.pin_id = null;
  box.name = null;
  box.suggested_pin_id = null;
  box.suggested_name = null;
  box.match_strength = null;
  box.review_state = 'confirmed';
  return true;
}

export function addManualBox(state: ScanState, frame: PixelFrame, coords: Coords): ScanBox | null {
  const [x, y, width, height] = coords;
  if (state.width !== frame.width || state.height !== frame.height ||
      !Number.isInteger(x) || !Number.isInteger(y) || !Number.isInteger(width) ||
      !Number.isInteger(height) || x < 0 || y < 0 || width < 4 || height < 4 ||
      x + width > frame.width || y + height > frame.height) return null;
  const box: ScanBox = { coords, track_id: makeId(), review_state: 'confirmed',
    pin_id: null, name: null, signature: signature(frame, coords) };
  state.boxes.push(box);
  return box;
}

export function rejectTrack(state: ScanState, trackId: string): boolean {
  const box = state.boxes.find(candidate => candidate.track_id === trackId);
  if (!box) return false;
  box.pin_id = null;
  box.name = null;
  box.suggested_pin_id = null;
  box.suggested_name = null;
  box.match_strength = null;
  box.review_state = 'rejected';
  return true;
}

import type { Coords, PixelFrame } from './scanner.js';

const GRID = 8;
const CELLS = GRID * GRID;
const LENGTH = CELLS + 3;
const LEARNED_KEY = 'pinident.mobile.identity-examples.v1';

export interface CatalogReference {
  pin_id: string;
  name: string;
  source: string;
  rights: string;
  feature: number[];
}

export interface IdentityIndex {
  format: 'pinident-catalog-reference-index';
  version: 1;
  catalog_state: [number, number, number];
  catalog_rights_unverified: number;
  references: CatalogReference[];
}

export interface IdentityMatch {
  pin_id: string;
  name: string;
  /** Relative visual match strength, not a calibrated probability. */
  strength: number;
}

export interface LearnedExamples {
  positive: CatalogReference[];
  negative: number[][];
}

export function parseIdentityIndex(value: unknown): IdentityIndex {
  if (!value || typeof value !== 'object') throw new Error('Invalid catalog reference index.');
  const row = value as Record<string, unknown>;
  if (row.format !== 'pinident-catalog-reference-index' || row.version !== 1 ||
      !Array.isArray(row.catalog_state) || row.catalog_state.length !== 3 ||
      !row.catalog_state.every(item => Number.isSafeInteger(item) && item >= 0) ||
      !Number.isSafeInteger(row.catalog_rights_unverified) ||
      (row.catalog_rights_unverified as number) < 0 ||
      !Array.isArray(row.references) || !row.references.length ||
      !row.references.every(value => {
        if (!value || typeof value !== 'object') return false;
        const item = value as Record<string, unknown>;
        return typeof item.pin_id === 'string' && /^pinpanion:\d+$/.test(item.pin_id) &&
          typeof item.name === 'string' && !!item.name.trim() &&
          typeof item.source === 'string' && typeof item.rights === 'string' &&
          Array.isArray(item.feature) && item.feature.length === LENGTH &&
          item.feature.every(part => typeof part === 'number' && Number.isFinite(part));
      })) throw new Error('Unsupported or damaged catalog reference index.');
  return value as IdentityIndex;
}

/** Keep the sampling rule identical to build_identity_index.py. */
export function referenceFeature(frame: PixelFrame, coords: Coords): number[] | null {
  const [x, y, width, height] = coords;
  if (width < 1 || height < 1 || x < 0 || y < 0 ||
      x + width > frame.width || y + height > frame.height ||
      frame.data.length !== frame.width * frame.height * 4) return null;
  const luminance = new Array<number>(CELLS);
  const channels = [0, 0, 0];
  for (let gy = 0; gy < GRID; gy++) for (let gx = 0; gx < GRID; gx++) {
    const px = x + Math.min(width - 1, Math.floor((gx + 0.5) * width / GRID));
    const py = y + Math.min(height - 1, Math.floor((gy + 0.5) * height / GRID));
    const offset = (py * frame.width + px) * 4;
    const red = frame.data[offset]! / 255;
    const green = frame.data[offset + 1]! / 255;
    const blue = frame.data[offset + 2]! / 255;
    luminance[gy * GRID + gx] = red * 0.299 + green * 0.587 + blue * 0.114;
    channels[0]! += red; channels[1]! += green; channels[2]! += blue;
  }
  const mean = luminance.reduce((sum, value) => sum + value, 0) / CELLS;
  const norm = Math.hypot(...luminance.map(value => value - mean));
  if (norm < 0.18) return null;
  return [...luminance.map(value => (value - mean) / norm),
    ...channels.map(value => value / CELLS)];
}

function likeness(feature: number[], reference: number[]): { score: number; colour: number } {
  let shape = 0;
  for (let i = 0; i < CELLS; i++) shape += feature[i]! * reference[i]!;
  shape = Math.max(0, Math.min(1, shape));
  const colour = 1 - (Math.abs(feature[64]! - reference[64]!) +
    Math.abs(feature[65]! - reference[65]!) +
    Math.abs(feature[66]! - reference[66]!)) / 3;
  return { score: shape * shape * (0.7 + 0.3 * Math.max(0, colour)), colour };
}

export function matchReference(index: IdentityIndex, feature: number[] | null,
                               negatives: readonly number[][] = []): IdentityMatch | null {
  if (!feature) return null;
  let best: CatalogReference | null = null;
  let bestShape = 0, bestColour = 0, runnerUp = 0;
  for (const reference of index.references) {
    const matched = likeness(feature, reference.feature);
    if (matched.score > bestShape) {
      if (best && best.pin_id !== reference.pin_id) runnerUp = Math.max(runnerUp, bestShape);
      best = reference; bestShape = matched.score; bestColour = matched.colour;
    } else if (reference.pin_id !== best?.pin_id) runnerUp = Math.max(runnerUp, matched.score);
  }
  if (!best) return null;
  if (negatives.some(negative => likeness(feature, negative).score >= Math.max(0.65, bestShape * 0.95)))
    return null;
  // Similar catalog designs should not produce a confident named suggestion.
  const separation = Math.min(1, Math.max(0, bestShape - runnerUp) / 0.15);
  const strength = Math.max(0, Math.min(1, bestShape * (0.5 + 0.5 * separation)));
  if (bestColour < 0.45) return null;
  return { pin_id: best.pin_id, name: best.name, strength };
}

/** Detector padding varies, so compare the whole proposal and a few inner crops. */
export function matchProposal(index: IdentityIndex, frame: PixelFrame, coords: Coords,
                              negatives: readonly number[][] = []): IdentityMatch | null {
  const whole = referenceFeature(frame, coords);
  if (whole && negatives.some(negative => likeness(whole, negative).score >= 0.8))
    return null;
  let best: IdentityMatch | null = null;
  for (const fraction of [0, 0.05, 0.1, 0.15]) {
    const insetX = Math.round(coords[2] * fraction);
    const insetY = Math.round(coords[3] * fraction);
    const inner: Coords = [coords[0] + insetX, coords[1] + insetY,
      coords[2] - 2 * insetX, coords[3] - 2 * insetY];
    const feature = fraction === 0 ? whole : referenceFeature(frame, inner);
    const match = matchReference(index, feature, negatives);
    if (match && (!best || match.strength > best.strength)) best = match;
  }
  return best;
}

function validFeature(value: unknown): value is number[] {
  return Array.isArray(value) && value.length === LENGTH &&
    value.every(part => typeof part === 'number' && Number.isFinite(part));
}

export function loadLearnedExamples(storage: Pick<Storage, 'getItem'> = localStorage): LearnedExamples {
  try {
    const value: unknown = JSON.parse(storage.getItem(LEARNED_KEY) ?? 'null');
    if (!value || typeof value !== 'object') return { positive: [], negative: [] };
    const row = value as Record<string, unknown>;
    const positive = Array.isArray(row.positive) ? row.positive.filter(item => {
      if (!item || typeof item !== 'object') return false;
      const reference = item as Record<string, unknown>;
      return typeof reference.pin_id === 'string' && /^pinpanion:\d+$/.test(reference.pin_id) &&
        typeof reference.name === 'string' && validFeature(reference.feature);
    }) as CatalogReference[] : [];
    const negative = Array.isArray(row.negative)
      ? row.negative.filter(validFeature) : [];
    return { positive: positive.slice(-200), negative: negative.slice(-200) };
  } catch { return { positive: [], negative: [] }; }
}

export function learnPositive(examples: LearnedExamples, pinId: string, name: string,
                              feature: number[] | null): void {
  if (!feature || !/^pinpanion:\d+$/.test(pinId) || !name.trim()) return;
  examples.positive.push({ pin_id: pinId, name: name.trim(), source: 'local review',
    rights: 'local review; source rights unverified', feature });
  examples.positive = examples.positive.slice(-200);
}

export function learnNegative(examples: LearnedExamples, feature: number[] | null): void {
  if (!feature) return;
  examples.negative.push(feature);
  examples.negative = examples.negative.slice(-200);
}

export function saveLearnedExamples(examples: LearnedExamples,
                                    storage: Pick<Storage, 'setItem'> = localStorage): void {
  try { storage.setItem(LEARNED_KEY, JSON.stringify(examples)); }
  catch { /* Scanning still works if local storage is full or disabled. */ }
}

export function combineIdentityIndex(catalog: IdentityIndex | null,
                                     examples: LearnedExamples): IdentityIndex | null {
  if (!catalog && !examples.positive.length) return null;
  return { format: 'pinident-catalog-reference-index', version: 1,
    catalog_state: catalog?.catalog_state ?? [0, 0, 0],
    catalog_rights_unverified: catalog?.catalog_rights_unverified ?? 0,
    references: [...(catalog?.references ?? []), ...examples.positive] };
}

export async function loadIdentityIndex(): Promise<IdentityIndex | null> {
  const response = await fetch('/identity-index.json', { cache: 'no-cache' });
  if (response.status === 404) return null;
  if (!response.ok) throw new Error(`Catalog reference index download failed (${response.status}).`);
  if (!response.headers.get('content-type')?.includes('application/json')) return null;
  return parseIdentityIndex(await response.json() as unknown);
}

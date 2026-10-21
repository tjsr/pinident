import type { Coords, PixelFrame } from './scanner.js';
import type { JsonInput, JsonRecord } from './json_input.js';

const GRID = 16;
const FEATURE_LENGTH = GRID * GRID + 2;

export interface MobileModel {
  format: 'pinident-linear-svm';
  version: 1;
  feature_version: 2;
  training_digest: string;
  sample_counts: { positive: number; negative: number };
  web_review_revision?: number;
  weights: number[];
  bias: number;
}

export function parseMobileModel(value: JsonInput): MobileModel {
  if (typeof value !== 'object' || value === null) throw new Error('Invalid mobile model.');
  const row = value as JsonRecord;
  const counts = row.sample_counts as JsonRecord | undefined;
  if (row.format !== 'pinident-linear-svm' || row.version !== 1 ||
      row.feature_version !== 2 || typeof row.training_digest !== 'string' ||
      !/^[a-f0-9]{64}$/.test(row.training_digest) ||
      !counts || !Number.isInteger(counts.positive) || !Number.isInteger(counts.negative) ||
      (counts.positive as number) < 1 || (counts.negative as number) < 1 ||
      !Array.isArray(row.weights) || row.weights.length !== FEATURE_LENGTH ||
      !row.weights.every((weight: JsonInput) => typeof weight === 'number' && Number.isFinite(weight)) ||
      typeof row.bias !== 'number' || !Number.isFinite(row.bias)) {
    throw new Error('Unsupported or damaged mobile model.');
  }
  return value as MobileModel;
}

export function proposalFeature(frame: PixelFrame, coords: Coords): Float32Array | null {
  const [x, y, width, height] = coords;
  if (width < 1 || height < 1 || x < 0 || y < 0 ||
      x + width > frame.width || y + height > frame.height ||
      frame.data.length !== frame.width * frame.height * 4) return null;
  const luminance = new Float64Array(GRID * GRID);
  const pixel = (px: number, py: number, channel: number): number =>
    frame.data[((y + py) * frame.width + x + px) * 4 + channel]!;
  for (let gy = 0; gy < GRID; gy++) {
    const sy = Math.min(height - 1, Math.max(0, (gy + 0.5) * height / GRID - 0.5));
    const y0 = Math.floor(sy), y1 = Math.min(height - 1, y0 + 1), fy = sy - y0;
    for (let gx = 0; gx < GRID; gx++) {
      const sx = Math.min(width - 1, Math.max(0, (gx + 0.5) * width / GRID - 0.5));
      const x0 = Math.floor(sx), x1 = Math.min(width - 1, x0 + 1), fx = sx - x0;
      const channel = (index: number): number => {
        const top = pixel(x0, y0, index) * (1 - fx) + pixel(x1, y0, index) * fx;
        const bottom = pixel(x0, y1, index) * (1 - fx) + pixel(x1, y1, index) * fx;
        return top * (1 - fy) + bottom * fy;
      };
      luminance[gy * GRID + gx] = (channel(0) * 0.299 + channel(1) * 0.587 +
        channel(2) * 0.114) / 255;
    }
  }
  const mean = luminance.reduce((sum, value) => sum + value, 0) / luminance.length;
  let norm = 0;
  for (const value of luminance) norm += (value - mean) ** 2;
  norm = Math.sqrt(norm);
  if (norm < 0.25) return null;
  const feature = new Float32Array(FEATURE_LENGTH);
  for (let i = 0; i < luminance.length; i++) feature[i] = (luminance[i]! - mean) / norm;
  const sums = [0, 0, 0];
  for (let py = 0; py < height; py++) {
    for (let px = 0; px < width; px++) {
      for (let c = 0; c < 3; c++) sums[c] = sums[c]! + pixel(px, py, c);
    }
  }
  const divisor = width * height * 255;
  feature[256] = ((sums[0]! - sums[1]!) / divisor) * 0.15;
  feature[257] = ((sums[2]! - sums[1]!) / divisor) * 0.15;
  return feature;
}

export function classifyProposal(model: MobileModel, frame: PixelFrame,
                                 coords: Coords): 'positive' | 'negative' | null {
  const feature = proposalFeature(frame, coords);
  if (!feature) return null;
  let score = model.bias;
  for (let i = 0; i < feature.length; i++) score += model.weights[i]! * feature[i]!;
  return score >= 0 ? 'positive' : 'negative';
}

export async function loadMobileModel(): Promise<MobileModel | null> {
  const response = await fetch('/model.json', { cache: 'no-cache' });
  if (response.status === 404) return null;
  if (!response.ok) throw new Error(`Model download failed (${response.status}).`);
  if (!response.headers.get('content-type')?.includes('application/json')) return null;
  const model = await response.json() as JsonInput;
  return parseMobileModel(model);
}

import { makeId, type PinIdentity } from './library.js';
import type { Coords, ScanBox } from './scanner.js';
import type { JsonInput } from './json_input.js';

export type TrainingLabel = 'not_pin' | 'pin_unknown' | 'pin_identified';
export interface TrainingReview {
  kind: 'training';
  id: string;
  created_at: string;
  source: string;
  rights_status: 'unverified';
  time_seconds: number | null;
  coords: Coords;
  image_data_url: string;
  suggested_pin_id: string | null;
  suggested_name: string | null;
  label: TrainingLabel;
  pin_id: string | null;
  name: string | null;
  remote_revision?: number;
}

export interface DifficultyMark {
  kind: 'difficulty';
  id: string;
  created_at: string;
  source: string;
  rights_status: 'unverified';
  time_seconds: number | null;
  coords: Coords;
  image_data_url: string;
  signature: number[];
  remote_saved?: boolean;
}

export type DifficultyHint = Pick<DifficultyMark, 'id' | 'signature'>;

export interface FrameAnnotation {
  kind: 'annotation';
  id: string;
  created_at: string;
  source: string;
  rights_status: 'unverified';
  time_seconds: number | null;
  width: number;
  height: number;
  image_data_url: string;
  boxes: Array<{ coords: Coords; track_id: string; review_state: ScanBox['review_state'];
    pin_id: string | null; name: string | null }>;
  remote_revision?: number;
}

export type ReviewRecord = TrainingReview | DifficultyMark | FrameAnnotation;

export function clampDrawnBox(start: [number, number], end: [number, number],
                             width: number, height: number): Coords | null {
  const x0 = Math.max(0, Math.min(width, Math.min(start[0], end[0])));
  const y0 = Math.max(0, Math.min(height, Math.min(start[1], end[1])));
  const x1 = Math.max(0, Math.min(width, Math.max(start[0], end[0])));
  const y1 = Math.max(0, Math.min(height, Math.max(start[1], end[1])));
  if (x1 - x0 < 4 || y1 - y0 < 4) return null;
  return [Math.round(x0), Math.round(y0), Math.round(x1 - x0), Math.round(y1 - y0)];
}

export function makeTrainingReview(source: string, time: number | null, box: ScanBox,
                                   image: string, label: TrainingLabel,
                                   identity: PinIdentity | null = null): TrainingReview {
  if (label === 'pin_identified' && !identity) throw new Error('Choose a pin identity.');
  return { kind: 'training', id: makeId(), created_at: new Date().toISOString(),
    source, rights_status: 'unverified', time_seconds: time, coords: [...box.coords], image_data_url: image,
    suggested_pin_id: box.suggested_pin_id ?? null,
    suggested_name: box.suggested_name ?? null, label,
    pin_id: label === 'pin_identified' ? identity!.pin_id : null,
    name: label === 'pin_identified' ? identity!.name : null };
}

export function makeDifficultyMark(source: string, time: number | null, box: ScanBox,
                                   image: string): DifficultyMark {
  return { kind: 'difficulty', id: makeId(), created_at: new Date().toISOString(),
    source, rights_status: 'unverified', time_seconds: time, coords: [...box.coords],
    image_data_url: image, signature: [...box.signature] };
}

function signatureSimilarity(a: number[], b: number[]): number {
  if (a.length !== 36 || b.length !== 36) return 0;
  let dot = 0, normA = 0, normB = 0;
  for (let index = 0; index < 36; index++) {
    dot += a[index]! * b[index]!;
    normA += a[index]! * a[index]!;
    normB += b[index]! * b[index]!;
  }
  return normA > 0.01 && normB > 0.01 ? dot / Math.sqrt(normA * normB) : 0;
}

/** Marks affect ordering only; they never classify a candidate. */
export function difficultyPriority(signature: number[], hints: DifficultyHint[]): number {
  return hints.reduce((count, hint) => count +
    (signatureSimilarity(signature, hint.signature) >= 0.94 ? 1 : 0), 0);
}

export function makeFrameAnnotation(source: string, time: number | null, width: number,
                                    height: number, image: string, boxes: ScanBox[],
                                    previous?: FrameAnnotation): FrameAnnotation {
  return { kind: 'annotation', id: previous?.id ?? makeId(),
    created_at: previous?.created_at ?? new Date().toISOString(), source,
    rights_status: 'unverified', time_seconds: time, width, height, image_data_url: image,
    boxes: boxes.map(box => ({
      coords: [...box.coords], track_id: box.track_id, review_state: box.review_state,
      pin_id: box.pin_id, name: box.name
    })) };
}

const DB_NAME = 'pinident.mobile.reviews';
const STORE = 'records';

function database(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, 1);
    request.onupgradeneeded = () => { request.result.createObjectStore(STORE, { keyPath: 'id' }); };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

export async function saveRecord(record: ReviewRecord): Promise<void> {
  const db = await database();
  try {
    await new Promise<void>((resolve, reject) => {
      const tx = db.transaction(STORE, 'readwrite');
      tx.objectStore(STORE).put(record);
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
      tx.onabort = () => reject(tx.error);
    });
  } finally { db.close(); }
}

/** A completed upload must not overwrite a newer local edit of the same frame. */
async function markUploadedIfCurrent(before: ReviewRecord, after: ReviewRecord): Promise<void> {
  const db = await database();
  try {
    await new Promise<void>((resolve, reject) => {
      const tx = db.transaction(STORE, 'readwrite');
      const store = tx.objectStore(STORE);
      const request = store.get(before.id);
      request.onsuccess = () => {
        if (JSON.stringify(request.result) === JSON.stringify(before)) store.put(after);
      };
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
      tx.onabort = () => reject(tx.error);
    });
  } finally { db.close(); }
}

/** Both review crops and annotated frames retain their images in D1. */
async function postSharedRecord(record: TrainingReview | FrameAnnotation): Promise<number> {
  const { remote_revision: _localRevision, ...payload } = record;
  const response = await fetch('/api/training-reviews', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  });
  if (!response.ok) throw new Error(`Shared review upload failed (${response.status}).`);
  const result = await response.json() as JsonInput;
  if (typeof result !== 'object' || result === null ||
      !('revision' in result) || !Number.isSafeInteger(result.revision))
    throw new Error('Shared review upload returned an invalid revision.');
  return result.revision as number;
}

export function postTrainingReview(review: TrainingReview): Promise<number> {
  return postSharedRecord(review);
}

export function postFrameAnnotation(annotation: FrameAnnotation): Promise<number> {
  return postSharedRecord(annotation);
}

/** Upload a durable local review. A failed request leaves it queued in IndexedDB. */
export async function uploadTrainingReview(review: TrainingReview): Promise<TrainingReview> {
  const revision = await postTrainingReview(review);
  const updated = { ...review, remote_revision: revision };
  await markUploadedIfCurrent(review, updated);
  return updated;
}

export async function postDifficultyMark(mark: DifficultyMark): Promise<void> {
  const response = await fetch('/api/difficulty-marks', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ id: mark.id, signature: mark.signature })
  });
  if (!response.ok) throw new Error(`Difficulty mark upload failed (${response.status}).`);
}

export async function loadSharedDifficultyHints(signal?: AbortSignal): Promise<DifficultyHint[]> {
  const response = await fetch('/api/difficulty-marks', { signal, cache: 'no-store' });
  if (!response.ok) throw new Error(`Difficulty hints unavailable (${response.status}).`);
  const data = await response.json() as JsonInput;
  if (typeof data !== 'object' || data === null || !('marks' in data) ||
      !Array.isArray(data.marks) || data.marks.length > 500 ||
      !data.marks.every((mark: JsonInput) => typeof mark === 'object' && mark !== null &&
        'id' in mark && typeof mark.id === 'string' &&
        'signature' in mark && Array.isArray(mark.signature) && mark.signature.length === 36 &&
        mark.signature.every((value: JsonInput) => typeof value === 'number' && Number.isFinite(value))))
    throw new Error('Invalid shared difficulty hints.');
  return data.marks as DifficultyHint[];
}

let pendingSync: Promise<number> | null = null;
export function syncPendingReviews(): Promise<number> {
  if (pendingSync) return pendingSync;
  pendingSync = (async () => {
    const records = await loadRecords();
    let uploaded = 0;
    for (const review of records) {
      if (review.kind === 'difficulty') {
        if (review.remote_saved) continue;
        await postDifficultyMark(review);
        await saveRecord({ ...review, remote_saved: true });
      } else if (review.kind === 'training') {
        if (review.remote_revision) continue;
        await uploadTrainingReview(review);
      } else if (review.kind === 'annotation') {
        if (review.remote_revision) continue;
        const revision = await postFrameAnnotation(review);
        await markUploadedIfCurrent(review, { ...review, remote_revision: revision });
      } else continue;
      uploaded++;
    }
    return uploaded;
  })().finally(() => { pendingSync = null; });
  return pendingSync;
}

export async function loadRecords(): Promise<ReviewRecord[]> {
  const db = await database();
  try {
    return await new Promise<ReviewRecord[]>((resolve, reject) => {
      const tx = db.transaction(STORE, 'readonly');
      const request = tx.objectStore(STORE).getAll();
      request.onsuccess = () => resolve(request.result as ReviewRecord[]);
      request.onerror = () => reject(request.error);
    });
  } finally { db.close(); }
}

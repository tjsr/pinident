import type { PinIdentity } from './library.js';

const DATABASE_NAME = 'pinident.mobile.catalog';
const STORE = 'snapshot';

export interface CatalogPin extends PinIdentity {
  year?: number;
  category_ids: number[];
  set_id?: number;
  event_id?: number;
  group_id?: number;
}

export interface NamedFacet { id: number; name: string; }

export interface CatalogSnapshot {
  pins: CatalogPin[];
  categories: NamedFacet[];
  sets: NamedFacet[];
  events: NamedFacet[];
  groups: NamedFacet[];
}

export interface CatalogFilters {
  query: string;
  year: number | null;
  categoryId: number | null;
  setId: number | null;
  eventId: number | null;
  groupId: number | null;
}

export type FacetKey = Exclude<keyof CatalogFilters, 'query'>;
export interface FacetChoice { id: number; label: string; count: number; }

export function emptyCatalog(): CatalogSnapshot {
  return { pins: [], categories: [], sets: [], events: [], groups: [] };
}

export function emptyFilters(): CatalogFilters {
  return { query: '', year: null, categoryId: null, setId: null, eventId: null, groupId: null };
}

function validId(value: unknown): value is number {
  return typeof value === 'number' && Number.isSafeInteger(value) && value >= 0;
}

function optionalId(value: unknown): number | undefined {
  return validId(value) ? value : undefined;
}

function namedFacets(value: unknown): NamedFacet[] {
  if (!Array.isArray(value)) return [];
  const facets = new Map<number, string>();
  for (const entry of value as unknown[]) {
    if (typeof entry !== 'object' || entry === null) continue;
    const row = entry as Record<string, unknown>;
    if (validId(row.id) && typeof row.name === 'string' && row.name.trim()) {
      facets.set(row.id, row.name.trim());
    }
  }
  return [...facets].map(([id, name]) => ({ id, name }));
}

export function parseFeed(value: unknown): CatalogSnapshot {
  if (typeof value !== 'object' || value === null || !('pins' in value) ||
      !Array.isArray(value.pins)) throw new Error('Unexpected Pinpanion catalog format.');
  const feed = value as Record<string, unknown>;
  const pins: CatalogPin[] = [];
  const seen = new Set<number>();
  for (const row of feed.pins as unknown[]) {
    if (typeof row !== 'object' || row === null) continue;
    const pin = row as Record<string, unknown>;
    if (!validId(pin.id) || seen.has(pin.id) ||
        typeof pin.name !== 'string' || !pin.name.trim()) continue;
    seen.add(pin.id);
    const image = typeof pin.image_name === 'string' &&
      /^[^/\\?#]+$/.test(pin.image_name) && !['.', '..'].includes(pin.image_name)
      ? `https://pinpanion.com/imgs/${encodeURIComponent(pin.image_name)}` : undefined;
    const categories = Array.isArray(pin.categoryIds)
      ? [...new Set((pin.categoryIds as unknown[]).filter(validId))] : [];
    pins.push({
      pin_id: `pinpanion:${pin.id}`,
      name: pin.name.trim(),
      category_ids: categories,
      ...(image ? { image_url: image } : {}),
      ...(validId(pin.year) ? { year: pin.year } : {}),
      ...(optionalId(pin.setId) !== undefined ? { set_id: pin.setId as number } : {}),
      ...(optionalId(pin.paxEventId) !== undefined ? { event_id: pin.paxEventId as number } : {}),
      ...(optionalId(pin.groupId) !== undefined ? { group_id: pin.groupId as number } : {})
    });
  }
  return {
    pins,
    categories: namedFacets(feed.categories),
    sets: namedFacets(feed.sets),
    events: namedFacets(feed.events),
    groups: namedFacets(feed.groups)
  };
}

function matches(pin: CatalogPin, filters: CatalogFilters, except?: FacetKey): boolean {
  const needle = filters.query.trim().toLocaleLowerCase();
  if (needle && !pin.name.toLocaleLowerCase().includes(needle) &&
      !pin.pin_id.toLocaleLowerCase().includes(needle)) return false;
  if (except !== 'year' && filters.year !== null && pin.year !== filters.year) return false;
  if (except !== 'categoryId' && filters.categoryId !== null &&
      !pin.category_ids.includes(filters.categoryId)) return false;
  if (except !== 'setId' && filters.setId !== null && pin.set_id !== filters.setId) return false;
  if (except !== 'eventId' && filters.eventId !== null && pin.event_id !== filters.eventId) return false;
  if (except !== 'groupId' && filters.groupId !== null && pin.group_id !== filters.groupId) return false;
  return true;
}

export function filterCatalog(snapshot: CatalogSnapshot, filters: CatalogFilters): CatalogPin[] {
  return snapshot.pins.filter(pin => matches(pin, filters)).sort((a, b) =>
    a.name.localeCompare(b.name) || a.pin_id.localeCompare(b.pin_id));
}

function valuesFor(pin: CatalogPin, key: FacetKey): number[] {
  switch (key) {
    case 'year': return pin.year === undefined ? [] : [pin.year];
    case 'categoryId': return pin.category_ids;
    case 'setId': return pin.set_id === undefined ? [] : [pin.set_id];
    case 'eventId': return pin.event_id === undefined ? [] : [pin.event_id];
    case 'groupId': return pin.group_id === undefined ? [] : [pin.group_id];
  }
}

export function facetChoices(snapshot: CatalogSnapshot, filters: CatalogFilters,
                             key: FacetKey): FacetChoice[] {
  const labels = new Map<number, string>();
  const source = key === 'categoryId' ? snapshot.categories : key === 'setId' ? snapshot.sets :
    key === 'eventId' ? snapshot.events : key === 'groupId' ? snapshot.groups : [];
  for (const facet of source) labels.set(facet.id, facet.name);
  const counts = new Map<number, number>();
  for (const pin of snapshot.pins) {
    if (!matches(pin, filters, key)) continue;
    for (const id of valuesFor(pin, key)) counts.set(id, (counts.get(id) ?? 0) + 1);
  }
  const selected = filters[key];
  if (selected !== null && !counts.has(selected)) counts.set(selected, 0);
  const choices = [...counts].map(([id, count]) => ({
    id, count, label: key === 'year' ? String(id) : labels.get(id) ?? `#${id}`
  }));
  return choices.sort((a, b) => key === 'year' ? b.id - a.id :
    a.label.localeCompare(b.label) || a.id - b.id);
}

function openDatabase(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DATABASE_NAME, 1);
    request.onupgradeneeded = () => request.result.createObjectStore(STORE);
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

export async function loadCachedCatalog(): Promise<CatalogSnapshot> {
  const database = await openDatabase();
  try {
    return await new Promise<CatalogSnapshot>((resolve, reject) => {
      const transaction = database.transaction(STORE, 'readonly');
      const request = transaction.objectStore(STORE).get('pins');
      request.onsuccess = () => {
        const saved: unknown = request.result;
        if (Array.isArray(saved)) {
          // The first PWA version cached only IDs, names, and artwork URLs.
          resolve({ ...emptyCatalog(), pins: saved.map((pin: CatalogPin) =>
            ({ ...pin, category_ids: [] })) });
        } else if (typeof saved === 'object' && saved !== null && 'pins' in saved &&
                   Array.isArray(saved.pins)) {
          resolve(saved as CatalogSnapshot);
        } else resolve(emptyCatalog());
      };
      request.onerror = () => reject(request.error);
    });
  } finally { database.close(); }
}

async function saveCatalog(snapshot: CatalogSnapshot): Promise<void> {
  const database = await openDatabase();
  try {
    await new Promise<void>((resolve, reject) => {
      const transaction = database.transaction(STORE, 'readwrite');
      transaction.objectStore(STORE).put(snapshot, 'pins');
      transaction.oncomplete = () => resolve();
      transaction.onerror = () => reject(transaction.error);
    });
  } finally { database.close(); }
}

export async function refreshCatalog(): Promise<CatalogSnapshot> {
  const response = await fetch('/catalog.json', { cache: 'no-store' });
  if (!response.ok) throw new Error('Catalog update unavailable.');
  const snapshot = parseFeed(await response.json() as unknown);
  if (!snapshot.pins.length) throw new Error('Catalog update was empty.');
  await saveCatalog(snapshot);
  return snapshot;
}

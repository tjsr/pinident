export const STORAGE_KEY = 'pinident.mobile.library.v1';

export interface PinIdentity {
  pin_id: string;
  name: string;
  image_url?: string;
}

export interface LibraryItem extends PinIdentity {
  quantity: number;
  added_at: string;
}

export interface Collection {
  id: string;
  name: string;
  items: LibraryItem[];
}

export interface Library {
  version: 1;
  activeId: string;
  collections: Collection[];
}

export interface StorageLike {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
}

export function makeId(): string {
  if (globalThis.crypto?.randomUUID) return globalThis.crypto.randomUUID();
  if (globalThis.crypto?.getRandomValues) {
    const bytes = globalThis.crypto.getRandomValues(new Uint8Array(16));
    return [...bytes].map(byte => byte.toString(16).padStart(2, '0')).join('');
  }
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
}

export function emptyLibrary(): Library {
  const id = makeId();
  return { version: 1, activeId: id, collections: [{ id, name: 'My pins', items: [] }] };
}

function isLibrary(value: unknown): value is Library {
  if (typeof value !== 'object' || value === null) return false;
  const candidate = value as Partial<Library>;
  return candidate.version === 1 && typeof candidate.activeId === 'string' &&
    Array.isArray(candidate.collections) && candidate.collections.length > 0 &&
    candidate.collections.every((collection: unknown) => {
      if (typeof collection !== 'object' || collection === null) return false;
      const row = collection as Partial<Collection>;
      return typeof row.id === 'string' && typeof row.name === 'string' &&
        Array.isArray(row.items) && row.items.every((item: unknown) => {
          if (typeof item !== 'object' || item === null) return false;
          const pin = item as Partial<LibraryItem>;
          return typeof pin.pin_id === 'string' && typeof pin.name === 'string' &&
            Number.isInteger(pin.quantity) && (pin.quantity ?? 0) > 0 &&
            typeof pin.added_at === 'string';
        });
    });
}

export function loadLibrary(storage: StorageLike = localStorage): Library {
  try {
    const raw = storage.getItem(STORAGE_KEY);
    const value: unknown = raw ? JSON.parse(raw) : null;
    if (isLibrary(value)) return value;
  } catch { /* A damaged local copy must not stop scanning. */ }
  return emptyLibrary();
}

export function saveLibrary(state: Library, storage: StorageLike = localStorage): void {
  storage.setItem(STORAGE_KEY, JSON.stringify(state));
}

export function activeCollection(state: Library): Collection {
  return state.collections.find(collection => collection.id === state.activeId) ?? state.collections[0]!;
}

export function createCollection(state: Library, name: string): string {
  const trimmed = name.trim();
  if (!trimmed || trimmed.length > 80) throw new Error('Enter a collection name under 80 characters.');
  const id = makeId();
  state.collections.push({ id, name: trimmed, items: [] });
  state.activeId = id;
  return id;
}

export function addPin(state: Library, pin: PinIdentity): void {
  if (typeof pin.pin_id !== 'string' || !pin.pin_id ||
      typeof pin.name !== 'string' || !pin.name.trim()) throw new Error('A pin needs an ID and name.');
  const collection = activeCollection(state);
  const existing = collection.items.find(item => item.pin_id === pin.pin_id);
  if (existing) existing.quantity += 1;
  else collection.items.push({ pin_id: pin.pin_id, name: pin.name.trim(), quantity: 1,
    added_at: new Date().toISOString() });
}

export function changeQuantity(state: Library, pinId: string, change: number): void {
  const collection = activeCollection(state);
  const item = collection.items.find(row => row.pin_id === pinId);
  if (!item || !Number.isInteger(change)) return;
  item.quantity += change;
  if (item.quantity <= 0) collection.items.splice(collection.items.indexOf(item), 1);
}

export function exportLibrary(state: Library): string {
  return JSON.stringify(state, null, 2);
}

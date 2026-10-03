import assert from 'node:assert/strict';
import test from 'node:test';
import { activeCollection, addPin, changeQuantity, createCollection, emptyLibrary,
  exportLibrary, loadLibrary, makeId, saveLibrary, STORAGE_KEY,
  type StorageLike } from './dist/library.js';

test('pins accumulate in the selected collection and survive reload', () => {
  const data = new Map<string, string>();
  const storage: StorageLike = { getItem: key => data.get(key) ?? null,
    setItem: (key, value) => { data.set(key, value); } };
  const state = emptyLibrary();
  addPin(state, { pin_id: 'pinpanion:7', name: 'Test Pin' });
  addPin(state, { pin_id: 'pinpanion:7', name: 'Test Pin' });
  assert.equal(activeCollection(state).items[0]?.quantity, 2);
  createCollection(state, 'Spare pins');
  addPin(state, { pin_id: 'pinpanion:7', name: 'Test Pin' });
  assert.equal(activeCollection(state).items[0]?.quantity, 1);
  saveLibrary(state, storage);
  assert.equal(JSON.parse(data.get(STORAGE_KEY)!).collections.length, 2);
  assert.deepEqual(loadLibrary(storage), state);
  assert.deepEqual(JSON.parse(exportLibrary(state)), state);
});

test('quantities and invalid entries are handled explicitly', () => {
  const state = emptyLibrary();
  assert.throws(() => addPin(state, { pin_id: '', name: 'Unknown' }));
  assert.throws(() => createCollection(state, '   '));
  addPin(state, { pin_id: 'local:1', name: '  My custom pin  ' });
  assert.equal(activeCollection(state).items[0]?.name, 'My custom pin');
  changeQuantity(state, 'local:1', -1);
  assert.equal(activeCollection(state).items.length, 0);
  assert.match(makeId(), /^[A-Za-z0-9_-]{1,64}$/);
});

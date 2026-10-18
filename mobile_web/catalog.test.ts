import assert from 'node:assert/strict';
import test from 'node:test';
import { emptyFilters, facetChoices, filterCatalog, loadRecentPinIds,
  parseFeed, prioritizeRecents, rememberRecentPin } from './dist/catalog.js';

const feed = {
  categories: [{ id: 38, name: 'Core' }, { id: 16, name: 'Merch' }],
  sets: [{ id: 1, name: 'Starter Set' }],
  events: [{ id: 52, name: 'PAX East 2025' }],
  groups: [{ id: 9, name: 'Staff Heads' }],
  pins: [
    { id: 5, name: 'Pinny Arcade Logo', image_name: 'logo.webp', year: 2013,
      categoryIds: [38], setId: 1, paxEventId: null, groupId: null },
    { id: 8, name: 'Merch', image_name: 'merch.webp', year: 2013,
      categoryIds: [16, 38], setId: null, paxEventId: null, groupId: null },
    { id: 90, name: 'East Staff', image_name: 'east.webp', year: 2025,
      categoryIds: [38], setId: null, paxEventId: 52, groupId: 9 },
    { id: -1, name: 'Invalid' }
  ]
};

test('public feed metadata becomes named, searchable catalog facets', () => {
  const catalog = parseFeed(feed);
  assert.equal(catalog.pins.length, 3);
  assert.deepEqual(catalog.pins[0], {
    pin_id: 'pinpanion:5', name: 'Pinny Arcade Logo',
    image_url: 'https://pinpanion.com/imgs/logo.webp', year: 2013,
    category_ids: [38], set_id: 1
  });
  assert.deepEqual(facetChoices(catalog, emptyFilters(), 'year').map(choice => choice.label),
    ['2025', '2013']);
  assert.deepEqual(facetChoices(catalog, emptyFilters(), 'categoryId').map(choice =>
    [choice.label, choice.count]), [['Core', 3], ['Merch', 1]]);
  assert.throws(() => parseFeed({ pins: {} }));
});

test('combined filters narrow valid pins and facet lists without free-form names', () => {
  const catalog = parseFeed(feed);
  const filters = { ...emptyFilters(), year: 2013, categoryId: 38 };
  assert.deepEqual(filterCatalog(catalog, filters).map(pin => pin.pin_id),
    ['pinpanion:8', 'pinpanion:5']);
  assert.deepEqual(facetChoices(catalog, filters, 'setId').map(choice =>
    [choice.label, choice.count]), [['Starter Set', 1]]);
  filters.setId = 1;
  assert.deepEqual(filterCatalog(catalog, filters).map(pin => pin.pin_id), ['pinpanion:5']);
  filters.query = 'MERCH';
  assert.equal(filterCatalog(catalog, filters).length, 0);
  assert.deepEqual(facetChoices(catalog, filters, 'setId').map(choice =>
    [choice.label, choice.count]), [['Starter Set', 0]]);
});

test('recent catalog pins persist in newest-first order without duplicating remaining pins', () => {
  const values = new Map<string, string>();
  const storage = {
    getItem: (key: string) => values.get(key) ?? null,
    setItem: (key: string, value: string) => { values.set(key, value); }
  };
  let ids = loadRecentPinIds(storage);
  for (const id of ['pinpanion:5', 'pinpanion:8', 'pinpanion:90',
    'pinpanion:101', 'pinpanion:102', 'pinpanion:103', 'pinpanion:8'])
    ids = rememberRecentPin(ids, id, storage);
  assert.deepEqual(ids, ['pinpanion:8', 'pinpanion:103', 'pinpanion:102',
    'pinpanion:101', 'pinpanion:90']);
  assert.deepEqual(loadRecentPinIds(storage), ids);
  const pins = parseFeed(feed).pins;
  const grouped = prioritizeRecents(pins, ids, true);
  assert.deepEqual(grouped.recents.map(pin => pin.pin_id), ['pinpanion:8', 'pinpanion:90']);
  assert.deepEqual(grouped.remaining.map(pin => pin.pin_id), ['pinpanion:5']);
  assert.deepEqual(prioritizeRecents(pins, ids, false).recents, []);
});

import { activeCollection, addPin, changeQuantity, createCollection, exportLibrary,
         loadLibrary, makeId, saveLibrary, type PinIdentity } from './library.js';
import { emptyCatalog, emptyFilters, facetChoices, filterCatalog, loadCachedCatalog,
         refreshCatalog, type CatalogFilters, type CatalogPin, type CatalogSnapshot,
         type FacetKey } from './catalog.js';
import { rejectTrack, scanFrame, tagTrack, type ScanBox, type ScanState } from './scanner.js';

function $<T extends HTMLElement = HTMLElement>(id: string): T {
  const node = document.getElementById(id);
  if (!node) throw new Error(`Missing UI element: ${id}`);
  return node as T;
}
const state = loadLibrary();
let catalog: CatalogSnapshot = emptyCatalog();
let catalogFilters: CatalogFilters = emptyFilters();
let catalogLimit = 60;
let mediaStream: MediaStream | null = null;
let objectUrl: string | null = null;
let source: HTMLVideoElement | HTMLImageElement | null = null;
let scanTimer: number | null = null;
let scanSession = makeId();
let tagTarget: ScanBox | null = null;
let lastResult: ScanState | null = null;
let lastSnapshot: HTMLCanvasElement | null = null;
let toastTimer: number | null = null;
const scanCanvas = document.createElement('canvas');
const facetFields: ReadonlyArray<{ id: string; key: FacetKey; all: string }> = [
  { id: 'filterYear', key: 'year', all: 'All years' },
  { id: 'filterCategory', key: 'categoryId', all: 'All categories' },
  { id: 'filterSet', key: 'setId', all: 'All sets' },
  { id: 'filterEvent', key: 'eventId', all: 'All events' },
  { id: 'filterGroup', key: 'groupId', all: 'All groups' }
];

function errorText(error: unknown): string { return error instanceof Error ? error.message : String(error); }
function setStatus(message: string): void { $('scanStatus').textContent = message; }
function toast(message: string): void {
  const element = $('toast');
  element.textContent = message;
  element.hidden = false;
  if (toastTimer !== null) clearTimeout(toastTimer);
  toastTimer = window.setTimeout(() => { element.hidden = true; }, 3000);
}
function element<K extends keyof HTMLElementTagNameMap>(tag: K, className = '', text?: string): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text != null) node.textContent = text;
  return node;
}
function showTab(name: 'scan' | 'library'): void {
  const library = name === 'library';
  $('scanView').hidden = library;
  $('libraryView').hidden = !library;
  $('libraryTab').classList.toggle('active', library);
  $('scanTab').classList.toggle('active', !library);
  $('libraryTab').setAttribute('aria-current', library ? 'page' : 'false');
  $('scanTab').setAttribute('aria-current', library ? 'false' : 'page');
  if (library) renderLibrary();
}
function persist(): void { saveLibrary(state); renderCollectionSelect(); renderLibrary(); }
function renderCollectionSelect(): void {
  const select = $<HTMLSelectElement>('collectionSelect');
  select.replaceChildren();
  for (const collection of state.collections) {
    const option = element('option', '', collection.name);
    option.value = collection.id;
    select.append(option);
  }
  select.value = activeCollection(state).id;
  const count = state.collections.reduce((total, collection) =>
    total + collection.items.reduce((sum, item) => sum + item.quantity, 0), 0);
  $('libraryCount').textContent = String(count);
}
function renderLibrary(): void {
  const target = $('libraryItems');
  target.replaceChildren();
  const collection = activeCollection(state);
  if (!collection.items.length) {
    target.append(element('p', 'empty-message', 'This collection is empty. Scan a pin or add one from the catalog.'));
    return;
  }
  for (const item of [...collection.items].sort((a, b) => a.name.localeCompare(b.name))) {
    const card = element('article', 'pin-card');
    const detail = element('div', 'pin-details');
    detail.append(element('strong', '', item.name), element('small', '', item.pin_id));
    const quantity = element('div', 'quantity');
    const minus = element('button', '', '−');
    minus.type = 'button'; minus.setAttribute('aria-label', `Remove one ${item.name}`);
    minus.addEventListener('click', () => { changeQuantity(state, item.pin_id, -1); persist(); });
    const count = element('span', '', String(item.quantity));
    const plus = element('button', '', '+');
    plus.type = 'button'; plus.setAttribute('aria-label', `Add one ${item.name}`);
    plus.addEventListener('click', () => { changeQuantity(state, item.pin_id, 1); persist(); });
    quantity.append(minus, count, plus);
    card.append(detail, quantity);
    target.append(card);
  }
}
function savePin(pin: PinIdentity): void {
  try {
    addPin(state, pin);
    persist();
    toast(`${pin.name} added to ${activeCollection(state).name}`);
    if ($<HTMLDialogElement>('tagDialog').open) $<HTMLDialogElement>('tagDialog').close();
  } catch (error) { toast(errorText(error)); }
}
function choosePin(pin: PinIdentity, box: ScanBox | null = tagTarget): void {
  if (box && lastResult && tagTrack(lastResult, box.track_id, pin)) {
    drawOverlay(lastResult);
    if (lastSnapshot) renderResults(lastResult, lastSnapshot);
  }
  tagTarget = null;
  savePin(pin);
}
function openSearch(target: ScanBox | null = null): void {
  tagTarget = target;
  stopLoop();
  if (source === $<HTMLVideoElement>('videoPreview') && !mediaStream) source.pause();
  catalogFilters = emptyFilters();
  catalogLimit = 60;
  $<HTMLInputElement>('catalogQuery').value = '';
  $<HTMLInputElement>('customPinName').value = '';
  renderCatalogPicker();
  $<HTMLDialogElement>('tagDialog').showModal();
}
function renderCatalogPicker(): void {
  const count = catalog.pins.length;
  $('catalogStatus').textContent = count
    ? `${count.toLocaleString()} known pins available to select${navigator.onLine ? '' : ' offline'}.`
    : 'No Pinpanion list is cached yet. Connect once to load known pins, then they remain selectable offline.';
  for (const field of facetFields) {
    const select = $<HTMLSelectElement>(field.id);
    select.replaceChildren();
    const all = element('option', '', field.all);
    all.value = '';
    select.append(all);
    for (const choice of facetChoices(catalog, catalogFilters, field.key)) {
      const option = element('option', '', `${choice.label} (${choice.count})`);
      option.value = String(choice.id);
      select.append(option);
    }
    select.value = catalogFilters[field.key] === null ? '' : String(catalogFilters[field.key]);
    select.disabled = count === 0;
  }
  renderCatalogResults();
}
function renderCatalogResults(): void {
  const pins = filterCatalog(catalog, catalogFilters);
  const target = $('catalogResults');
  target.replaceChildren();
  $('catalogCount').textContent = pins.length
    ? `Showing ${Math.min(catalogLimit, pins.length).toLocaleString()} of ${pins.length.toLocaleString()} pins`
    : 'No pins match these filters.';
  if (!pins.length) target.append(element('p', 'empty-message', catalog.pins.length
    ? 'No known pins match. Change or clear a filter.'
    : 'The known pin list is unavailable. Try again when connected.'));
  for (const pin of pins.slice(0, catalogLimit)) {
      const button = element('button', 'catalog-item');
      button.type = 'button';
      if (pin.image_url && navigator.onLine) {
        const image = element('img');
        image.src = pin.image_url;
        image.alt = '';
        image.loading = 'lazy';
        button.append(image);
      }
      const text = element('span');
      const details = pin.year === undefined ? [pin.pin_id] : [String(pin.year), pin.pin_id];
      if (pin.set_id !== undefined) details.push(catalog.sets.find(set => set.id === pin.set_id)?.name ?? `Set #${pin.set_id}`);
      if (pin.event_id !== undefined) details.push(catalog.events.find(event => event.id === pin.event_id)?.name ?? `Event #${pin.event_id}`);
      text.append(element('strong', '', pin.name), element('small', '', details.join(' · ')));
      button.append(text);
      button.addEventListener('click', () => choosePin(pin));
      target.append(button);
  }
  $('showMorePins').hidden = pins.length <= catalogLimit;
  $('clearCatalogFilters').hidden = !catalogFilters.query &&
    facetFields.every(field => catalogFilters[field.key] === null);
}
function searchPins(): void {
  catalogFilters.query = $<HTMLInputElement>('catalogQuery').value.trim();
  catalogLimit = 60;
  renderCatalogPicker();
}
function stopLoop(): void { if (scanTimer !== null) clearInterval(scanTimer); scanTimer = null; }
function stopSource(): void {
  stopLoop();
  scanSession = makeId();
  lastResult = null; lastSnapshot = null;
  if (mediaStream) { for (const track of mediaStream.getTracks()) track.stop(); mediaStream = null; }
  const video = $<HTMLVideoElement>('videoPreview');
  video.pause(); video.srcObject = null; video.removeAttribute('src'); video.load();
  if (objectUrl) { URL.revokeObjectURL(objectUrl); objectUrl = null; }
  $<HTMLImageElement>('photoPreview').removeAttribute('src');
  video.hidden = true; $('photoPreview').hidden = true; $('stageEmpty').hidden = false;
  $('stopButton').hidden = true; source = null;
  clearOverlay();
}
function showSource(kind: 'camera' | 'video' | 'photo'): void {
  $('stageEmpty').hidden = true;
  $('videoPreview').hidden = kind === 'photo';
  $('photoPreview').hidden = kind !== 'photo';
  $('stopButton').hidden = false;
  scanSession = makeId();
  $('results').replaceChildren(element('p', 'empty-message', 'Scanning…'));
}
function startLoop(): void {
  stopLoop();
  scanOne();
  scanTimer = window.setInterval(scanOne, 850);
}
async function startCamera(): Promise<void> {
  stopSource();
  if (!navigator.mediaDevices?.getUserMedia) {
    setStatus('Live camera needs HTTPS. Use Take or choose photo, or open a video.');
    return;
  }
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: { ideal: 'environment' } }, audio: false });
    const video = $<HTMLVideoElement>('videoPreview');
    video.controls = false;
    video.srcObject = mediaStream;
    source = video;
    showSource('camera');
    await video.play();
    setStatus('Scanning on this device. Move slowly and review suggested pin areas.');
    startLoop();
  } catch (error) { stopSource(); setStatus(`Camera unavailable: ${errorText(error)}. Try a photo instead.`); }
}
function loadVideo(file?: File): void {
  if (!file) return;
  stopSource();
  const video = $<HTMLVideoElement>('videoPreview');
  objectUrl = URL.createObjectURL(file);
  video.controls = true;
  video.src = objectUrl;
  source = video;
  showSource('video');
  setStatus('Press play on the video. Candidate areas are scanned on this device as it runs.');
  void video.play().catch(() => {});
}
function loadPhoto(file?: File): void {
  if (!file) return;
  stopSource();
  const image = $<HTMLImageElement>('photoPreview');
  objectUrl = URL.createObjectURL(file);
  image.onload = () => { source = image; showSource('photo'); scanOne(); };
  image.src = objectUrl;
  setStatus('Scanning photo on this device…');
}
function capture(): { frame: ImageData; snapshot: HTMLCanvasElement } | null {
  if (!source) return null;
  const width = source instanceof HTMLVideoElement ? source.videoWidth : source.naturalWidth;
  const height = source instanceof HTMLVideoElement ? source.videoHeight : source.naturalHeight;
  if (!width || !height) return null;
  const scale = Math.min(1, 420 / Math.max(width, height));
  scanCanvas.width = Math.round(width * scale);
  scanCanvas.height = Math.round(height * scale);
  const context = scanCanvas.getContext('2d', { willReadFrequently: true });
  if (!context) throw new Error('Canvas is unavailable.');
  context.drawImage(source, 0, 0, scanCanvas.width, scanCanvas.height);
  const snapshot = document.createElement('canvas');
  snapshot.width = scanCanvas.width; snapshot.height = scanCanvas.height;
  snapshot.getContext('2d')?.drawImage(scanCanvas, 0, 0);
  return { frame: context.getImageData(0, 0, scanCanvas.width, scanCanvas.height), snapshot };
}
function clearOverlay(): void {
  const canvas = $<HTMLCanvasElement>('boxOverlay');
  canvas.getContext('2d')?.clearRect(0, 0, canvas.width, canvas.height);
}
function drawOverlay(result: ScanState): void {
  const canvas = $<HTMLCanvasElement>('boxOverlay');
  const rect = $('stage').getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.round(rect.width * ratio);
  canvas.height = Math.round(rect.height * ratio);
  const context = canvas.getContext('2d');
  if (!context) return;
  context.scale(ratio, ratio);
  const fit = Math.min(rect.width / result.width, rect.height / result.height);
  const left = (rect.width - result.width * fit) / 2;
  const top = (rect.height - result.height * fit) / 2;
  context.lineWidth = 2;
  context.font = 'bold 13px system-ui';
  for (const box of result.boxes) {
    if (box.review_state === 'rejected') continue;
    const [x, y, width, height] = box.coords;
    const colour = box.review_state === 'confirmed' ? '#39b86b' :
      box.review_state === 'inherited' ? '#ff9d38' : box.pin_id ? '#ef5558' : '#5aa7ff';
    context.strokeStyle = colour;
    context.strokeRect(left + x * fit, top + y * fit, width * fit, height * fit);
  }
}
function cropThumbnail(frame: HTMLCanvasElement, coords: ScanBox['coords']): string {
  const [x, y, width, height] = coords;
  if (width < 1 || height < 1) return '';
  const canvas = document.createElement('canvas');
  const scale = Math.min(1, 100 / Math.max(width, height));
  canvas.width = Math.max(1, Math.round(width * scale));
  canvas.height = Math.max(1, Math.round(height * scale));
  canvas.getContext('2d')?.drawImage(frame, x, y, width, height, 0, 0, canvas.width, canvas.height);
  return canvas.toDataURL('image/jpeg', 0.75);
}
function renderResults(result: ScanState, snapshot: HTMLCanvasElement): void {
  const target = $('results');
  target.replaceChildren();
  const visible = result.boxes.filter(box => box.review_state !== 'rejected');
  if (!visible.length) {
    target.append(element('p', 'empty-message', 'No pin areas detected in this frame. Try a clearer angle or search the catalog.'));
    return;
  }
  for (const box of visible.slice(0, 40)) {
    const card = element('article', 'pin-card');
    const thumb = element('img', 'pin-thumb');
    thumb.alt = 'Detected pin area';
    thumb.src = cropThumbnail(snapshot, box.coords);
    const detail = element('div', 'pin-details');
    detail.append(element('strong', '', box.name || 'Possible pin'),
      element('small', '', box.pin_id || 'Identity unknown'));
    const actions = element('div', 'card-actions');
    const button = element('button', '', box.pin_id ? 'Add pin' : 'Tag pin');
    button.type = 'button';
    button.addEventListener('click', () => {
      if (box.pin_id && box.name) choosePin({ pin_id: box.pin_id, name: box.name }, box);
      else openSearch(box);
    });
    const reject = element('button', 'reject-button', 'Reject');
    reject.type = 'button';
    reject.addEventListener('click', () => {
      if (lastResult && rejectTrack(lastResult, box.track_id)) {
        drawOverlay(lastResult);
        renderResults(lastResult, snapshot);
        toast('Area rejected for this scan.');
      }
    });
    actions.append(button, reject);
    card.append(thumb, detail, actions);
    target.append(card);
  }
}
function scanOne(): void {
  if (!source || (source instanceof HTMLVideoElement && source.paused)) return;
  const currentSession = scanSession;
  try {
    const captured = capture();
    if (!captured) return;
    if (currentSession !== scanSession) return;
    lastResult = scanFrame(captured.frame, lastResult);
    lastSnapshot = captured.snapshot;
    drawOverlay(lastResult);
    renderResults(lastResult, captured.snapshot);
    const count = lastResult.boxes.filter(box => box.review_state !== 'rejected').length;
    setStatus(`${count} area${count === 1 ? '' : 's'} in this frame. Review before adding pins.`);
  } catch (error) { setStatus(`Scan paused: ${errorText(error)}`); stopLoop(); }
}
async function initializeCatalog(): Promise<void> {
  try { catalog = await loadCachedCatalog(); } catch { /* Scanning remains available. */ }
  const indicator = $('connectionStatus');
  indicator.textContent = catalog.pins.length ? `${catalog.pins.length} pins cached` : 'Catalog not cached';
  indicator.classList.toggle('offline', !catalog.pins.length);
  if ($<HTMLDialogElement>('tagDialog').open) renderCatalogPicker();
  if (!navigator.onLine) return;
  await updateCatalog();
}
let catalogRefreshing = false;
async function updateCatalog(): Promise<void> {
  if (catalogRefreshing) return;
  if (!navigator.onLine) { toast('Connect to update the known pin list.'); return; }
  catalogRefreshing = true;
  try {
    catalog = await refreshCatalog();
    const indicator = $('connectionStatus');
    indicator.textContent = `${catalog.pins.length} pins cached offline`;
    indicator.classList.remove('offline');
    if ($<HTMLDialogElement>('tagDialog').open) renderCatalogPicker();
  } catch {
    if ($<HTMLDialogElement>('tagDialog').open && !catalog.pins.length) {
      $('catalogStatus').textContent = 'Could not load the known pin list. Retry while connected.';
    }
  } finally { catalogRefreshing = false; }
}

$('scanTab').addEventListener('click', () => showTab('scan'));
$('libraryTab').addEventListener('click', () => showTab('library'));
$('cameraButton').addEventListener('click', () => { void startCamera(); });
$('stopButton').addEventListener('click', () => { stopSource(); setStatus('Scanning stopped.'); });
$<HTMLInputElement>('photoInput').addEventListener('change', event => {
  const target = event.currentTarget as HTMLInputElement;
  loadPhoto(target.files?.[0]); target.value = '';
});
$<HTMLInputElement>('videoInput').addEventListener('change', event => {
  const target = event.currentTarget as HTMLInputElement;
  loadVideo(target.files?.[0]); target.value = '';
});
$<HTMLVideoElement>('videoPreview').addEventListener('play', () => {
  if (source === $<HTMLVideoElement>('videoPreview') && !mediaStream) startLoop();
});
$<HTMLVideoElement>('videoPreview').addEventListener('pause', () => { if (!mediaStream) stopLoop(); });
$<HTMLVideoElement>('videoPreview').addEventListener('ended', stopLoop);
$('manualSearchButton').addEventListener('click', () => openSearch());
$('addCatalogPinButton').addEventListener('click', () => openSearch());
$<HTMLInputElement>('catalogQuery').addEventListener('input', searchPins);
$('retryCatalogButton').addEventListener('click', () => { void updateCatalog(); });
for (const field of facetFields) {
  $<HTMLSelectElement>(field.id).addEventListener('change', event => {
    const value = (event.currentTarget as HTMLSelectElement).value;
    catalogFilters[field.key] = value === '' ? null : Number(value);
    catalogLimit = 60;
    renderCatalogPicker();
  });
}
$('clearCatalogFilters').addEventListener('click', () => {
  catalogFilters = emptyFilters();
  catalogLimit = 60;
  $<HTMLInputElement>('catalogQuery').value = '';
  renderCatalogPicker();
});
$('showMorePins').addEventListener('click', () => {
  catalogLimit += 60;
  renderCatalogResults();
});
$('saveCustomButton').addEventListener('click', () => {
  const name = $<HTMLInputElement>('customPinName').value.trim();
  if (!name) { toast('Enter a name for this pin.'); return; }
  choosePin({ pin_id: `local:${makeId()}`, name });
});
$<HTMLDialogElement>('tagDialog').addEventListener('close', () => {
  tagTarget = null;
  if (mediaStream && source === $<HTMLVideoElement>('videoPreview')) startLoop();
});
$<HTMLSelectElement>('collectionSelect').addEventListener('change', event => {
  state.activeId = (event.currentTarget as HTMLSelectElement).value; persist();
});
$('newCollectionButton').addEventListener('click', () => {
  const name = prompt('Name this collection');
  if (name === null) return;
  try { createCollection(state, name); persist(); } catch (error) { toast(errorText(error)); }
});
$('exportButton').addEventListener('click', () => {
  const blob = new Blob([exportLibrary(state)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const link = element('a');
  link.href = url; link.download = 'pinident-library.json';
  document.body.append(link); link.click(); link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 5000);
});
window.addEventListener('resize', () => { if (lastResult) drawOverlay(lastResult); });
window.addEventListener('online', () => { void initializeCatalog(); });
renderCollectionSelect(); renderLibrary(); void initializeCatalog();
if ('serviceWorker' in navigator && window.isSecureContext) {
  void navigator.serviceWorker.register('/service-worker.js').catch(() => {});
}

import { activeCollection, addPin, changeQuantity, createCollection, exportLibrary,
         loadLibrary, makeId, saveLibrary, type PinIdentity } from './library.js';
import { emptyCatalog, emptyFilters, facetChoices, filterCatalog, loadCachedCatalog,
         loadRecentPinIds, prioritizeRecents, refreshCatalog, rememberRecentPin,
         type CatalogFilters, type CatalogPin, type CatalogSnapshot,
         type FacetKey } from './catalog.js';
import { addManualBox, areaAtPoint, confirmUnknown, deleteAreas, rejectTrack,
         resolveAreaDrag, scanFrame, tagTrack,
         type AreaDragMode, type PixelFrame, type ScanBox, type ScanDiagnostics, type ScanState } from './scanner.js';
import { clampDrawnBox, loadRecords, makeFrameAnnotation,
         makeDifficultyMark, makeTrainingReview, difficultyPriority,
         loadSharedDifficultyHints, saveRecord, syncPendingReviews,
         type DifficultyHint, type FrameAnnotation, type ReviewRecord } from './reviews.js';
import { loadMobileModel, type MobileModel } from './model.js';
import { combineIdentityIndex, learnNegative, learnPositive, loadIdentityIndex,
         loadLearnedExamples, referenceFeature, saveLearnedExamples,
         type IdentityIndex } from './identity.js';
import { completeScan, scanControls } from './scan_session.js';
import { containFrame, framePoint } from './stage_geometry.js';

function $<T extends HTMLElement = HTMLElement>(id: string): T {
  const node = document.getElementById(id);
  if (!node) throw new Error(`Missing UI element: ${id}`);
  return node as T;
}
const state = loadLibrary();
let catalog: CatalogSnapshot = emptyCatalog();
let recentPinIds = loadRecentPinIds();
let detectionModel: MobileModel | null = null;
const learnedExamples = loadLearnedExamples();
let catalogIdentityIndex: IdentityIndex | null = null;
let identityIndex: IdentityIndex | null = combineIdentityIndex(null, learnedExamples);
let identityLoadRevision = 0;
let modelLoadRevision = 0;
let catalogFilters: CatalogFilters = emptyFilters();
let catalogLimit = 60;
let mediaStream: MediaStream | null = null;
let objectUrl: string | null = null;
let source: HTMLVideoElement | HTMLImageElement | null = null;
let scanTimer: number | null = null;
let scanSession = makeId();
let scanSequence = 0;
let lastDetectionLogAt = 0;
let tagTarget: ScanBox | null = null;
let pickerAddsToLibrary = false;
let lastResult: ScanState | null = null;
let lastSnapshot: HTMLCanvasElement | null = null;
let lastFrame: PixelFrame | null = null;
let lastFrameTime: number | null = null;
let savedFrameRecord: FrameAnnotation | null = null;
let frameSaving = false;
let finishingScan = false;
let sourceName = '';
let frozen = false;
let areaDragMode: AreaDragMode = 'auto';
const selectedTrackIds = new Set<string>();
let draftStart: [number, number] | null = null;
let draftEnd: [number, number] | null = null;
let draftStartClient: [number, number] | null = null;
let records: ReviewRecord[] = [];
type TrainingCandidate = { box: ScanBox; image: string; source: string; time: number | null;
  frame: HTMLCanvasElement };
const trainingQueue: TrainingCandidate[] = [];
const queuedCandidates = new Set<string>();
let sharedDifficultyHints: DifficultyHint[] = [];
let trainingQueueLoading = false;
let classificationSaving = false;
let toastTimer: number | null = null;
const scanCanvas = document.createElement('canvas');
const facetFields: ReadonlyArray<{ id: string; key: FacetKey; all: string }> = [
  { id: 'filterYear', key: 'year', all: 'All years' },
  { id: 'filterCategory', key: 'categoryId', all: 'All categories' },
  { id: 'filterSet', key: 'setId', all: 'All sets' },
  { id: 'filterEvent', key: 'eventId', all: 'All events' },
  { id: 'filterGroup', key: 'groupId', all: 'All groups' }
];

function errorText(error: Error | string): string {
  return error instanceof Error ? error.message : String(error);
}
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
function showTab(name: 'scan' | 'training' | 'library'): void {
  const library = name === 'library';
  $('scanView').hidden = name !== 'scan';
  $('trainingView').hidden = name !== 'training';
  $('libraryView').hidden = !library;
  $('libraryTab').classList.toggle('active', library);
  $('trainingTab').classList.toggle('active', name === 'training');
  $('scanTab').classList.toggle('active', name === 'scan');
  $('libraryTab').setAttribute('aria-current', library ? 'page' : 'false');
  $('trainingTab').setAttribute('aria-current', name === 'training' ? 'page' : 'false');
  $('scanTab').setAttribute('aria-current', name === 'scan' ? 'page' : 'false');
  if (library) renderLibrary();
  if (name === 'training') renderTraining();
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
function rememberPinSelection(pin: PinIdentity): void {
  if (catalog.pins.some(candidate => candidate.pin_id === pin.pin_id))
    recentPinIds = rememberRecentPin(recentPinIds, pin.pin_id);
}
function savePin(pin: PinIdentity): void {
  try {
    addPin(state, pin);
    rememberPinSelection(pin);
    persist();
    toast(`${pin.name} added to ${activeCollection(state).name}`);
    if ($<HTMLDialogElement>('tagDialog').open) $<HTMLDialogElement>('tagDialog').close();
  } catch (error) { toast(errorText(error instanceof Error ? error : String(error))); }
}
function choosePin(pin: PinIdentity, box: ScanBox | null = tagTarget): void {
  if (box && (!lastResult || !tagTrack(lastResult, box.track_id, pin))) {
    toast('That area changed. Select it again on the paused frame.');
    return;
  }
  if (box && lastResult) {
    if (lastFrame) {
      learnPositive(learnedExamples, pin.pin_id, pin.name,
        referenceFeature(lastFrame, box.coords));
      saveLearnedExamples(learnedExamples);
      identityIndex = combineIdentityIndex(catalogIdentityIndex, learnedExamples);
    }
    drawOverlay(lastResult);
    if (lastSnapshot) renderResults(lastResult, lastSnapshot);
  }
  tagTarget = null;
  if (pickerAddsToLibrary) savePin(pin);
  else {
    rememberPinSelection(pin);
    toast(`${pin.name} tagged on this frame.`);
    $<HTMLDialogElement>('tagDialog').close();
  }
}
function openSearch(target: ScanBox | null = null, addToLibrary = false): void {
  if (target && !frozen && source instanceof HTMLVideoElement) freezeCurrent();
  else stopLoop();
  tagTarget = target ? lastResult?.boxes.find(box => box.track_id === target.track_id) ?? null : null;
  if (target && !tagTarget) { toast('That area changed. Select it again on the paused frame.'); return; }
  pickerAddsToLibrary = addToLibrary;
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
  const { recents, remaining } = prioritizeRecents(pins, recentPinIds, !catalogFilters.query.trim());
  const visibleCount = recents.length + Math.min(catalogLimit, remaining.length);
  const target = $('catalogResults');
  target.replaceChildren();
  $('catalogCount').textContent = pins.length
    ? `Showing ${visibleCount.toLocaleString()} of ${pins.length.toLocaleString()} pins`
    : 'No pins match these filters.';
  if (!pins.length) target.append(element('p', 'empty-message', catalog.pins.length
    ? 'No known pins match. Change or clear a filter.'
    : 'The known pin list is unavailable. Try again when connected.'));
  const appendPin = (pin: CatalogPin): void => {
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
  };
  if (recents.length) {
    target.append(element('h3', 'catalog-section-heading', 'Recents'));
    recents.forEach(appendPin);
    if (remaining.length) {
      target.append(element('hr', 'catalog-divider'));
      target.append(element('h3', 'catalog-section-heading', 'All other pins'));
    }
  }
  remaining.slice(0, catalogLimit).forEach(appendPin);
  $('showMorePins').hidden = remaining.length <= catalogLimit;
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
  lastResult = null; lastSnapshot = null; lastFrame = null; lastFrameTime = null;
  savedFrameRecord = null; frozen = false;
  sourceName = ''; draftStart = null; draftEnd = null; draftStartClient = null;
  selectedTrackIds.clear(); areaDragMode = 'auto';
  queuedCandidates.clear();
  if (mediaStream) { for (const track of mediaStream.getTracks()) track.stop(); mediaStream = null; }
  const video = $<HTMLVideoElement>('videoPreview');
  video.pause(); video.srcObject = null; video.removeAttribute('src'); video.load();
  if (objectUrl) { URL.revokeObjectURL(objectUrl); objectUrl = null; }
  $<HTMLImageElement>('photoPreview').removeAttribute('src');
  video.hidden = true; $('photoPreview').hidden = true; $('stageEmpty').hidden = false;
  $('stage').style.aspectRatio = '';
  $('stopButton').hidden = true; source = null;
  $('results').replaceChildren(element('p', 'empty-message',
    'Detected pins will appear here. You can also browse the catalog.'));
  clearOverlay();
  updateAnnotationControls();
}
function showSource(kind: 'camera' | 'video' | 'photo'): void {
  $('stageEmpty').hidden = true;
  $('videoPreview').hidden = kind === 'photo';
  $('photoPreview').hidden = kind !== 'photo';
  $('stopButton').hidden = false;
  scanSession = makeId();
  $('results').replaceChildren(element('p', 'empty-message', 'Scanning…'));
  updateAnnotationControls();
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
    sourceName = 'Live camera';
    showSource('camera');
    await video.play();
    setStatus('Scanning on this device. Move slowly and review suggested pin areas.');
    startLoop();
  } catch (error) { stopSource(); setStatus(`Camera unavailable: ${errorText(error instanceof Error ? error : String(error))}. Try a photo instead.`); }
}
function loadVideo(file?: File): void {
  if (!file) return;
  stopSource();
  const video = $<HTMLVideoElement>('videoPreview');
  objectUrl = URL.createObjectURL(file);
  video.controls = true;
  video.src = objectUrl;
  source = video;
  sourceName = file.name;
  showSource('video');
  setStatus('Press play on the video. Candidate areas are scanned on this device as it runs.');
  void video.play().catch(() => {});
}
function loadPhoto(file?: File): void {
  if (!file) return;
  stopSource();
  const image = $<HTMLImageElement>('photoPreview');
  objectUrl = URL.createObjectURL(file);
  sourceName = file.name;
  image.onload = () => { source = image; frozen = true; showSource('photo'); scanOne(true); updateAnnotationControls(); };
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
function refreshAreaSelection(): void {
  if (lastResult) {
    const visible = new Set(lastResult.boxes.filter(box => box.review_state !== 'rejected')
      .map(box => box.track_id));
    for (const id of selectedTrackIds) if (!visible.has(id)) selectedTrackIds.delete(id);
  } else selectedTrackIds.clear();
  updateAnnotationControls();
  if (lastResult) drawOverlay(lastResult);
  if (lastResult && lastSnapshot) renderResults(lastResult, lastSnapshot);
}
function toggleAreaSelection(trackId: string): void {
  if (!lastResult?.boxes.some(box => box.track_id === trackId && box.review_state !== 'rejected')) return;
  if (selectedTrackIds.has(trackId)) selectedTrackIds.delete(trackId);
  else selectedTrackIds.add(trackId);
  refreshAreaSelection();
}
function deleteSelectedAreas(): void {
  if (!frozen || !lastResult || !selectedTrackIds.size) return;
  for (let index = trainingQueue.length - 1; index >= 0; index--) {
    const item = trainingQueue[index]!;
    if (item.frame === lastSnapshot && selectedTrackIds.has(item.box.track_id))
      trainingQueue.splice(index, 1);
  }
  const removed = deleteAreas(lastResult, selectedTrackIds);
  selectedTrackIds.clear();
  refreshAreaSelection();
  renderTraining();
  if (removed) {
    const remaining = lastResult.boxes.filter(box => box.review_state !== 'rejected').length;
    setStatus(`${remaining} area${remaining === 1 ? '' : 's'} in this frame. Save the annotation to keep this change.`);
    toast(`${removed} area${removed === 1 ? '' : 's'} deleted. Save the frame to keep this change.`);
  }
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
  const placement = containFrame(rect.width, rect.height, result.width, result.height);
  if (!placement) return;
  const { fit, left, top } = placement;
  context.lineWidth = 2;
  context.font = 'bold 13px system-ui';
  for (const box of result.boxes) {
    if (box.review_state === 'rejected') continue;
    const [x, y, width, height] = box.coords;
    const colour = box.review_state === 'confirmed' ? '#39b86b' :
      box.review_state === 'inherited' ? '#ff9d38' : box.pin_id ? '#ef5558' : '#5aa7ff';
    context.strokeStyle = colour;
    context.strokeRect(left + x * fit, top + y * fit, width * fit, height * fit);
    const label = box.name || box.suggested_name;
    const strength = box.match_strength;
    if (label) {
      const title = strength !== null && strength !== undefined && !box.name
        ? `${Math.round(strength * 100)}% match · ${label}` : label;
      const short = title.length > 36 ? `${title.slice(0, 33)}…` : title;
      const labelWidth = Math.min(rect.width - 4, context.measureText(short).width + 10);
      const labelX = Math.max(2, Math.min(rect.width - labelWidth - 2, left + x * fit));
      const labelY = top + y * fit >= 21 ? top + y * fit - 20 : top + y * fit + 2;
      context.fillStyle = '#102439df';
      context.fillRect(labelX, labelY, labelWidth, 19);
      context.fillStyle = '#fff';
      context.fillText(short, labelX + 5, labelY + 14, labelWidth - 10);
    }
    if (selectedTrackIds.has(box.track_id)) {
      context.save();
      context.fillStyle = '#ffd86b35';
      context.fillRect(left + x * fit, top + y * fit, width * fit, height * fit);
      context.lineWidth = 4;
      context.strokeStyle = '#ffd86b';
      context.strokeRect(left + x * fit, top + y * fit, width * fit, height * fit);
      context.restore();
    }
  }
  if (draftStart && draftEnd) {
    const coords = clampDrawnBox(draftStart, draftEnd, result.width, result.height);
    if (coords) {
      context.setLineDash([6, 4]);
      context.strokeStyle = resolveAreaDrag(result, coords, areaDragMode).action === 'select'
        ? '#ffd86b' : '#fff';
      context.strokeRect(left + coords[0] * fit, top + coords[1] * fit,
        coords[2] * fit, coords[3] * fit);
    }
  }
}
function cropThumbnail(frame: HTMLCanvasElement, coords: ScanBox['coords'], maxSize = 100): string {
  const [x, y, width, height] = coords;
  if (width < 1 || height < 1) return '';
  const canvas = document.createElement('canvas');
  const scale = Math.min(1, maxSize / Math.max(width, height));
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
    card.classList.toggle('selected-area', selectedTrackIds.has(box.track_id));
    const thumb = element('img', 'pin-thumb');
    thumb.alt = 'Detected pin area';
    thumb.src = cropThumbnail(snapshot, box.coords);
    const detail = element('div', 'pin-details');
    detail.append(element('strong', '', box.name || box.suggested_name || 'Possible pin'),
      element('small', '', box.name ? 'Tagged pin' :
        box.suggested_name && box.match_strength !== null &&
        box.match_strength !== undefined
          ? `${Math.round(box.match_strength * 100)}% visual match · provisional`
          : 'Identity unknown'));
    const actions = element('div', 'card-actions');
    const select = element('button', '', selectedTrackIds.has(box.track_id) ? 'Deselect' : 'Select');
    select.type = 'button';
    select.hidden = !frozen;
    select.setAttribute('aria-pressed', String(selectedTrackIds.has(box.track_id)));
    select.addEventListener('click', () => toggleAreaSelection(box.track_id));
    const button = element('button', '', 'Tag pin');
    button.type = 'button';
    button.addEventListener('click', () => openSearch(box));
    const unknown = element('button', '', 'Pin, unknown');
    unknown.type = 'button';
    unknown.addEventListener('click', () => {
      if (lastResult && confirmUnknown(lastResult, box.track_id)) {
        drawOverlay(lastResult); renderResults(lastResult, snapshot);
      }
    });
    const reject = element('button', 'reject-button', 'Reject');
    reject.type = 'button';
    reject.addEventListener('click', () => {
      if (lastResult && rejectTrack(lastResult, box.track_id)) {
        if (lastFrame) {
          learnNegative(learnedExamples, referenceFeature(lastFrame, box.coords));
          saveLearnedExamples(learnedExamples);
        }
        refreshAreaSelection();
        toast('Area rejected for this scan.');
      }
    });
    actions.append(select, button, unknown, reject);
    if (box.pin_id && box.name) {
      const add = element('button', '', 'Add to library');
      add.type = 'button';
      add.addEventListener('click', () => savePin({ pin_id: box.pin_id!, name: box.name! }));
      actions.append(add);
    }
    card.append(thumb, detail, actions);
    target.append(card);
  }
}
function scanOne(force = false): void {
  if (!source || (!force && source instanceof HTMLVideoElement && source.paused)) return;
  const currentSession = scanSession;
  try {
    const captured = capture();
    if (!captured) return;
    if (currentSession !== scanSession) return;
    selectedTrackIds.clear();
    savedFrameRecord = null;
    const diagnostics: ScanDiagnostics = { proposals: [], modelRejected: [], tracked: 0 };
    lastResult = scanFrame(captured.frame, lastResult, detectionModel, diagnostics,
      identityIndex, learnedExamples.negative);
    lastSnapshot = captured.snapshot;
    lastFrame = captured.frame;
    lastFrameTime = frameTime();
    const mediaWidth = source instanceof HTMLVideoElement ? source.videoWidth : source.naturalWidth;
    const mediaHeight = source instanceof HTMLVideoElement ? source.videoHeight : source.naturalHeight;
    $('stage').style.aspectRatio = `${mediaWidth} / ${mediaHeight}`;
    drawOverlay(lastResult);
    renderResults(lastResult, captured.snapshot);
    const count = lastResult.boxes.filter(box => box.review_state !== 'rejected').length;
    const now = Date.now();
    if (force || (count >= 15 && now - lastDetectionLogAt >= 10000)) {
      lastDetectionLogAt = now;
      const stage = $('stage').getBoundingClientRect();
      const { fit, left, top } = containFrame(stage.width, stage.height,
        lastResult.width, lastResult.height)!;
      const preview = source.getBoundingClientRect();
      const round = (value: number): number => Math.round(value * 100) / 100;
      console.info('PINIDENT_DETECTION web ' + JSON.stringify({
        run: `${scanSession}:${++scanSequence}`,
        source_size: [mediaWidth, mediaHeight],
        scan_size: [lastResult.width, lastResult.height],
        stage_css_size: [round(stage.width), round(stage.height)],
        preview_css_size: [round(preview.width), round(preview.height)],
        overlay_css_fit: round(fit),
        overlay_css_offset: [round(left), round(top)],
        model_version: detectionModel?.version ?? null,
        proposals_xywh: diagnostics.proposals,
        model_rejected_xywh: diagnostics.modelRejected,
        below_match_threshold_xywh: diagnostics.belowMatchThreshold,
        size_filtered_xywh: diagnostics.sizeFiltered,
        tracked_count: diagnostics.tracked,
        boxes: lastResult.boxes.map(box => ({
          source_xywh: box.coords,
          overlay_css_xywh: [round(left + box.coords[0] * fit),
            round(top + box.coords[1] * fit), round(box.coords[2] * fit), round(box.coords[3] * fit)],
          in_scan_bounds: box.coords[0] >= 0 && box.coords[1] >= 0 && box.coords[2] > 0 &&
            box.coords[3] > 0 && box.coords[0] + box.coords[2] <= lastResult!.width &&
            box.coords[1] + box.coords[3] <= lastResult!.height,
          review_state: box.review_state, pin_id: box.pin_id, name: box.name,
          suggested_pin_id: box.suggested_pin_id,
          match_strength: box.match_strength,
        })),
      }));
    }
    setStatus(identityIndex
      ? `${count} area${count === 1 ? '' : 's'} above 50% visual match. Review before adding pins.`
      : 'Catalog matching is unavailable. Draw an area manually to annotate this frame.');
    updateAnnotationControls();
  } catch (error) { setStatus(`Scan paused: ${errorText(error instanceof Error ? error : String(error))}`); stopLoop(); }
}

function frameTime(): number | null {
  return source instanceof HTMLVideoElement && !mediaStream && Number.isFinite(source.currentTime)
    ? source.currentTime : null;
}
function updateAnnotationControls(): void {
  const active = !!source && !!lastResult && !!lastSnapshot;
  const isStill = !!source && (frozen || source instanceof HTMLImageElement);
  const still = active && isStill;
  const controls = scanControls(!!source, active, isStill, source instanceof HTMLImageElement);
  $('scanView').classList.toggle('scan-active', !controls.sourceChoices);
  $('scanView').classList.toggle('scan-running', controls.running);
  $('stopButton').hidden = !controls.stop;
  $('manualSearchButton').hidden = !controls.browse;
  $('freezeButton').hidden = !controls.freeze;
  $('resumeButton').hidden = !controls.resume;
  $('saveFrameButton').hidden = !controls.stillActions;
  $<HTMLButtonElement>('saveFrameButton').disabled = frameSaving || finishingScan;
  $('saveFrameButton').textContent = savedFrameRecord ? 'Update frame annotation' : 'Save frame annotation';
  $('finishScanButton').hidden = !controls.finish;
  $<HTMLButtonElement>('finishScanButton').disabled = frameSaving || finishingScan;
  $<HTMLButtonElement>('stopButton').disabled = finishingScan;
  $('queueTrainingButton').hidden = !controls.stillActions;
  $('drawAreaButton').hidden = !controls.stillActions;
  $('selectAreasButton').hidden = !controls.stillActions;
  $('deleteAreasButton').hidden = !controls.stillActions;
  const deleteButton = $<HTMLButtonElement>('deleteAreasButton');
  deleteButton.disabled = !still || selectedTrackIds.size === 0;
  deleteButton.textContent = selectedTrackIds.size <= 1 ? 'Delete area' :
    `Delete areas${selectedTrackIds.size ? ` (${selectedTrackIds.size})` : ''}`;
  $('autoAreaButton').hidden = !controls.stillActions;
  $('autoAreaButton').setAttribute('aria-pressed', String(areaDragMode === 'auto'));
  $('drawAreaButton').setAttribute('aria-pressed', String(areaDragMode === 'draw'));
  $('selectAreasButton').setAttribute('aria-pressed', String(areaDragMode === 'select'));
  $('annotationHint').hidden = !still;
  $('annotationHint').textContent = areaDragMode === 'auto'
    ? 'Drag across existing areas to select those covered by more than 50%; drag empty space to draw a missing area. Tap an area to select it.'
    : areaDragMode === 'select'
      ? 'Tap an area to select it, or drag to select areas covered by more than 50%. Delete areas or switch to Draw area.'
      : 'Drag to draw a missing area, including over existing boxes. Tap a rectangle to select it.';
  $('boxOverlay').classList.toggle('drawing', still);
}
function freezeCurrent(): void {
  if (!source) return;
  stopLoop();
  frozen = true;
  if (source instanceof HTMLVideoElement) source.pause();
  scanOne(true);
  setStatus('Frame paused. Draw missing areas or select and delete unwanted rectangles, then save the annotation.');
  updateAnnotationControls();
}
function stagePoint(event: PointerEvent): [number, number] | null {
  if (!lastResult) return null;
  const rect = $('stage').getBoundingClientRect();
  const placement = containFrame(rect.width, rect.height, lastResult.width, lastResult.height);
  return placement ? framePoint(placement, [event.clientX - rect.left, event.clientY - rect.top],
    lastResult.width, lastResult.height) : null;
}
function difficultyHints(): DifficultyHint[] {
  const hints = new Map<string, DifficultyHint>();
  for (const hint of sharedDifficultyHints) hints.set(hint.id, hint);
  for (const record of records) {
    if (record.kind === 'difficulty') hints.set(record.id, record);
  }
  return [...hints.values()];
}
function reorderTrainingQueue(): void {
  const allHints = difficultyHints();
  trainingQueue.sort((left, right) =>
    difficultyPriority(left.box.signature, allHints) -
    difficultyPriority(right.box.signature, allHints));
}
async function refreshDifficultyHints(): Promise<void> {
  if (!navigator.onLine) return;
  const controller = new AbortController();
  const timer = window.setTimeout(() => controller.abort(), 2500);
  try { sharedDifficultyHints = await loadSharedDifficultyHints(controller.signal); }
  catch { /* Locally saved difficulty marks still order the queue offline. */ }
  finally { clearTimeout(timer); }
}
async function queueCurrentFrame(): Promise<void> {
  if (!lastResult || !lastSnapshot || !frozen || trainingQueueLoading) return;
  const snapshot = lastSnapshot, queuedSource = sourceName, session = scanSession;
  const time = frameTime();
  const candidates = lastResult.boxes.filter(box => box.review_state !== 'rejected' &&
    !queuedCandidates.has(`${scanSession}:${time?.toFixed(1) ?? 'photo'}:${box.track_id}`));
  if (!candidates.length) { toast('No new areas to review. Draw another area or capture a different frame.'); return; }
  trainingQueueLoading = true;
  try {
    await refreshDifficultyHints();
    if (!frozen || lastSnapshot !== snapshot ||
        sourceName !== queuedSource || scanSession !== session) return;
    const hints = difficultyHints();
    candidates.sort((left, right) =>
      difficultyPriority(left.signature, hints) - difficultyPriority(right.signature, hints));
    for (const box of candidates.slice(0, 40)) {
      queuedCandidates.add(`${session}:${time?.toFixed(1) ?? 'photo'}:${box.track_id}`);
      trainingQueue.push({ box: { ...box, coords: [...box.coords], signature: [...box.signature] },
        image: cropThumbnail(snapshot, box.coords, 320), source: queuedSource,
        time, frame: snapshot });
    }
    reorderTrainingQueue();
    showTab('training');
  } finally {
    trainingQueueLoading = false;
  }
}
function renderOtherPins(): void {
  const query = $<HTMLInputElement>('otherPinQuery').value.trim().toLocaleLowerCase();
  const select = $<HTMLSelectElement>('otherPinSelect');
  select.replaceChildren();
  const empty = element('option', '', query ? 'Choose a matching pin' : 'Choose a recent or other pin');
  empty.value = ''; select.append(empty);
  const appendOption = (parent: HTMLOptGroupElement | HTMLSelectElement, pin: CatalogPin): void => {
    const option = element('option', '', `${pin.name} · ${pin.pin_id}`);
    option.value = pin.pin_id; parent.append(option);
  };
  if (!query) {
    const { recents, remaining } = prioritizeRecents(filterCatalog(catalog, emptyFilters()),
      recentPinIds, true);
    if (recents.length) {
      const group = element('optgroup'); group.label = 'Recents';
      recents.forEach(pin => appendOption(group, pin)); select.append(group);
    }
    if (remaining.length) {
      const group = element('optgroup'); group.label = recents.length ? 'All other pins' : 'All pins';
      remaining.forEach(pin => appendOption(group, pin)); select.append(group);
    }
    return;
  }
  if (query.length < 2) return;
  catalog.pins.filter(pin => pin.name.toLocaleLowerCase().includes(query) ||
    pin.pin_id.toLocaleLowerCase().includes(query)).slice(0, 100)
    .forEach(pin => appendOption(select, pin));
}
function renderTraining(): void {
  const item = trainingQueue[0];
  $('trainingCount').textContent = String(trainingQueue.length);
  $('trainingCard').hidden = !item;
  const saved = records.filter(record => record.kind === 'training').length;
  const frames = records.filter(record => record.kind === 'annotation').length;
  const difficult = records.filter(record => record.kind === 'difficulty').length;
  const pending = records.filter(record =>
    record.kind === 'training' ? !record.remote_revision :
    record.kind === 'difficulty' ? !record.remote_saved :
    record.kind === 'annotation' && !record.remote_revision).length;
  $('trainingStatus').textContent = item
    ? `${trainingQueue.length} candidate${trainingQueue.length === 1 ? '' : 's'} to review · ${saved} classified · ${frames} frames saved · ${difficult} marked difficult · ${pending} awaiting cloud sync.`
    : `${saved} classified · ${frames} frames saved · ${difficult} marked difficult · ${pending} awaiting cloud sync. Capture another frame in Scan to review more.`;
  if (!item) return;
  $<HTMLImageElement>('trainingImage').src = item.image;
  $('trainingSuggestion').textContent = item.box.suggested_pin_id && item.box.suggested_name
    ? `Suggested pin: ${item.box.suggested_name}` : 'No model identity suggested. Choose unknown or search the catalog.';
  $('suggestedPinButton').hidden = !(item.box.suggested_pin_id && item.box.suggested_name);
  $('suggestedPinButton').textContent = `This is ${item.box.suggested_name ?? 'the suggested pin'}`;
  $<HTMLInputElement>('otherPinQuery').value = '';
  renderOtherPins();
}
async function markUnknown(): Promise<void> {
  const item = trainingQueue[0];
  if (!item || classificationSaving) return;
  classificationSaving = true;
  try {
    const mark = makeDifficultyMark(item.source, item.time, item.box, item.image);
    await saveRecord(mark);
    records.push(mark);
    trainingQueue.shift();
    reorderTrainingQueue();
    renderTraining();
    void syncPendingReviews().then(() => syncPendingReviews())
      .then(() => refreshSavedReviews()).catch(error => {
        toast(`Difficulty mark saved locally; cloud sync pending: ${errorText(error instanceof Error ? error : String(error))}`);
        renderTraining();
      });
  } catch (error) { toast(`Difficulty mark could not be saved: ${errorText(error instanceof Error ? error : String(error))}`); }
  finally { classificationSaving = false; }
}
async function classify(label: 'not_pin' | 'pin_unknown' | 'pin_identified',
                        identity: PinIdentity | null = null): Promise<void> {
  const item = trainingQueue[0];
  if (!item || classificationSaving) return;
  classificationSaving = true;
  try {
    const review = makeTrainingReview(item.source, item.time, item.box, item.image, label, identity);
    await saveRecord(review);
    records.push(review);
    if ((label === 'pin_identified' && identity) || label === 'not_pin') {
      const canvas = item.frame;
      const pixels = canvas.getContext('2d')?.getImageData(0, 0, canvas.width, canvas.height);
      if (pixels) {
        const feature = referenceFeature({ width: canvas.width, height: canvas.height,
          data: pixels.data }, item.box.coords);
        if (label === 'pin_identified' && identity)
          learnPositive(learnedExamples, identity.pin_id, identity.name, feature);
        else learnNegative(learnedExamples, feature);
        saveLearnedExamples(learnedExamples);
        identityIndex = combineIdentityIndex(catalogIdentityIndex, learnedExamples);
      }
    }
    if (label === 'pin_identified' && identity) rememberPinSelection(identity);
    void syncPendingReviews().then(() => syncPendingReviews())
      .then(() => refreshSavedReviews()).catch(error => {
      toast(`Review saved locally; cloud sync pending: ${errorText(error instanceof Error ? error : String(error))}`);
      renderTraining();
    });
    if (lastResult && sourceName === item.source &&
        (item.time === null ? frameTime() === null : Math.abs((frameTime() ?? -100) - item.time) < 0.1) &&
        lastResult.boxes.some(box => box.track_id === item.box.track_id)) {
      if (label === 'not_pin') rejectTrack(lastResult, item.box.track_id);
      else if (label === 'pin_unknown') confirmUnknown(lastResult, item.box.track_id);
      else if (identity) tagTrack(lastResult, item.box.track_id, identity);
      drawOverlay(lastResult);
      if (lastSnapshot) renderResults(lastResult, lastSnapshot);
    }
    trainingQueue.shift(); renderTraining();
  } catch (error) { toast(`Review could not be saved: ${errorText(error instanceof Error ? error : String(error))}`); }
  finally { classificationSaving = false; }
}
async function saveCurrentFrame(): Promise<boolean> {
  if (!lastResult || !lastSnapshot || !frozen || frameSaving) return false;
  frameSaving = true;
  updateAnnotationControls();
  try {
    const session = scanSession, snapshot = lastSnapshot;
    const record = makeFrameAnnotation(sourceName, lastFrameTime, lastResult.width,
      lastResult.height, lastSnapshot.toDataURL('image/jpeg', 0.88), lastResult.boxes,
      savedFrameRecord ?? undefined);
    await saveRecord(record);
    if (scanSession === session && lastSnapshot === snapshot) savedFrameRecord = record;
    const existing = records.findIndex(saved => saved.id === record.id);
    if (existing >= 0) records[existing] = record;
    else records.push(record);
    renderTraining(); updateAnnotationControls();
    toast(`Frame saved (${record.boxes.length} areas). Cloud sync queued.`);
    void syncPendingReviews().then(() => syncPendingReviews())
      .then(() => refreshSavedReviews()).catch(error => {
        toast(`Frame saved locally; cloud sync pending: ${errorText(error instanceof Error ? error : String(error))}`);
        renderTraining();
      });
    return true;
  } catch (error) { toast(`Annotation could not be saved: ${errorText(error instanceof Error ? error : String(error))}`); return false; }
  finally { frameSaving = false; updateAnnotationControls(); }
}
async function finishCurrentScan(): Promise<void> {
  if (!source || !lastResult || !lastSnapshot || finishingScan || frameSaving) return;
  const session = scanSession;
  const count = lastResult.boxes.filter(box => box.review_state !== 'rejected').length;
  finishingScan = true;
  stopLoop();
  frozen = true;
  if (source instanceof HTMLVideoElement) source.pause();
  updateAnnotationControls();
  try {
    const done = await completeScan(session, () => scanSession, saveCurrentFrame, stopSource);
    if (done) {
      setStatus(`Scan finished. ${count} area${count === 1 ? '' : 's'} saved on this device; cloud sync queued.`);
      toast('Scan saved. Ready for another image.');
    } else if (scanSession === session) {
      setStatus('Frame could not be saved. Review it and tap OK again.');
    }
  } catch (error) {
    setStatus(`Scan could not be finished: ${errorText(error instanceof Error ? error : String(error))}. The frame remains open.`);
  } finally { finishingScan = false; updateAnnotationControls(); }
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
async function initializeModel(): Promise<void> {
  const revision = ++modelLoadRevision;
  try {
    const next = await loadMobileModel();
    if (revision !== modelLoadRevision) return;
    detectionModel = next;
    $('modelStatus').textContent = detectionModel
      ? `Pin/non-pin model active · ${detectionModel.sample_counts.positive} pin and ${detectionModel.sample_counts.negative} non-pin reviews`
      : 'Pin/non-pin model not published. Using visual proposals only.';
  } catch (error) {
    if (revision !== modelLoadRevision) return;
    $('modelStatus').textContent = `Pin/non-pin model unavailable: ${errorText(error instanceof Error ? error : String(error))}`;
  }
}
async function initializeIdentityIndex(): Promise<void> {
  const revision = ++identityLoadRevision;
  try {
    const loaded = await loadIdentityIndex();
    if (revision !== identityLoadRevision) return;
    const wasAvailable = !!identityIndex;
    catalogIdentityIndex = loaded;
    identityIndex = combineIdentityIndex(catalogIdentityIndex, learnedExamples);
    $('identityStatus').textContent = identityIndex
      ? `Visual matching active · ${identityIndex.references.length} reference examples · percentages are match strengths`
      : 'Catalog matching unavailable. Automatic areas wait for a reference index; you can draw areas manually.';
    if (!wasAvailable && identityIndex && source && lastFrame &&
        !lastResult?.boxes.length && !savedFrameRecord &&
        (source instanceof HTMLImageElement || frozen)) scanOne(true);
  } catch (error) {
    if (revision !== identityLoadRevision) return;
    identityIndex = combineIdentityIndex(catalogIdentityIndex, learnedExamples);
    $('identityStatus').textContent = `Catalog matching unavailable: ${errorText(error instanceof Error ? error : String(error))}`;
  }
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
$('trainingTab').addEventListener('click', () => showTab('training'));
$('libraryTab').addEventListener('click', () => showTab('library'));
$('cameraButton').addEventListener('click', () => { void startCamera(); });
$('stopButton').addEventListener('click', () => {
  stopSource(); setStatus('Scan stopped. Changes since the last save were discarded.');
});
for (const id of ['cameraPhotoInput', 'photoInput']) {
  $<HTMLInputElement>(id).addEventListener('change', event => {
    const target = event.currentTarget as HTMLInputElement;
    loadPhoto(target.files?.[0]); target.value = '';
  });
}
$<HTMLInputElement>('videoInput').addEventListener('change', event => {
  const target = event.currentTarget as HTMLInputElement;
  loadVideo(target.files?.[0]); target.value = '';
});
$<HTMLVideoElement>('videoPreview').addEventListener('play', () => {
  if (source === $<HTMLVideoElement>('videoPreview')) {
    frozen = false; updateAnnotationControls(); startLoop();
  }
});
$<HTMLVideoElement>('videoPreview').addEventListener('pause', () => {
  stopLoop();
  if (source === $<HTMLVideoElement>('videoPreview') && !frozen) {
    frozen = true; scanOne(true); updateAnnotationControls();
  }
});
$<HTMLVideoElement>('videoPreview').addEventListener('seeked', () => {
  if (source === $<HTMLVideoElement>('videoPreview')) {
    lastResult = null;
    if (source.paused) scanOne(true);
  }
});
$<HTMLVideoElement>('videoPreview').addEventListener('ended', stopLoop);
$('freezeButton').addEventListener('click', freezeCurrent);
$('resumeButton').addEventListener('click', () => {
  frozen = false; updateAnnotationControls();
  if (source instanceof HTMLVideoElement) void source.play().catch(error => setStatus(errorText(error instanceof Error ? error : String(error))));
});
$('queueTrainingButton').addEventListener('click', () => { void queueCurrentFrame(); });
$('saveFrameButton').addEventListener('click', () => { void saveCurrentFrame(); });
$('finishScanButton').addEventListener('click', () => { void finishCurrentScan(); });
$('autoAreaButton').addEventListener('click', () => {
  areaDragMode = 'auto'; updateAnnotationControls();
  if (lastResult) drawOverlay(lastResult);
});
$('drawAreaButton').addEventListener('click', () => {
  areaDragMode = 'draw'; updateAnnotationControls();
  if (lastResult) drawOverlay(lastResult);
});
$('selectAreasButton').addEventListener('click', () => {
  areaDragMode = 'select'; updateAnnotationControls();
  if (lastResult) drawOverlay(lastResult);
});
$('deleteAreasButton').addEventListener('click', deleteSelectedAreas);
const overlay = $<HTMLCanvasElement>('boxOverlay');
overlay.addEventListener('pointerdown', event => {
  if (!frozen || !lastResult || !lastFrame || !event.isPrimary || event.button !== 0) return;
  const point = stagePoint(event);
  if (!point) return;
  draftStart = point; draftEnd = point;
  draftStartClient = [event.clientX, event.clientY];
  overlay.setPointerCapture(event.pointerId);
});
overlay.addEventListener('pointermove', event => {
  if (!draftStart || !lastResult) return;
  draftEnd = stagePoint(event) ?? draftEnd;
  drawOverlay(lastResult);
});
overlay.addEventListener('pointerup', event => {
  if (!draftStart || !lastResult || !lastFrame) return;
  const start = draftStart;
  const end = stagePoint(event) ?? draftEnd;
  const tap = draftStartClient !== null &&
    Math.hypot(event.clientX - draftStartClient[0], event.clientY - draftStartClient[1]) < 8;
  draftStart = null; draftEnd = null; draftStartClient = null;
  if (tap) {
    const trackId = end ? areaAtPoint(lastResult, end) : null;
    if (trackId) toggleAreaSelection(trackId);
    else if (areaDragMode !== 'draw') { selectedTrackIds.clear(); refreshAreaSelection(); }
    else drawOverlay(lastResult);
    return;
  }
  const coords = end ? clampDrawnBox(start, end, lastResult.width, lastResult.height) : null;
  const outcome = coords ? resolveAreaDrag(lastResult, coords, areaDragMode) : null;
  if (outcome?.action === 'select') {
    selectedTrackIds.clear();
    for (const trackId of outcome.trackIds) selectedTrackIds.add(trackId);
    refreshAreaSelection();
  } else {
    if (coords) {
      addManualBox(lastResult, lastFrame, coords);
      if (lastSnapshot) renderResults(lastResult, lastSnapshot);
    }
    drawOverlay(lastResult);
  }
});
overlay.addEventListener('pointercancel', () => {
  draftStart = null; draftEnd = null; draftStartClient = null;
  if (lastResult) drawOverlay(lastResult);
});
$('notPinButton').addEventListener('click', () => { void classify('not_pin'); });
$('difficultButton').addEventListener('click', () => { void markUnknown(); });
$('unknownPinButton').addEventListener('click', () => { void classify('pin_unknown'); });
$('suggestedPinButton').addEventListener('click', () => {
  const box = trainingQueue[0]?.box;
  if (box?.suggested_pin_id && box.suggested_name)
    void classify('pin_identified', { pin_id: box.suggested_pin_id, name: box.suggested_name });
});
$<HTMLInputElement>('otherPinQuery').addEventListener('input', renderOtherPins);
$('otherPinButton').addEventListener('click', () => {
  const id = $<HTMLSelectElement>('otherPinSelect').value;
  const pin = catalog.pins.find(candidate => candidate.pin_id === id);
  if (!pin) { toast('Choose a known pin from the list.'); return; }
  void classify('pin_identified', pin);
});
$('manualSearchButton').addEventListener('click', () => openSearch());
$('addCatalogPinButton').addEventListener('click', () => openSearch(null, true));
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
  if (source === $<HTMLVideoElement>('videoPreview') && !frozen && !source.paused) startLoop();
});
$<HTMLSelectElement>('collectionSelect').addEventListener('change', event => {
  state.activeId = (event.currentTarget as HTMLSelectElement).value; persist();
});
$('newCollectionButton').addEventListener('click', () => {
  const name = prompt('Name this collection');
  if (name === null) return;
  try { createCollection(state, name); persist(); } catch (error) { toast(errorText(error instanceof Error ? error : String(error))); }
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
window.addEventListener('online', () => {
  void initializeCatalog(); void initializeModel(); void initializeIdentityIndex();
  void syncPendingReviews().then(() => { void refreshSavedReviews(); }).catch(() => {});
});
document.addEventListener('visibilitychange', () => {
  if (!document.hidden) {
    void initializeModel();
    void initializeIdentityIndex();
    void syncPendingReviews().then(() => refreshSavedReviews()).catch(() => {});
  }
});
setInterval(() => {
  if (!document.hidden && navigator.onLine) {
    void initializeModel(); void initializeIdentityIndex();
  }
}, 5 * 60_000);
renderCollectionSelect(); renderLibrary(); void initializeCatalog(); void initializeModel();
void initializeIdentityIndex();
async function refreshSavedReviews(): Promise<void> {
  const saved = await loadRecords();
  const ids = new Set(saved.map(record => record.id));
  records = [...saved, ...records.filter(record => !ids.has(record.id))];
  renderTraining();
}
void refreshSavedReviews().then(() => {
  void syncPendingReviews().then(() => refreshSavedReviews()).catch(() => renderTraining());
}).catch(error => {
  $('trainingStatus').textContent = `Saved reviews unavailable: ${errorText(error instanceof Error ? error : String(error))}`;
});
if ('serviceWorker' in navigator && window.isSecureContext) {
  void navigator.serviceWorker.register('/service-worker.js').catch(() => {});
}

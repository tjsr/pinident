"""Local reviewed pin/non-pin examples and a conservative proposal ranker.

The database retains cropped source pixels and provenance for later model
training. The online ranker is a nearest-neighbour visual baseline. An
optional saved OpenCV SVM can classify its proposed rectangles; neither is a
neural object detector.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import cv2
import numpy as np

from boxdata import Coordinate
from logutil import getLog
from pin_catalog import PinMatch


DEFAULT_FEEDBACK_DB = Path(__file__).with_name('detection_feedback.sqlite3')
DESCRIPTOR_VERSION = 1
_GRAY_BYTES = 32 * 32 * np.dtype(np.float32).itemsize
_CHROMA_BYTES = 2 * np.dtype(np.float32).itemsize


@dataclass(frozen=True)
class FeedbackExample:
    example_id: str
    label: str
    source_path: str
    frame_index: int
    track_id: str
    pin_id: str | None
    coords: Coordinate
    crop_png: bytes
    tags: tuple[str, ...] = ()


def make_feedback_example(example_id: str, label: str, image: np.ndarray,
                          coords: Coordinate, source_path: str, frame_index: int,
                          track_id: str, pin_id: str | None,
                          tags: tuple[str, ...] = ()) -> FeedbackExample:
    if label not in ('positive', 'negative', 'wrong_identity'):
        raise ValueError(f'Unknown feedback label: {label}')
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError('Expected an RGB uint8 frame')
    x, y, width, height = coords
    if (width <= 0 or height <= 0 or x < 0 or y < 0 or
            x + width > image.shape[1] or y + height > image.shape[0]):
        raise ValueError('Feedback box is outside the source frame')
    crop = cv2.cvtColor(image[y:y + height, x:x + width], cv2.COLOR_RGB2BGR)
    encoded, png = cv2.imencode('.png', crop)
    if not encoded:
        raise ValueError('Could not encode feedback crop')
    return FeedbackExample(example_id, label, source_path, frame_index, track_id,
                           pin_id, coords, png.tobytes(), tags)


class FeedbackStore:
    def __init__(self, path: str | Path | None = None):
        self.path = str(path) if path is not None else ':memory:'
        self.db = sqlite3.connect(self.path)
        if self.path != ':memory:':
            # The wx connection must remain writable while the startup worker
            # reads and upgrades descriptors through its own connection.
            self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute("""CREATE TABLE IF NOT EXISTS feedback_examples (
            example_id TEXT PRIMARY KEY,
            label TEXT NOT NULL CHECK(label IN ('positive', 'negative')),
            source_path TEXT NOT NULL,
            frame_index INTEGER NOT NULL,
            track_id TEXT NOT NULL,
            pin_id TEXT,
            coords_json TEXT NOT NULL,
            crop_png BLOB NOT NULL,
            rights_status TEXT NOT NULL DEFAULT 'unverified',
            tags_json TEXT NOT NULL DEFAULT '[]')""")
        columns = {row[1] for row in self.db.execute('PRAGMA table_info(feedback_examples)')}
        if 'tags_json' not in columns:
            self.db.execute("ALTER TABLE feedback_examples ADD COLUMN tags_json TEXT NOT NULL DEFAULT '[]'")
        self.db.execute("""CREATE TABLE IF NOT EXISTS identity_rejections (
            example_id TEXT PRIMARY KEY,
            source_path TEXT NOT NULL,
            frame_index INTEGER NOT NULL,
            track_id TEXT NOT NULL,
            pin_id TEXT,
            coords_json TEXT NOT NULL,
            crop_png BLOB NOT NULL,
            tags_json TEXT NOT NULL,
            rights_status TEXT NOT NULL DEFAULT 'unverified')""")
        for table in ('feedback_examples', 'identity_rejections'):
            columns = {row[1] for row in self.db.execute(f'PRAGMA table_info({table})')}
            for name, kind in (('descriptor_version', 'INTEGER'),
                               ('descriptor_gray', 'BLOB'),
                               ('descriptor_chroma', 'BLOB')):
                if name not in columns:
                    self.db.execute(f'ALTER TABLE {table} ADD COLUMN {name} {kind}')
        self.db.commit()

    @staticmethod
    def _descriptor_columns(crop_png: bytes) -> tuple[int, bytes | None, bytes | None]:
        descriptor = _descriptor_from_png(crop_png)
        if descriptor is None:
            return DESCRIPTOR_VERSION, None, None
        gray, chroma = descriptor
        return DESCRIPTOR_VERSION, gray.tobytes(), chroma.tobytes()

    def save_identity_rejection(self, example: FeedbackExample) -> None:
        """Retain a wrong-name crop without treating it as a non-pin."""
        if example.label != 'wrong_identity':
            raise ValueError('Expected wrong_identity feedback')
        with self.db:
            self._upsert_identity_rejection(example)

    def _upsert_identity_rejection(self, example: FeedbackExample) -> None:
        version, gray, chroma = self._descriptor_columns(example.crop_png)
        self.db.execute("""INSERT INTO identity_rejections
            (example_id, source_path, frame_index, track_id, pin_id,
             coords_json, crop_png, tags_json, descriptor_version,
             descriptor_gray, descriptor_chroma)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(example_id) DO UPDATE SET
            source_path=excluded.source_path, frame_index=excluded.frame_index,
            track_id=excluded.track_id, pin_id=excluded.pin_id,
            coords_json=excluded.coords_json, crop_png=excluded.crop_png,
            tags_json=excluded.tags_json,
            descriptor_version=excluded.descriptor_version,
            descriptor_gray=excluded.descriptor_gray,
            descriptor_chroma=excluded.descriptor_chroma""",
            (example.example_id, example.source_path, example.frame_index,
             example.track_id, example.pin_id, json.dumps(example.coords),
             example.crop_png, json.dumps(example.tags), version, gray, chroma))

    def delete_identity_rejection(self, example_id: str) -> None:
        with self.db:
            self.db.execute('DELETE FROM identity_rejections WHERE example_id = ?',
                            (example_id,))

    def identity_rejections_for_source(self, source_path: str) -> list[FeedbackExample]:
        rows = self.db.execute("""SELECT example_id, source_path, frame_index,
            track_id, pin_id, coords_json, crop_png, tags_json
            FROM identity_rejections WHERE source_path = ?""", (source_path,))
        return [FeedbackExample(row[0], 'wrong_identity', row[1], row[2], row[3],
                                row[4], tuple(json.loads(row[5])), row[6],
                                tuple(json.loads(row[7]))) for row in rows]

    def identity_rejections(self) -> list[FeedbackExample]:
        rows = self.db.execute("""SELECT example_id, source_path, frame_index,
            track_id, pin_id, coords_json, crop_png, tags_json
            FROM identity_rejections ORDER BY rowid""")
        return [FeedbackExample(row[0], 'wrong_identity', row[1], row[2], row[3],
                                row[4], tuple(json.loads(row[5])), row[6],
                                tuple(json.loads(row[7]))) for row in rows]

    def get_identity_rejection(self, example_id: str) -> FeedbackExample | None:
        row = self.db.execute("""SELECT example_id, source_path, frame_index,
            track_id, pin_id, coords_json, crop_png, tags_json
            FROM identity_rejections WHERE example_id = ?""", (example_id,)).fetchone()
        return (FeedbackExample(row[0], 'wrong_identity', row[1], row[2], row[3],
                                row[4], tuple(json.loads(row[5])), row[6],
                                tuple(json.loads(row[7]))) if row else None)

    def delete_missing_identity_rejections(self, source_path: str,
                                           active_ids: set[str]) -> None:
        with self.db:
            for example in self.identity_rejections_for_source(source_path):
                if example.example_id not in active_ids:
                    self.db.execute('DELETE FROM identity_rejections WHERE example_id = ?',
                                    (example.example_id,))

    def save(self, example: FeedbackExample) -> None:
        with self.db:
            self._upsert(example)

    def _upsert(self, example: FeedbackExample) -> None:
        version, gray, chroma = self._descriptor_columns(example.crop_png)
        self.db.execute("""INSERT INTO feedback_examples
                (example_id, label, source_path, frame_index, track_id, pin_id,
                 coords_json, crop_png, rights_status, tags_json,
                 descriptor_version, descriptor_gray, descriptor_chroma)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'unverified', ?, ?, ?, ?)
                ON CONFLICT(example_id) DO UPDATE SET
                label=excluded.label, source_path=excluded.source_path,
                frame_index=excluded.frame_index, track_id=excluded.track_id,
                pin_id=excluded.pin_id, coords_json=excluded.coords_json,
                crop_png=excluded.crop_png, tags_json=excluded.tags_json,
                descriptor_version=excluded.descriptor_version,
                descriptor_gray=excluded.descriptor_gray,
                descriptor_chroma=excluded.descriptor_chroma""",
                (example.example_id, example.label, example.source_path,
                 example.frame_index, example.track_id, example.pin_id,
                 json.dumps(example.coords), example.crop_png, json.dumps(example.tags),
                 version, gray, chroma))

    def example_count(self) -> int:
        return sum(int(self.db.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0])
                   for table in ('feedback_examples', 'identity_rejections'))

    def descriptor_rows(self, progress: Callable[[int, int], None] | None = None
                        ) -> Iterator[tuple[str, str, str | None, str,
                                            np.ndarray, np.ndarray]]:
        """Yield usable cached vectors, upgrading legacy crops in bounded batches."""
        total = self.example_count()
        done = 0
        for table, label_column in (('feedback_examples', 'label'),
                                    ('identity_rejections', "'wrong_identity'")):
            upper_rowid = int(self.db.execute(
                f'SELECT COALESCE(MAX(rowid), 0) FROM {table}').fetchone()[0])
            last_rowid = 0
            while batch := self.db.execute(f'''SELECT rowid, example_id,
                {label_column}, pin_id, tags_json, descriptor_version,
                descriptor_gray, descriptor_chroma FROM {table}
                WHERE rowid > ? AND rowid <= ? ORDER BY rowid LIMIT 64''',
                (last_rowid, upper_rowid)).fetchall():
                last_rowid = batch[-1][0]
                updates: list[tuple[int, bytes | None, bytes | None, str,
                                    int, int, int]] = []
                for (_rowid, example_id, label, pin_id, tags_json, version,
                     gray_blob, chroma_blob) in batch:
                    descriptor = _decode_descriptor(version, gray_blob, chroma_blob)
                    if version != DESCRIPTOR_VERSION or descriptor is None and (
                            gray_blob is not None or chroma_blob is not None):
                        crop = self.db.execute(
                            f'SELECT crop_png FROM {table} WHERE example_id = ?',
                            (example_id,)).fetchone()
                        if crop is not None:
                            version, gray_blob, chroma_blob = self._descriptor_columns(crop[0])
                            updates.append((version, gray_blob, chroma_blob,
                                            example_id, DESCRIPTOR_VERSION,
                                            _GRAY_BYTES, _CHROMA_BYTES))
                            descriptor = _decode_descriptor(version, gray_blob, chroma_blob)
                    if descriptor is not None:
                        tags: tuple[str, ...] = tuple(json.loads(tags_json))
                        name = next((tag for tag in tags if tag.strip()), pin_id or '')
                        yield example_id, label, pin_id, name, *descriptor
                    done += 1
                if updates:
                    with self.db:
                        self.db.executemany(f'''UPDATE {table}
                            SET descriptor_version = ?, descriptor_gray = ?,
                                descriptor_chroma = ?
                            WHERE example_id = ? AND
                                (descriptor_version IS NULL OR descriptor_version != ?
                                 OR COALESCE(length(descriptor_gray), -1) != ?
                                 OR COALESCE(length(descriptor_chroma), -1) != ?)''',
                            updates)
                if progress is not None:
                    progress(done, total)

    def replace_source(self, source_path: str, examples: list[FeedbackExample]) -> None:
        """Apply a complete annotation snapshot without leaving stale source crops."""
        ids = {example.example_id for example in examples}
        with self.db:
            for example in examples:
                self._upsert(example)
            for (example_id,) in self.db.execute(
                    'SELECT example_id FROM feedback_examples WHERE source_path = ?',
                    (source_path,)).fetchall():
                if example_id not in ids:
                    self.db.execute('DELETE FROM feedback_examples WHERE example_id = ?',
                                    (example_id,))

    def apply_source_delta(self, source_path: str,
                           examples: list[FeedbackExample],
                           removed_ids: set[str]) -> None:
        """Upsert only changed examples and delete explicitly stale source IDs."""
        if any(example.source_path != source_path for example in examples):
            raise ValueError('Training example source does not match session')
        with self.db:
            for example in examples:
                if example.label == 'wrong_identity':
                    self._upsert_identity_rejection(example)
                else:
                    self._upsert(example)
            for example_id in removed_ids:
                self.db.execute('''DELETE FROM feedback_examples
                    WHERE source_path = ? AND example_id = ?''',
                    (source_path, example_id))
                self.db.execute('''DELETE FROM identity_rejections
                    WHERE source_path = ? AND example_id = ?''',
                    (source_path, example_id))

    def delete(self, example_id: str) -> None:
        with self.db:
            self.db.execute('DELETE FROM feedback_examples WHERE example_id = ?',
                            (example_id,))

    def delete_missing_source(self, source_path: str, active_ids: set[str]) -> None:
        rows = self.db.execute('SELECT example_id FROM feedback_examples WHERE source_path = ?',
                               (source_path,)).fetchall()
        with self.db:
            for (example_id,) in rows:
                if example_id not in active_ids:
                    self.db.execute('DELETE FROM feedback_examples WHERE example_id = ?',
                                    (example_id,))

    def examples(self) -> list[FeedbackExample]:
        rows = self.db.execute("""SELECT example_id, label, source_path, frame_index,
            track_id, pin_id, coords_json, crop_png, tags_json FROM feedback_examples ORDER BY rowid""")
        return [FeedbackExample(row[0], row[1], row[2], row[3], row[4], row[5],
                                tuple(json.loads(row[6])), row[7],
                                tuple(json.loads(row[8]))) for row in rows]

    def counts(self) -> dict[str, int]:
        counts = {'positive': 0, 'negative': 0}
        for label, count in self.db.execute(
                'SELECT label, COUNT(*) FROM feedback_examples GROUP BY label'):
            counts[label] = count
        return counts

    def get(self, example_id: str) -> FeedbackExample | None:
        row = self.db.execute('''SELECT example_id, label, source_path, frame_index,
            track_id, pin_id, coords_json, crop_png, tags_json FROM feedback_examples
            WHERE example_id = ?''', (example_id,)).fetchone()
        return (FeedbackExample(row[0], row[1], row[2], row[3], row[4], row[5],
                                tuple(json.loads(row[6])), row[7],
                                tuple(json.loads(row[8]))) if row else None)

    def metadata_for_source(self, source_path: str) -> dict[str, tuple]:
        rows = self.db.execute('''SELECT example_id, label, frame_index, track_id,
            pin_id, coords_json, tags_json FROM feedback_examples WHERE source_path = ?''',
                               (source_path,))
        return {row[0]: (row[1], row[2], row[3], row[4], tuple(json.loads(row[5])),
                         tuple(json.loads(row[6])))
                for row in rows}

    def training_metadata_for_source(self, source_path: str) -> dict[str, tuple]:
        """Include identity corrections without loading stored image blobs."""
        metadata = self.metadata_for_source(source_path)
        rows = self.db.execute('''SELECT example_id, frame_index, track_id,
            pin_id, coords_json, tags_json FROM identity_rejections
            WHERE source_path = ?''', (source_path,))
        for example_id, frame_index, track_id, pin_id, coords, tags in rows:
            if example_id in metadata:
                raise ValueError(f'Feedback ID exists in both stores: {example_id}')
            metadata[example_id] = ('wrong_identity', frame_index, track_id,
                                    pin_id, tuple(json.loads(coords)),
                                    tuple(json.loads(tags)))
        return metadata

    def identity_rejection_count(self, source_path: str | None = None) -> int:
        if source_path is None:
            return int(self.db.execute(
                'SELECT COUNT(*) FROM identity_rejections').fetchone()[0])
        return int(self.db.execute(
            'SELECT COUNT(*) FROM identity_rejections WHERE source_path = ?',
            (source_path,)).fetchone()[0])

    def close(self) -> None:
        self.db.close()


def _descriptor(crop: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    if crop.size == 0:
        return None
    lab = cv2.cvtColor(crop, cv2.COLOR_RGB2LAB)
    gray = cv2.resize(lab[:, :, 0], (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    gray -= gray.mean()
    norm = float(np.linalg.norm(gray))
    if norm < 64:
        return None
    gray /= norm
    chroma = np.median(lab.reshape(-1, 3), axis=0)[1:].astype(np.float32)
    return gray.ravel(), chroma


def _descriptor_from_png(crop_png: bytes) -> tuple[np.ndarray, np.ndarray] | None:
    bgr = cv2.imdecode(np.frombuffer(crop_png, dtype=np.uint8), cv2.IMREAD_COLOR)
    return _descriptor(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)) if bgr is not None else None


def _decode_descriptor(version: int | None, gray_blob: bytes | None,
                       chroma_blob: bytes | None
                       ) -> tuple[np.ndarray, np.ndarray] | None:
    if (version != DESCRIPTOR_VERSION or gray_blob is None or chroma_blob is None or
            len(gray_blob) != _GRAY_BYTES or len(chroma_blob) != _CHROMA_BYTES):
        return None
    return (np.frombuffer(gray_blob, dtype=np.float32),
            np.frombuffer(chroma_blob, dtype=np.float32))


def _classifier_feature(descriptor: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    gray, chroma = descriptor
    return np.concatenate((gray, chroma / 80.0 * 0.15)).astype(np.float32)


class FeedbackModel:
    """Use reviewed crops to rank proposals and suppress very close negatives."""

    def __init__(self, store: FeedbackStore, *, load: bool = True):
        self.store = store
        self.version = 0
        self.trained_model = None
        self._samples: dict[str, tuple[str, str | None, str, np.ndarray, np.ndarray]] = {}
        self._rebuild_vectors()
        if load:
            self.reload()

    def reload(self, progress: Callable[[int, int], None] | None = None) -> None:
        samples: dict[str, tuple[str, str | None, str, np.ndarray, np.ndarray]] = {}
        for example_id, label, pin_id, name, gray, chroma in self.store.descriptor_rows(progress):
            samples[example_id] = label, pin_id, name, gray, chroma
        self._samples = samples
        self._rebuild_vectors()
        self.version += 1

    def refresh(self, example_ids: set[str]) -> None:
        if any((self._samples.get(example_id, ('',))[0] in ('positive', 'negative') or
                self.store.get(example_id) is not None)
               for example_id in example_ids):
            self.trained_model = None
        for example_id in example_ids:
            self._samples.pop(example_id, None)
            example = (self.store.get(example_id) or
                       self.store.get_identity_rejection(example_id))
            if example is not None:
                self._load_example(example)
        self._rebuild_vectors()
        self.version += 1

    def snapshot_for_processing(self) -> 'FeedbackModel':
        """Capture stable ranker vectors without sharing SQLite with a worker."""
        snapshot = object.__new__(FeedbackModel)
        snapshot.store = None
        snapshot.version = self.version
        snapshot._vectors = {label: (gray, chroma)
                             for label, (gray, chroma) in self._vectors.items()}
        snapshot._identity_vectors = self._identity_vectors
        snapshot._wrong_identity_vectors = self._wrong_identity_vectors
        snapshot.identity_sample_count = self.identity_sample_count
        snapshot.sample_counts = dict(self.sample_counts)
        snapshot.trained_model = self.trained_model
        return snapshot

    def _load_example(self, example: FeedbackExample) -> None:
        descriptor = _descriptor_from_png(example.crop_png)
        if descriptor is not None:
            self._samples[example.example_id] = (
                example.label, example.pin_id,
                next((tag for tag in example.tags if tag.strip()), example.pin_id or ''),
                *descriptor)

    def _rebuild_vectors(self) -> None:
        grouped: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {
            'positive': [], 'negative': []}
        identities: dict[str, tuple[str, list[tuple[np.ndarray, np.ndarray]]]] = {}
        wrong: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
        for label, pin_id, name, gray, chroma in self._samples.values():
            if label in grouped:
                grouped[label].append((gray, chroma))
            if pin_id and label == 'positive':
                entry = identities.setdefault(pin_id, (name, []))
                entry[1].append((gray, chroma))
            elif pin_id and label == 'wrong_identity':
                wrong.setdefault(pin_id, []).append((gray, chroma))
        self._vectors = {
            label: (np.stack([item[0] for item in samples]) if samples else None,
                    np.stack([item[1] for item in samples]) if samples else None)
            for label, samples in grouped.items()}
        self.sample_counts = {label: len(samples)
                              for label, samples in grouped.items()}
        self._identity_vectors = {
            pin_id: (name, np.stack([item[0] for item in samples]),
                     np.stack([item[1] for item in samples]))
            for pin_id, (name, samples) in identities.items()}
        self._wrong_identity_vectors = {
            pin_id: (np.stack([item[0] for item in samples]),
                     np.stack([item[1] for item in samples]))
            for pin_id, samples in wrong.items()}
        self.identity_sample_count = sum(len(samples) for _, samples in identities.values())

    @staticmethod
    def _similarity(descriptor: tuple[np.ndarray, np.ndarray],
                    vectors: np.ndarray, colours: np.ndarray) -> float:
        gray, chroma = descriptor
        texture = np.maximum(0.0, vectors @ gray)
        colour = np.maximum(0.0, 1.0 - np.linalg.norm(colours - chroma, axis=1) / 80.0)
        return float(np.max(0.85 * texture + 0.15 * colour))

    def suggest_identity(self, crop: np.ndarray,
                         excluded_pin_ids: frozenset[str] = frozenset(),
                         excluded_names: frozenset[str] = frozenset()) -> PinMatch | None:
        """Suggest only a well-separated visual match to directly reviewed IDs."""
        if not self._identity_vectors:
            return None
        descriptor = _descriptor(crop)
        if descriptor is None:
            return None
        ranked: list[tuple[float, str, str]] = []
        for pin_id, (name, vectors, colours) in self._identity_vectors.items():
            if pin_id in excluded_pin_ids or name.casefold() in excluded_names:
                continue
            positive = self._similarity(descriptor, vectors, colours)
            negatives = self._wrong_identity_vectors.get(pin_id)
            if negatives is not None and self._similarity(descriptor, *negatives) >= positive - 0.05:
                continue
            ranked.append((positive, pin_id, name))
        ranked.sort(reverse=True)
        if not ranked or ranked[0][0] < 0.90 or (
                len(ranked) > 1 and ranked[0][0] < ranked[1][0] + 0.05):
            return None
        score, pin_id, name = ranked[0]
        return PinMatch(pin_id, name, 0, score)

    def _best_similarity(self, descriptor: tuple[np.ndarray, np.ndarray],
                         label: str) -> float:
        vectors, colours = self._vectors[label]
        if vectors is None or colours is None:
            return 0.0
        gray, chroma = descriptor
        texture = np.maximum(0.0, vectors @ gray)
        colour = np.maximum(0.0, 1.0 - np.linalg.norm(colours - chroma, axis=1) / 80.0)
        return float(np.max(0.85 * texture + 0.15 * colour))

    def _best_similarities(self, descriptors: list[tuple[np.ndarray, np.ndarray]],
                           label: str) -> np.ndarray:
        """Score proposal crops together against the stored review matrix."""
        vectors, colours = self._vectors[label]
        if vectors is None or colours is None or not descriptors:
            return np.zeros(len(descriptors), dtype=np.float32)
        grays = np.stack([gray for gray, _chroma in descriptors])
        chromas = np.stack([chroma for _gray, chroma in descriptors])
        texture = np.maximum(0.0, vectors @ grays.T)
        colour = np.maximum(
            0.0, 1.0 - np.linalg.norm(
                colours[:, None, :] - chromas[None, :, :], axis=2) / 80.0)
        return np.max(0.85 * texture + 0.15 * colour, axis=0)

    def filter_and_rank(self, image: np.ndarray,
                        boxes: list[Coordinate]) -> list[Coordinate]:
        scored: list[tuple[float, int, Coordinate]] = []
        candidates: list[tuple[int, Coordinate, tuple[np.ndarray, np.ndarray], float]] = []
        suppressed = 0
        trained_negative = 0
        unscored = 0
        trained = self.trained_model
        for index, coords in enumerate(boxes):
            x, y, width, height = coords
            if (width <= 0 or height <= 0 or x < 0 or y < 0 or
                    x + width > image.shape[1] or y + height > image.shape[0]):
                continue
            descriptor = _descriptor(image[y:y + height, x:x + width])
            if descriptor is None:
                unscored += 1
                scored.append((0.0, index, coords))
                continue
            model_penalty = 0.0
            if trained is not None:
                feature = trained.feature(image[y:y + height, x:x + width])
                if feature is not None and trained.classify_feature(feature) == 'negative':
                    # Stock artwork and real video crops have different
                    # backgrounds. A binary prediction is useful ordering
                    # evidence, but a hard veto loses visible pins entirely.
                    trained_negative += 1
                    model_penalty = 0.03
            candidates.append((index, coords, descriptor, model_penalty))
        descriptors = [descriptor for _index, _coords, descriptor, _penalty in candidates]
        positives = self._best_similarities(descriptors, 'positive')
        negatives = self._best_similarities(descriptors, 'negative')
        for (index, coords, _descriptor_value, model_penalty), positive, negative in zip(
                candidates, positives, negatives):
            if negative >= 0.88 and negative > positive + 0.06:
                suppressed += 1
                continue
            scored.append((float(positive - negative - model_penalty), index, coords))
        scored.sort(key=lambda item: (-item[0], item[1]))
        getLog().info(
            'feedback revision=%d samples=%d/%d proposals=%d suppressed=%d '
            'trained_negative=%d unscored=%d kept=%d',
            self.version, self.sample_counts['positive'], self.sample_counts['negative'],
            len(boxes), suppressed, trained_negative, unscored, len(scored))
        return [coords for _, _, coords in scored]

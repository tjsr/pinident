"""Local reviewed pin/non-pin examples and a conservative proposal ranker.

The database retains cropped source pixels and provenance for later model
training. The current online model is a nearest-neighbour visual baseline,
not a learned neural object detector.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from boxdata import Coordinate
from logutil import getLog


DEFAULT_FEEDBACK_DB = Path(__file__).with_name('detection_feedback.sqlite3')


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
        self.db.commit()

    def save_identity_rejection(self, example: FeedbackExample) -> None:
        """Retain a wrong-name crop without treating it as a non-pin."""
        if example.label != 'wrong_identity':
            raise ValueError('Expected wrong_identity feedback')
        with self.db:
            self._upsert_identity_rejection(example)

    def _upsert_identity_rejection(self, example: FeedbackExample) -> None:
        self.db.execute("""INSERT INTO identity_rejections
            (example_id, source_path, frame_index, track_id, pin_id,
             coords_json, crop_png, tags_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(example_id) DO UPDATE SET
            source_path=excluded.source_path, frame_index=excluded.frame_index,
            track_id=excluded.track_id, pin_id=excluded.pin_id,
            coords_json=excluded.coords_json, crop_png=excluded.crop_png,
            tags_json=excluded.tags_json""",
            (example.example_id, example.source_path, example.frame_index,
             example.track_id, example.pin_id, json.dumps(example.coords),
             example.crop_png, json.dumps(example.tags)))

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
        self.db.execute("""INSERT INTO feedback_examples
                (example_id, label, source_path, frame_index, track_id, pin_id,
                 coords_json, crop_png, rights_status, tags_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'unverified', ?)
                ON CONFLICT(example_id) DO UPDATE SET
                label=excluded.label, source_path=excluded.source_path,
                frame_index=excluded.frame_index, track_id=excluded.track_id,
                pin_id=excluded.pin_id, coords_json=excluded.coords_json,
                crop_png=excluded.crop_png, tags_json=excluded.tags_json""",
                (example.example_id, example.label, example.source_path,
                 example.frame_index, example.track_id, example.pin_id,
                 json.dumps(example.coords), example.crop_png, json.dumps(example.tags)))

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


class FeedbackModel:
    """Use reviewed crops to rank proposals and suppress very close negatives."""

    def __init__(self, store: FeedbackStore):
        self.store = store
        self.version = 0
        self.reload()

    def reload(self) -> None:
        self._samples: dict[str, tuple[str, np.ndarray, np.ndarray]] = {}
        for example in self.store.examples():
            self._load_example(example)
        self._rebuild_vectors()
        self.version += 1

    def refresh(self, example_ids: set[str]) -> None:
        for example_id in example_ids:
            self._samples.pop(example_id, None)
            example = self.store.get(example_id)
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
        snapshot.sample_counts = dict(self.sample_counts)
        return snapshot

    def _load_example(self, example: FeedbackExample) -> None:
        encoded = np.frombuffer(example.crop_png, dtype=np.uint8)
        bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if bgr is None:
            return
        descriptor = _descriptor(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        if descriptor is not None:
            self._samples[example.example_id] = (example.label, *descriptor)

    def _rebuild_vectors(self) -> None:
        grouped: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {
            'positive': [], 'negative': []}
        for label, gray, chroma in self._samples.values():
            grouped[label].append((gray, chroma))
        self._vectors = {
            label: (np.stack([item[0] for item in samples]) if samples else None,
                    np.stack([item[1] for item in samples]) if samples else None)
            for label, samples in grouped.items()}
        self.sample_counts = {label: len(samples)
                              for label, samples in grouped.items()}

    def _best_similarity(self, descriptor: tuple[np.ndarray, np.ndarray],
                         label: str) -> float:
        vectors, colours = self._vectors[label]
        if vectors is None or colours is None:
            return 0.0
        gray, chroma = descriptor
        texture = np.maximum(0.0, vectors @ gray)
        colour = np.maximum(0.0, 1.0 - np.linalg.norm(colours - chroma, axis=1) / 80.0)
        return float(np.max(0.85 * texture + 0.15 * colour))

    def filter_and_rank(self, image: np.ndarray,
                        boxes: list[Coordinate]) -> list[Coordinate]:
        scored: list[tuple[float, int, Coordinate]] = []
        suppressed = 0
        unscored = 0
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
            positive = self._best_similarity(descriptor, 'positive')
            negative = self._best_similarity(descriptor, 'negative')
            if negative >= 0.88 and negative > positive + 0.06:
                suppressed += 1
                continue
            scored.append((positive - negative, index, coords))
        scored.sort(key=lambda item: (-item[0], item[1]))
        getLog().info(
            'feedback revision=%d samples=%d/%d proposals=%d suppressed=%d unscored=%d kept=%d',
            self.version, self.sample_counts['positive'], self.sample_counts['negative'],
            len(boxes), suppressed, unscored, len(scored))
        return [coords for _, _, coords in scored]

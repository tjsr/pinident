"""Local reference-image catalog and conservative visual pin matching."""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


DEFAULT_CATALOG = Path(__file__).with_name("known_pins.sqlite3")


@dataclass(frozen=True)
class PinReference:
    pin_id: str
    name: str
    image: np.ndarray
    mask: np.ndarray | None = None
    year: int | None = None
    show: str | None = None


@dataclass(frozen=True)
class CatalogStockImage:
    """One original Pinpanion image and its stored source provenance."""

    reference_id: int
    pin_id: str
    name: str
    image_bytes: bytes
    source: str
    rights: str
    tags: tuple[str, ...]


@dataclass(frozen=True)
class PinMatch:
    pin_id: str
    name: str
    inliers: int
    confidence: float | None = None
    year: int | None = None
    show: str | None = None


def _short_show_name(name: str, year: int | None) -> str:
    show = re.sub(r'^PAX\s+', '', name.strip(), flags=re.IGNORECASE)
    if year is not None:
        show = re.sub(rf'\s+{year}$', '', show)
    return show.strip()


class PinCatalog:
    def __init__(self, path: str | Path = DEFAULT_CATALOG):
        self.path = Path(path)

    @staticmethod
    def _ensure_schema(db: sqlite3.Connection) -> None:
        db.execute("""CREATE TABLE IF NOT EXISTS pins (
            pin_id TEXT PRIMARY KEY, name TEXT NOT NULL)""")
        db.execute("""CREATE TABLE IF NOT EXISTS pin_references (
            id INTEGER PRIMARY KEY, pin_id TEXT NOT NULL REFERENCES pins(pin_id),
            image BLOB NOT NULL, source TEXT NOT NULL, rights TEXT NOT NULL)""")
        db.execute("""CREATE TABLE IF NOT EXISTS pinpanion_metadata (
            pin_id TEXT PRIMARY KEY REFERENCES pins(pin_id),
            raw_json TEXT NOT NULL, image_url TEXT NOT NULL)""")
        db.execute("""CREATE TABLE IF NOT EXISTS pinpanion_events (
            event_id INTEGER PRIMARY KEY, name TEXT NOT NULL)""")

    def upsert_event_names(self, events: list[dict]) -> int:
        """Save changed public event names for existing and future pin records."""
        changed = 0
        with sqlite3.connect(self.path) as db:
            self._ensure_schema(db)
            for event in events:
                if not isinstance(event, dict):
                    continue
                event_id, name = event.get('id'), event.get('name')
                if (isinstance(event_id, bool) or not isinstance(event_id, int) or
                        not isinstance(name, str) or not name.strip()):
                    continue
                name = name.strip()
                previous = db.execute(
                    'SELECT name FROM pinpanion_events WHERE event_id = ?',
                    (event_id,)).fetchone()
                if previous is not None and previous[0] == name:
                    continue
                db.execute("""INSERT INTO pinpanion_events (event_id, name)
                    VALUES (?, ?) ON CONFLICT(event_id) DO UPDATE SET name = excluded.name""",
                    (event_id, name))
                changed += 1
        return changed

    def pin_ids(self) -> set[str]:
        if not self.path.is_file():
            return set()
        with sqlite3.connect(self.path) as db:
            self._ensure_schema(db)
            return {row[0] for row in db.execute("SELECT pin_id FROM pins")}

    def list_pins(self) -> list[tuple[str, str]]:
        """Return stable ID/name choices for the desktop annotation picker."""
        if not self.path.is_file():
            return []
        with sqlite3.connect(self.path) as db:
            self._ensure_schema(db)
            return [(pin_id, name) for pin_id, name in db.execute(
                "SELECT pin_id, name FROM pins ORDER BY name COLLATE NOCASE, pin_id")]

    def search_pins(self, query: str, limit: int = 30) -> list[dict[str, str | None]]:
        """Find catalog entries for manual mobile tagging without loading image features."""
        if not self.path.is_file():
            return []
        if not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100")
        escaped = (query.strip().replace('\\', '\\\\')
                   .replace('%', '\\%').replace('_', '\\_'))
        with sqlite3.connect(self.path) as db:
            self._ensure_schema(db)
            rows = db.execute("""SELECT p.pin_id, p.name, m.image_url FROM pins p
                LEFT JOIN pinpanion_metadata m ON m.pin_id = p.pin_id
                WHERE p.name LIKE ? ESCAPE '\\' OR p.pin_id LIKE ? ESCAPE '\\'
                ORDER BY p.name COLLATE NOCASE, p.pin_id LIMIT ?""",
                (f'%{escaped}%', f'%{escaped}%', limit)).fetchall()
        return [{'pin_id': pin_id, 'name': name, 'image_url': image_url}
                for pin_id, name, image_url in rows]

    def get_pin(self, pin_id: str) -> dict[str, str | None] | None:
        if not self.path.is_file():
            return None
        with sqlite3.connect(self.path) as db:
            self._ensure_schema(db)
            row = db.execute("""SELECT p.pin_id, p.name, m.image_url FROM pins p
                LEFT JOIN pinpanion_metadata m ON m.pin_id = p.pin_id
                WHERE p.pin_id = ?""", (pin_id,)).fetchone()
        return ({'pin_id': row[0], 'name': row[1], 'image_url': row[2]}
                if row else None)

    def add_reference(self, pin_id: str, name: str, image_path: str | Path,
                      source: str, rights: str) -> None:
        """Store a local image with its identity and provenance."""
        if not all(value.strip() for value in (pin_id, name, source, rights)):
            raise ValueError("Pin ID, name, source, and rights must be nonempty")
        image_bytes = Path(image_path).read_bytes()
        if cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR) is None:
            raise ValueError("Reference image could not be decoded")
        with sqlite3.connect(self.path) as db:
            self._ensure_schema(db)
            existing = db.execute("SELECT name FROM pins WHERE pin_id = ?", (pin_id,)).fetchone()
            if existing and existing[0] != name:
                raise ValueError(f"Pin {pin_id!r} already has name {existing[0]!r}")
            db.execute("INSERT OR IGNORE INTO pins (pin_id, name) VALUES (?, ?)", (pin_id, name))
            db.execute("INSERT INTO pin_references (pin_id, image, source, rights) VALUES (?, ?, ?, ?)",
                       (pin_id, image_bytes, source, rights))

    def import_pinpanion_pin(self, record: dict, image_bytes: bytes, image_url: str) -> bool:
        """Insert one Pinpanion pin and image atomically; return False if present."""
        pin_id = f"pinpanion:{record['id']}"
        name = record['name']
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Pinpanion pin name must be nonempty")
        if cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR) is None:
            raise ValueError("Pinpanion image could not be decoded")
        with sqlite3.connect(self.path) as db:
            self._ensure_schema(db)
            if db.execute("SELECT 1 FROM pins WHERE pin_id = ?", (pin_id,)).fetchone():
                return False
            db.execute("INSERT INTO pins (pin_id, name) VALUES (?, ?)", (pin_id, name))
            db.execute("INSERT INTO pin_references (pin_id, image, source, rights) VALUES (?, ?, ?, ?)",
                       (pin_id, image_bytes, image_url, "source terms unverified"))
            db.execute("INSERT INTO pinpanion_metadata (pin_id, raw_json, image_url) VALUES (?, ?, ?)",
                       (pin_id, json.dumps(record, sort_keys=True), image_url))
        return True

    def references(self) -> list[PinReference]:
        if not self.path.is_file():
            return []
        with sqlite3.connect(self.path) as db:
            self._ensure_schema(db)
            rows = db.execute("""SELECT p.pin_id, p.name, r.image, m.raw_json
                FROM pin_references r JOIN pins p ON p.pin_id = r.pin_id
                LEFT JOIN pinpanion_metadata m ON m.pin_id = p.pin_id
                ORDER BY p.pin_id, r.id""").fetchall()
            event_names = dict(db.execute('SELECT event_id, name FROM pinpanion_events'))
        references = []
        for pin_id, name, data, raw_json in rows:
            decoded = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
            if decoded is None:
                continue
            if decoded.ndim == 2:
                rgb = cv2.cvtColor(decoded, cv2.COLOR_GRAY2RGB)
                mask = None
            elif decoded.shape[2] == 4:
                rgb = cv2.cvtColor(decoded, cv2.COLOR_BGRA2RGB)
                mask = np.uint8(decoded[:, :, 3] > 128) * 255
            else:
                rgb = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
                mask = None
            metadata = json.loads(raw_json) if raw_json else {}
            raw_year = metadata.get('year')
            year = (raw_year if isinstance(raw_year, int) and
                    not isinstance(raw_year, bool) else None)
            event_id = metadata.get('paxEventId')
            event_name = (event_names.get(event_id)
                          if isinstance(event_id, int) and not isinstance(event_id, bool)
                          else None)
            show = _short_show_name(event_name, year) if event_name else None
            references.append(PinReference(pin_id, name, rgb, mask, year, show))
        return references

    def stock_training_images(self) -> Iterator[CatalogStockImage]:
        """Read every locally imported Pinpanion stock image on the caller's thread.

        The metadata URL identifies the original feed image. Other manually
        added references, even for the same pin ID, are not stock images.
        """
        if not self.path.is_file():
            return
        with sqlite3.connect(self.path) as db:
            self._ensure_schema(db)
            rows = db.execute("""SELECT r.id, p.pin_id, p.name, r.image, r.source, r.rights,
                m.raw_json
                FROM pin_references r JOIN pins p ON p.pin_id = r.pin_id
                JOIN pinpanion_metadata m ON m.pin_id = p.pin_id
                WHERE r.source = m.image_url ORDER BY r.id""")
            for reference_id, pin_id, name, image, source, rights, raw_json in rows:
                source_name = json.loads(raw_json).get('name')
                tag = (source_name if isinstance(source_name, str) and
                       source_name.strip() else name)
                yield CatalogStockImage(reference_id, pin_id, name, image,
                                        source, rights, (tag,))

    def stock_training_state(self) -> tuple[int, int, int]:
        """Cheap append-only catalog revision for classifier freshness checks."""
        if not self.path.is_file():
            return (0, 0, 0)
        with sqlite3.connect(self.path) as db:
            self._ensure_schema(db)
            row = db.execute("""SELECT COUNT(*), COALESCE(MAX(r.id), 0),
                COALESCE(SUM(LENGTH(r.image)), 0)
                FROM pin_references r JOIN pinpanion_metadata m ON m.pin_id = r.pin_id
                WHERE r.source = m.image_url""").fetchone()
        return tuple(int(value) for value in row)


class PinMatcher:
    """Require repeatable local features and a consistent geometric match."""

    def __init__(self, references: list[PinReference]):
        self._orb = cv2.ORB_create(nfeatures=800)
        self._query_orb = cv2.ORB_create(nfeatures=400)
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self._references = []
        self._appearance_references = [
            (reference, self._appearance_vector(reference.image, reference.mask))
            for reference in references if reference.image.size]
        self._appearance_matrix = (np.stack([signature for _, signature in
                                             self._appearance_references])
                                   if self._appearance_references else None)
        self._pattern_matrix = (np.stack([
            self._pattern_vector(reference.image, reference.mask)
            for reference, _signature in self._appearance_references])
            if self._appearance_references else None)
        for reference in references:
            keypoints, descriptors = self._features(reference.image, reference.mask)
            if descriptors is not None and len(keypoints) >= 12:
                self._references.append((reference, keypoints, descriptors))
        self._index = None
        if len(self._references) > 32:
            # Train an approximate descriptor index so a large catalog does
            # not require a full comparison with every pin for every crop.
            # Multi-probe level 2 explores hundreds of neighbouring hash
            # buckets per query on a large catalog. Twelve independent tables
            # recover the same candidates much faster without that expansion.
            self._index = cv2.FlannBasedMatcher(
                dict(algorithm=6, table_number=12, key_size=20, multi_probe_level=0),
                dict(checks=16))
            self._index.add([item[2] for item in self._references])
            self._index.train()

    @property
    def has_references(self) -> bool:
        return bool(self._appearance_references)

    @staticmethod
    def _appearance_vector(rgb: np.ndarray,
                           mask: np.ndarray | None = None) -> np.ndarray:
        """Compact colour signature for catalog comparison."""
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        histogram = cv2.calcHist([hsv], [0, 1], mask, [12, 4], [0, 180, 0, 256])
        vector = histogram.ravel().astype(np.float32)
        length = float(np.linalg.norm(vector))
        return vector / length if length else vector

    @staticmethod
    def _pattern_vector(rgb: np.ndarray,
                        mask: np.ndarray | None = None) -> np.ndarray:
        """Normalized coarse layout; transparent stock margins use the pin's mean."""
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        if mask is not None and np.any(mask > 128):
            gray = np.where(mask > 128, gray, float(np.mean(gray[mask > 128])))
        pattern = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA).ravel()
        pattern -= float(np.mean(pattern))
        length = float(np.linalg.norm(pattern))
        return pattern / length if length > 1e-4 else np.zeros(64, dtype=np.float32)

    def _appearance_scores(self, crop: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compare colour and the best quarter-turn spatial layout in one batch."""
        assert self._appearance_matrix is not None and self._pattern_matrix is not None
        colour = np.clip(self._appearance_matrix @ self._appearance_vector(crop), 0.0, 1.0)
        pattern = self._pattern_vector(crop).reshape(8, 8)
        rotations = np.stack([np.rot90(pattern, turns).ravel()
                              for turns in range(4)], axis=1)
        layout = np.clip(np.max(self._pattern_matrix @ rotations, axis=1), 0.0, 1.0)
        strength = 0.35 * colour + 0.65 * layout
        return colour, layout, strength

    def _features(self, rgb: np.ndarray, mask: np.ndarray | None = None,
                  query: bool = False):
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        return (self._query_orb if query else self._orb).detectAndCompute(gray, mask)

    def _candidate_indices(self, descriptors: np.ndarray) -> list[int]:
        if self._index is None:
            return list(range(len(self._references)))
        pairs = self._index.knnMatch(descriptors, k=2)
        votes = Counter(pair[0].imgIdx for pair in pairs if len(pair) == 2
                        and pair[0].distance < 0.80 * pair[1].distance)
        return [index for index, count in votes.most_common(24) if count >= 3]

    def match(self, crop: np.ndarray,
              excluded_pin_ids: frozenset[str] = frozenset(),
              excluded_names: frozenset[str] = frozenset()) -> PinMatch | None:
        if not self._references or crop.size == 0 or min(crop.shape[:2]) < 24:
            return None
        keypoints, descriptors = self._features(crop, query=True)
        if descriptors is None or len(keypoints) < 12:
            return None
        best_by_pin: dict[str, PinMatch] = {}
        for index in self._candidate_indices(descriptors):
            reference, reference_points, reference_descriptors = self._references[index]
            if (reference.pin_id in excluded_pin_ids or
                    reference.name.casefold() in excluded_names):
                continue
            pairs = self._matcher.knnMatch(reference_descriptors, descriptors, k=2)
            good = [first for pair in pairs if len(pair) == 2
                    for first, second in [pair] if first.distance < 0.72 * second.distance]
            if len(good) < 10:
                continue
            source = np.float32([reference_points[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            target = np.float32([keypoints[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            _, mask = cv2.findHomography(source, target, cv2.RANSAC, 4.0)
            inliers = int(mask.sum()) if mask is not None else 0
            if inliers < 8 or inliers / len(good) < 0.60:
                continue
            old = best_by_pin.get(reference.pin_id)
            if old is None or inliers > old.inliers:
                best_by_pin[reference.pin_id] = PinMatch(
                    reference.pin_id, reference.name, inliers, inliers / len(good),
                    reference.year, reference.show)
        ranked = sorted(best_by_pin.values(), key=lambda match: match.inliers, reverse=True)
        if not ranked or (len(ranked) > 1 and ranked[0].inliers < ranked[1].inliers + 4):
            return None
        return ranked[0]

    def suggest(self, crop: np.ndarray,
                excluded_pin_ids: frozenset[str] = frozenset(),
                excluded_names: frozenset[str] = frozenset()) -> PinMatch | None:
        """Abstain unless stock artwork has a clear geometric or visual match."""
        strong = self.match(crop, excluded_pin_ids, excluded_names)
        if strong is not None:
            return strong
        if not self._appearance_references or crop.size == 0:
            return None
        colour, layout, strengths = self._appearance_scores(crop)
        by_pin: dict[str, tuple[float, float, float, PinReference]] = {}
        for index, (reference, _signature) in enumerate(self._appearance_references):
            if (reference.pin_id in excluded_pin_ids or
                    reference.name.casefold() in excluded_names):
                continue
            score = float(strengths[index])
            previous = by_pin.get(reference.pin_id)
            if previous is None or score > previous[0]:
                by_pin[reference.pin_id] = (
                    score, float(colour[index]), float(layout[index]), reference)
        ranked = sorted(by_pin.values(), key=lambda item: item[0], reverse=True)
        if not ranked:
            return None
        score, colour_score, layout_score, reference = ranked[0]
        if (score < 0.85 or colour_score < 0.75 or layout_score < 0.82 or
                (len(ranked) > 1 and score - ranked[1][0] < 0.08)):
            return None
        return PinMatch(reference.pin_id, reference.name, 0,
                        score,
                        reference.year, reference.show)

    def rank_matches(self, crop: np.ndarray, limit: int = 20,
                     excluded_pin_ids: frozenset[str] = frozenset(),
                     excluded_names: frozenset[str] = frozenset()) -> list[PinMatch]:
        """Rank distinct pins by colour and layout for an interactive choice list."""
        if self._appearance_matrix is None or crop.size == 0 or limit <= 0:
            return []
        _colour, _layout, scores = self._appearance_scores(crop)
        best_by_pin: dict[str, PinMatch] = {}
        for (reference, _signature), score in zip(self._appearance_references, scores):
            if (reference.pin_id in excluded_pin_ids or
                    reference.name.casefold() in excluded_names):
                continue
            confidence = max(0.0, min(1.0, float(score)))
            old = best_by_pin.get(reference.pin_id)
            if old is None or confidence > (old.confidence or 0.0):
                best_by_pin[reference.pin_id] = PinMatch(
                    reference.pin_id, reference.name, 0, confidence,
                    reference.year, reference.show)
        return sorted(best_by_pin.values(),
                      key=lambda match: (-float(match.confidence or 0.0),
                                         match.name.casefold(), match.pin_id))[:limit]


def main() -> None:
    parser = argparse.ArgumentParser(description="Add a local reference image to the known-pin catalog")
    parser.add_argument("pin_id")
    parser.add_argument("name")
    parser.add_argument("image", type=Path)
    parser.add_argument("--source", required=True, help="Image source or provenance")
    parser.add_argument("--rights", required=True, help="Recorded rights or usage status")
    parser.add_argument("--db", type=Path, default=DEFAULT_CATALOG)
    args = parser.parse_args()
    PinCatalog(args.db).add_reference(args.pin_id, args.name, args.image, args.source, args.rights)
    print(f"Added reference for {args.name} ({args.pin_id}) to {args.db}")


if __name__ == "__main__":
    main()

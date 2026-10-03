"""Local reference-image catalog and conservative visual pin matching."""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


DEFAULT_CATALOG = Path(__file__).with_name("known_pins.sqlite3")


@dataclass(frozen=True)
class PinReference:
    pin_id: str
    name: str
    image: np.ndarray
    mask: np.ndarray | None = None


@dataclass(frozen=True)
class PinMatch:
    pin_id: str
    name: str
    inliers: int
    confidence: float | None = None


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
            rows = db.execute("""SELECT p.pin_id, p.name, r.image FROM pin_references r
                JOIN pins p ON p.pin_id = r.pin_id ORDER BY p.pin_id, r.id""").fetchall()
        references = []
        for pin_id, name, data in rows:
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
            references.append(PinReference(pin_id, name, rgb, mask))
        return references


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
        for reference in references:
            keypoints, descriptors = self._features(reference.image, reference.mask)
            if descriptors is not None and len(keypoints) >= 12:
                self._references.append((reference, keypoints, descriptors))
        self._index = None
        if len(self._references) > 32:
            # Train an approximate descriptor index so a large catalog does
            # not require a full comparison with every pin for every crop.
            self._index = cv2.FlannBasedMatcher(
                dict(algorithm=6, table_number=12, key_size=20, multi_probe_level=2),
                dict(checks=64))
            self._index.add([item[2] for item in self._references])
            self._index.train()

    @property
    def has_references(self) -> bool:
        return bool(self._appearance_references)

    @staticmethod
    def _appearance_vector(rgb: np.ndarray,
                           mask: np.ndarray | None = None) -> np.ndarray:
        """Compact colour signature for a best-effort, rotation-tolerant guess."""
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        histogram = cv2.calcHist([hsv], [0, 1], mask, [12, 4], [0, 180, 0, 256])
        vector = histogram.ravel().astype(np.float32)
        length = float(np.linalg.norm(vector))
        return vector / length if length else vector

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
                    reference.pin_id, reference.name, inliers, inliers / len(good))
        ranked = sorted(best_by_pin.values(), key=lambda match: match.inliers, reverse=True)
        if not ranked or (len(ranked) > 1 and ranked[0].inliers < ranked[1].inliers + 4):
            return None
        return ranked[0]

    def suggest(self, crop: np.ndarray,
                excluded_pin_ids: frozenset[str] = frozenset(),
                excluded_names: frozenset[str] = frozenset()) -> PinMatch | None:
        """Choose the strongest available catalog guess for a user-marked pin.

        The fallback score is colour-histogram similarity, not a probability.
        """
        strong = self.match(crop, excluded_pin_ids, excluded_names)
        if strong is not None:
            return strong
        if not self._appearance_references or crop.size == 0:
            return None
        query = self._appearance_vector(crop)
        ranked = ((float(np.dot(query, signature)), reference)
                  for reference, signature in self._appearance_references
                  if reference.pin_id not in excluded_pin_ids
                  and reference.name.casefold() not in excluded_names)
        best = max(ranked, key=lambda item: item[0], default=None)
        if best is None:
            return None
        similarity, reference = best
        return PinMatch(reference.pin_id, reference.name, 0,
                        max(0.0, min(1.0, similarity)))

    def rank_matches(self, crop: np.ndarray, limit: int = 20,
                     excluded_pin_ids: frozenset[str] = frozenset(),
                     excluded_names: frozenset[str] = frozenset()) -> list[PinMatch]:
        """Rank distinct catalog pins by appearance for an interactive choice list.

        Scores are colour-histogram similarity, not calibrated probabilities.
        This uses the precomputed matrix so opening a menu does not run ORB.
        """
        if self._appearance_matrix is None or crop.size == 0 or limit <= 0:
            return []
        scores = self._appearance_matrix @ self._appearance_vector(crop)
        best_by_pin: dict[str, PinMatch] = {}
        for (reference, _signature), score in zip(self._appearance_references, scores):
            if (reference.pin_id in excluded_pin_ids or
                    reference.name.casefold() in excluded_names):
                continue
            confidence = max(0.0, min(1.0, float(score)))
            old = best_by_pin.get(reference.pin_id)
            if old is None or confidence > (old.confidence or 0.0):
                best_by_pin[reference.pin_id] = PinMatch(
                    reference.pin_id, reference.name, 0, confidence)
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

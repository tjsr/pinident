"""Per-media records of direct reviews that rejected a proposed pin ID."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TextIO, cast

from boxdata import BoxData, Coordinate


FalsePositiveKind = Literal['wrong_identity', 'not_pin']


@dataclass(frozen=True)
class FalsePositiveRecord:
    review_id: str
    kind: FalsePositiveKind
    frame_index: int
    coords: Coordinate
    pin_id: str | None
    name: str


def false_positive_path(annotation_filename: str) -> Path:
    return Path(annotation_filename).with_suffix('.false_positive_pins.json')


def collect_false_positives(
        frame_boxes: dict[int, list[BoxData]]) -> tuple[FalsePositiveRecord, ...]:
    """Deduplicate direct decisions copied to linked frames by review ID."""
    seen: set[tuple[FalsePositiveKind, str]] = set()
    records: list[FalsePositiveRecord] = []
    for frame_index, boxes in sorted(frame_boxes.items()):
        for box in boxes:
            for review in box.identity_rejections:
                key: tuple[FalsePositiveKind, str] = ('wrong_identity', review.example_id)
                if key in seen:
                    continue
                seen.add(key)
                records.append(FalsePositiveRecord(
                    review.example_id, 'wrong_identity', review.frame_index,
                    review.coords, review.pin_id, review.tags[0] if review.tags else ''))
            if (box.review_state == 'rejected' and
                    box.feedback_label == 'negative' and box.feedback_id):
                key = ('not_pin', box.feedback_id)
                if key in seen:
                    continue
                seen.add(key)
                records.append(FalsePositiveRecord(
                    box.feedback_id, 'not_pin', frame_index, box.coords,
                    box.pin_id, box.tags[0] if box.tags else ''))
    return tuple(records)


def false_positive_counts(
        records: tuple[FalsePositiveRecord, ...]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        if record.pin_id:
            counts[record.pin_id] = counts.get(record.pin_id, 0) + 1
    return counts


def write_false_positives(stream: TextIO,
                          records: tuple[FalsePositiveRecord, ...]) -> None:
    json.dump({'version': 1, 'records': [
        {'review_id': record.review_id, 'kind': record.kind,
         'frame_index': record.frame_index, 'coords': record.coords,
         'pin_id': record.pin_id, 'name': record.name}
        for record in records]}, stream)


def load_false_positives(
        annotation_filename: str) -> tuple[FalsePositiveRecord, ...] | None:
    """Return None for a media file that predates the separate log."""
    path = false_positive_path(annotation_filename)
    if not path.exists():
        return None
    with path.open('r', encoding='utf-8') as stream:
        raw_data: object = json.load(stream)
    if not isinstance(raw_data, dict):
        raise ValueError(f'Unsupported false-positive log: {path}')
    data = cast(dict[str, object], raw_data)
    if data.get('version') != 1:
        raise ValueError(f'Unsupported false-positive log: {path}')
    raw_records = data.get('records')
    if not isinstance(raw_records, list):
        raise ValueError(f'Invalid false-positive log: {path}')
    records: list[FalsePositiveRecord] = []
    for raw_item in raw_records:
        if not isinstance(raw_item, dict):
            raise ValueError(f'Invalid false-positive record: {path}')
        item = cast(dict[str, object], raw_item)
        kind_value = item.get('kind')
        review_id = item.get('review_id')
        frame_index = item.get('frame_index')
        coords_value = item.get('coords')
        pin_id = item.get('pin_id')
        name = item.get('name')
        if (kind_value not in ('wrong_identity', 'not_pin') or
                not isinstance(review_id, str) or
                not isinstance(frame_index, int) or
                not isinstance(coords_value, list) or len(coords_value) != 4 or
                any(not isinstance(value, int) for value in coords_value) or
                (pin_id is not None and not isinstance(pin_id, str)) or
                not isinstance(name, str)):
            raise ValueError(f'Invalid false-positive record: {path}')
        coords = cast(list[int], coords_value)
        kind: FalsePositiveKind = ('wrong_identity' if kind_value == 'wrong_identity'
                                   else 'not_pin')
        records.append(FalsePositiveRecord(
            review_id, kind, frame_index,
            (coords[0], coords[1], coords[2], coords[3]),
            pin_id, name))
    return tuple(records)

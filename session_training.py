"""Build reviewed examples from one annotation session on a frame-reader worker."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable
from uuid import NAMESPACE_URL, uuid5

import numpy as np

from boxdata import BoxData, Coordinate
from detection_feedback import FeedbackExample, make_feedback_example


@dataclass(frozen=True)
class TrainingObservation:
    frame_index: int
    coords: Coordinate
    track_id: str
    label: str
    pin_id: str | None
    tags: tuple[str, ...]
    example_id: str


@dataclass(frozen=True)
class TrainingDelta:
    pending: list[TrainingObservation]
    removed_ids: set[str]
    stored_ids: set[str]
    active_ids: set[str]
    new_count: int
    updated_count: int


def plan_training_delta(observations: list[TrainingObservation],
                        stored: dict[str, tuple]) -> TrainingDelta:
    """Compare training-relevant metadata without opening source frames."""
    active_ids = {item.example_id for item in observations}
    if len(active_ids) != len(observations):
        raise ValueError('Duplicate feedback example ID in session annotations')
    pending = [item for item in observations if stored.get(item.example_id) != (
        item.label, item.frame_index, item.track_id, item.pin_id,
        item.coords, item.tags)]
    stored_ids = set(stored)
    return TrainingDelta(
        pending, stored_ids - active_ids, stored_ids, active_ids,
        sum(item.example_id not in stored_ids for item in pending),
        sum(item.example_id in stored_ids for item in pending))


def reviewed_observations(frame_boxes: dict[int, list[BoxData]],
                          source_path: str) -> list[TrainingObservation]:
    """Use direct reviews; propagated observations are not independent labels."""
    observations: list[TrainingObservation] = []
    identity_observations: dict[str, TrainingObservation] = {}
    legacy_negative_tracks: set[str] = set()
    explicit_negative_tracks = {box.track_id for boxes in frame_boxes.values()
                                for box in boxes if box.feedback_label == 'negative'}
    for frame_index, boxes in sorted(frame_boxes.items()):
        for box in boxes:
            label = box.feedback_label
            if label is None and box.review_state == 'confirmed':
                label = 'positive'
            elif (label is None and box.review_state == 'rejected' and
                  box.source == 'automatic' and
                  box.track_id not in explicit_negative_tracks and
                  box.track_id not in legacy_negative_tracks):
                label = 'negative'
                legacy_negative_tracks.add(box.track_id)
            if label in ('positive', 'negative'):
                example_id = box.feedback_id or uuid5(
                    NAMESPACE_URL,
                    f'{source_path}|{frame_index}|{box.track_id}|{label}').hex
                observations.append(TrainingObservation(
                    frame_index, box.coords, box.track_id, label,
                    box.pin_id if label == 'positive' and box.identity_confirmed else None,
                    tuple(tag for tag in box.tags or [] if tag.strip()) if label == 'positive' else (),
                    example_id))
            for review in box.identity_rejections:
                source_frame = review.frame_index if review.frame_index >= 0 else frame_index
                observation = TrainingObservation(
                    source_frame, review.coords, box.track_id, 'wrong_identity',
                    review.pin_id, review.tags, review.example_id)
                # Propagated copies share an ID. Prefer the original frame's
                # box when it is present so track provenance stays stable.
                if (review.example_id not in identity_observations or
                        (frame_index == source_frame and box.coords == review.coords)):
                    identity_observations[review.example_id] = observation
    observations.extend(identity_observations.values())
    return observations


def build_training_examples(
        observations: list[TrainingObservation], source_path: str,
        read_frame: Callable[[int], np.ndarray | None],
        progress: Callable[[int, int, float | None], None],
) -> tuple[list[FeedbackExample], int]:
    """Decode once per frame, reporting completed observations and ETA seconds."""
    examples: list[FeedbackExample] = []
    skipped = 0
    started = perf_counter()
    current_frame = -1
    image = None
    encoded_crops: dict[Coordinate, bytes] = {}
    total = len(observations)
    progress(0, total, None)
    for completed, item in enumerate(observations, 1):
        if item.frame_index != current_frame:
            current_frame = item.frame_index
            image = read_frame(current_frame)
            encoded_crops.clear()
        try:
            if image is None:
                raise ValueError('Source frame unavailable')
            crop_png = encoded_crops.get(item.coords)
            if crop_png is None:
                encoded = make_feedback_example(
                    item.example_id, item.label, image, item.coords, source_path,
                    item.frame_index, item.track_id, item.pin_id, item.tags)
                crop_png = encoded.crop_png
                encoded_crops[item.coords] = crop_png
                examples.append(encoded)
            else:
                examples.append(FeedbackExample(
                    item.example_id, item.label, source_path, item.frame_index,
                    item.track_id, item.pin_id, item.coords, crop_png, item.tags))
        except ValueError:
            skipped += 1
        elapsed = perf_counter() - started
        remaining = elapsed / completed * (total - completed) if completed < total else 0.0
        progress(completed, total, remaining)
    return examples, skipped

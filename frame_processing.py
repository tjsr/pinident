"""CPU-bound frame analysis without wx or access to the UI video capture."""

from __future__ import annotations

import json
from dataclasses import dataclass
from time import perf_counter
from typing import Callable

import cv2
import numpy as np

from boxdata import BoxData, Coordinate, has_specific_identity
from detection_feedback import FeedbackModel
from frame_presence import marked_pin_count
from logutil import getLog
from pin_catalog import PinMatcher
from pin_detection import (add_adjacent_tracked_boxes, add_new_detections,
                           add_unique_boxes,
                           associate_frame_boxes, detect_pin_boxes,
                           detect_pin_boxes_relaxed,
                           filter_new_candidates_by_pin_size, identify_candidates,
                           inherit_corrected_bounds,
                           inherit_confirmed_catalog_matches,
                           match_existing_candidates, match_rejected_candidates)

LOW_RESOLUTION_SIDE = 640
_SavedBox = tuple[Coordinate, Coordinate | None, Coordinate, Coordinate | None]


@dataclass
class FrameJob:
    index: int
    previous_index: int | None
    previous_boxes: list[BoxData]
    existing_boxes: list[BoxData]
    rejected_next: list[BoxData]
    displayed_frame: np.ndarray | None
    previous_frame: np.ndarray | None
    feedback_model: FeedbackModel
    matcher: PinMatcher
    expected_pins: int = 0
    allow_empty_fallback: bool = False
    allow_motion_fallback: bool = False
    catalog_best_effort: bool = False
    diagnostic_run_id: str | None = None
    low_resolution: bool = False


@dataclass
class FrameResult:
    index: int
    frame: np.ndarray
    boxes: list[BoxData]
    existing_count: int
    total_ms: float
    primary_count: int = 0
    relaxed_count: int = 0
    catalog_best_effort: bool = False
    diagnostic_run_id: str | None = None


def _scene_changed(previous: np.ndarray, current: np.ndarray) -> bool:
    """Ignore static frames and small sensor noise before a playback search."""
    if previous.shape != current.shape:
        return False
    scale = min(1.0, 320 / max(current.shape[:2]))

    def blurred_gray(frame: np.ndarray) -> np.ndarray:
        small = (cv2.resize(frame, None, fx=scale, fy=scale,
                            interpolation=cv2.INTER_AREA) if scale < 1 else frame)
        return cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_RGB2GRAY),
                                (5, 5), 0)

    difference = cv2.absdiff(blurred_gray(previous), blurred_gray(current))
    return np.count_nonzero(difference >= 10) >= 0.02 * difference.size


def _near_frame_edge(box: BoxData, frame: np.ndarray) -> bool:
    x, y, width, height = box.coords
    frame_height, frame_width = frame.shape[:2]
    return (x < 0.30 * frame_width or x + width > 0.70 * frame_width or
            y < 0.30 * frame_height or y + height > 0.70 * frame_height)


def _scale_coords(coords: Coordinate, factor: float, minimum: int = 0) -> Coordinate:
    x, y, w, h = coords
    return (round(x * factor), round(y * factor),
            max(minimum, round(w * factor)), max(minimum, round(h * factor)))


def _shrink_boxes(boxes: list[BoxData], factor: float) -> dict[int, _SavedBox]:
    """Scale boxes into working space, remembering the exact source values."""
    saved: dict[int, _SavedBox] = {}
    for box in boxes:
        if id(box) in saved:
            continue
        original, adjustment = box.coords, box.bounds_adjustment
        box.coords = _scale_coords(original, factor, 1)
        if adjustment is not None:
            box.bounds_adjustment = _scale_coords(adjustment, factor)
        saved[id(box)] = (original, adjustment, box.coords, box.bounds_adjustment)
    return saved


def _expand_boxes(boxes: list[BoxData], saved: dict[int, _SavedBox], factor: float,
                  width: int, height: int) -> None:
    """Return working-space boxes to source pixels; untouched boxes keep exact values."""
    for box in boxes:
        entry = saved.get(id(box))
        if entry is not None and box.coords == entry[2]:
            box.coords = entry[0]
        else:
            x, y, w, h = _scale_coords(box.coords, 1 / factor, 1)
            x, y = min(x, width - 1), min(y, height - 1)
            box.coords = (x, y, min(w, width - x), min(h, height - y))
        if entry is not None and box.bounds_adjustment == entry[3]:
            box.bounds_adjustment = entry[1]
        elif box.bounds_adjustment is not None:
            box.bounds_adjustment = _scale_coords(box.bounds_adjustment, 1 / factor)


def process_frame(job: FrameJob, read_frame: Callable[[int], np.ndarray | None]) -> FrameResult:
    """Analyze an immutable annotation snapshot; the GUI commits the result."""
    started = perf_counter()
    frame = job.displayed_frame if job.displayed_frame is not None else read_frame(job.index)
    if frame is None:
        raise ValueError(f'Could not decode frame {job.index}')
    decoded = perf_counter()
    full_frame = frame
    factor = (min(1.0, LOW_RESOLUTION_SIDE / max(frame.shape[:2]))
              if job.low_resolution else 1.0)
    saved: dict[int, _SavedBox] = {}

    def shrink(image: np.ndarray) -> np.ndarray:
        if factor >= 1.0:
            return image
        return cv2.resize(image, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)

    if factor < 1.0:
        frame = shrink(full_frame)
        saved = _shrink_boxes([*job.previous_boxes, *job.existing_boxes,
                               *job.rejected_next], factor)
    proposals = detect_pin_boxes(frame, feedback_model=job.feedback_model)
    detected = perf_counter()
    previous = None
    if job.previous_index is not None and (job.previous_boxes or job.allow_motion_fallback):
        source_previous = (job.previous_frame if job.previous_frame is not None
                           else read_frame(job.previous_index))
        previous = shrink(source_previous) if source_previous is not None else None
    if job.previous_index is not None and job.previous_boxes:
        if previous is not None:
            for source, target in match_rejected_candidates(
                    previous, frame, job.previous_boxes, job.existing_boxes):
                target.track_id = source.track_id
                target.review_state = 'rejected'
                target.identity_confirmed = False
            for source, target in match_existing_candidates(
                    previous, frame, job.previous_boxes, job.existing_boxes):
                if (source.review_state == 'unconfirmed' and target.pin_id and
                        target.pin_id != source.pin_id):
                    continue
                target.track_id = source.track_id
                if source.review_state in ('confirmed', 'inherited'):
                    target.review_state = 'inherited'
                if (source.identity_confirmed or
                        not has_specific_identity(target)):
                    target.pin_id = source.pin_id
                    if not (target.source == 'user' and has_specific_identity(target)):
                        target.tags = (list(source.tags or [])
                                       if has_specific_identity(source) else [])
                    target.match_confidence = source.match_confidence
                    target.identity_confirmed = source.identity_confirmed
                inherit_corrected_bounds(previous, frame, source, target)
            by_track = {box.track_id: box for box in job.previous_boxes}
            for target in job.existing_boxes:
                source = by_track.get(target.track_id)
                if source is None or target.source != 'automatic' or target.review_state != 'unconfirmed':
                    continue
                if source.review_state == 'rejected':
                    target.review_state = 'rejected'
                    target.identity_confirmed = False
                elif source.review_state in ('confirmed', 'inherited'):
                    target.review_state = 'inherited'
                if source.review_state != 'rejected' and (
                        source.identity_confirmed or
                        not has_specific_identity(target)):
                    target.pin_id = source.pin_id
                    target.tags = (list(source.tags or [])
                                   if has_specific_identity(source) else [])
                    target.match_confidence = source.match_confidence
                    target.identity_confirmed = source.identity_confirmed
        proposed = (associate_frame_boxes(previous, frame, job.previous_boxes, proposals)
                    if previous is not None else add_new_detections([], proposals))
    elif job.rejected_next:
        following_source = read_frame(job.index + 1)
        following = shrink(following_source) if following_source is not None else None
        if following is not None:
            for source, target in match_rejected_candidates(
                    following, frame, job.rejected_next, job.existing_boxes):
                target.track_id = source.track_id
                target.review_state = 'rejected'
                target.identity_confirmed = False
        proposed = (associate_frame_boxes(following, frame, job.rejected_next, proposals)
                    if following is not None else add_new_detections([], proposals))
    else:
        proposed = add_new_detections([], proposals)
    unfiltered_count = len(proposed)
    proposed = filter_new_candidates_by_pin_size(frame, job.existing_boxes, proposed)
    size_filtered = unfiltered_count - len(proposed)
    if previous is not None and job.previous_boxes:
        tracked = [box for box in proposed if box.prev_box_id is not None]
        fresh = [box for box in proposed if box.prev_box_id is None]
        boxes = add_adjacent_tracked_boxes(
            previous, frame, job.previous_boxes, job.existing_boxes, tracked)
        boxes = add_unique_boxes(boxes, fresh)
    else:
        boxes = add_unique_boxes(job.existing_boxes, proposed)
    associated = perf_counter()
    catalog_inherited = 0
    if factor >= 1.0:
        identify_candidates(frame, boxes, job.matcher,
                            best_effort=job.catalog_best_effort,
                            feedback_model=job.feedback_model)
        if job.previous_index is not None:
            catalog_inherited += inherit_confirmed_catalog_matches(job.previous_boxes, boxes)
    matched = perf_counter()
    relaxed_count = 0
    full_fallback = (marked_pin_count(boxes) < job.expected_pins or
                     (job.allow_empty_fallback and not any(
                         box.review_state != 'rejected' for box in boxes)))
    motion_fallback = (job.allow_motion_fallback and previous is not None and
                       _scene_changed(previous, frame))
    fallback_detect_ms = 0.0
    fallback_catalog_ms = 0.0
    if full_fallback or motion_fallback:
        prior_box_ids = {id(box) for box in boxes}
        relaxed = detect_pin_boxes_relaxed(frame, feedback_model=job.feedback_model)
        relaxed_count = len(relaxed)
        extra = filter_new_candidates_by_pin_size(
            frame, boxes, add_new_detections([], relaxed))
        size_filtered += len(relaxed) - len(extra)
        if motion_fallback and not full_fallback:
            added = 0
            for candidate in extra:
                if not _near_frame_edge(candidate, frame):
                    continue
                expanded = add_unique_boxes(boxes, [candidate])
                if len(expanded) > len(boxes):
                    boxes = expanded
                    added += 1
                    if added >= 6:
                        break
        else:
            boxes = add_unique_boxes(boxes, extra)
        # The first pass already matched existing boxes. Only newly admitted
        # fallback proposals need the expensive catalog lookup.
        fallback_boxes = [box for box in boxes if id(box) not in prior_box_ids]
        fallback_catalog_started = perf_counter()
        fallback_detect_ms = (fallback_catalog_started - matched) * 1000
        if factor >= 1.0:
            identify_candidates(frame, fallback_boxes, job.matcher,
                                best_effort=job.catalog_best_effort,
                                feedback_model=job.feedback_model,
                                size_context=boxes)
            if job.previous_index is not None:
                catalog_inherited += inherit_confirmed_catalog_matches(job.previous_boxes, boxes)
        fallback_catalog_ms = (perf_counter() - fallback_catalog_started) * 1000
    if factor < 1.0:
        height, width = full_frame.shape[:2]
        everything = {id(box): box for box in (*job.previous_boxes, *job.existing_boxes,
                                               *job.rejected_next, *boxes)}
        _expand_boxes(list(everything.values()), saved, factor, width, height)
        frame = full_frame
        catalog_started = perf_counter()
        identify_candidates(frame, boxes, job.matcher,
                            best_effort=job.catalog_best_effort,
                            feedback_model=job.feedback_model)
        if job.previous_index is not None:
            catalog_inherited += inherit_confirmed_catalog_matches(job.previous_boxes, boxes)
        fallback_catalog_ms += (perf_counter() - catalog_started) * 1000
    finished = perf_counter()
    getLog().info(
        'frame=%d worker decode=%.1fms detect=%.1fms track=%.1fms catalog=%.1fms fallback=%.1fms fallback_detect=%.1fms fallback_catalog=%.1fms total=%.1fms proposals=%d relaxed=%d motion_search=%s size_filtered=%d boxes=%d catalog_inherited=%d',
        job.index, (decoded-started)*1000, (detected-decoded)*1000,
        (associated-detected)*1000, (matched-associated)*1000,
        (finished-matched)*1000, fallback_detect_ms, fallback_catalog_ms,
        (finished-started)*1000,
        len(proposals), relaxed_count, motion_fallback, size_filtered, len(boxes), catalog_inherited)
    if job.diagnostic_run_id is not None:
        diagnostic = {
            'run': job.diagnostic_run_id,
            'frame': job.index,
            'source_size': [int(frame.shape[1]), int(frame.shape[0])],
            'primary_xywh': [list(rect) for rect in proposals],
            'relaxed_xywh': [list(rect) for rect in relaxed] if full_fallback or motion_fallback else [],
            'full_fallback': full_fallback,
            'motion_fallback': motion_fallback,
            'size_filtered': size_filtered,
            'existing_count': len(job.existing_boxes),
            'result_count': len(boxes),
        }
        getLog().info('PINIDENT_DETECTION worker %s',
                      json.dumps(diagnostic, separators=(',', ':')))
    return FrameResult(job.index, frame, boxes, len(job.existing_boxes),
                       (finished-started)*1000, len(proposals), relaxed_count,
                       job.catalog_best_effort, job.diagnostic_run_id)

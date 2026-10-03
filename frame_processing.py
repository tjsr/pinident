"""CPU-bound frame analysis without wx or access to the UI video capture."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable

import numpy as np

from boxdata import BoxData, has_specific_identity
from detection_feedback import FeedbackModel
from frame_presence import marked_pin_count
from logutil import getLog
from pin_catalog import PinMatcher
from pin_detection import (add_new_detections, add_unique_boxes,
                           associate_frame_boxes, detect_pin_boxes,
                           detect_pin_boxes_relaxed, identify_candidates,
                           inherit_confirmed_catalog_matches,
                           match_existing_candidates, match_rejected_candidates)


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
    catalog_best_effort: bool = False


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


def process_frame(job: FrameJob, read_frame: Callable[[int], np.ndarray | None]) -> FrameResult:
    """Analyze an immutable annotation snapshot; the GUI commits the result."""
    started = perf_counter()
    frame = job.displayed_frame if job.displayed_frame is not None else read_frame(job.index)
    if frame is None:
        raise ValueError(f'Could not decode frame {job.index}')
    decoded = perf_counter()
    proposals = detect_pin_boxes(frame, feedback_model=job.feedback_model)
    detected = perf_counter()
    if job.previous_index is not None and job.previous_boxes:
        previous = job.previous_frame if job.previous_frame is not None else read_frame(job.previous_index)
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
        following = read_frame(job.index + 1)
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
    boxes = add_unique_boxes(job.existing_boxes, proposed)
    associated = perf_counter()
    identify_candidates(frame, boxes, job.matcher,
                        best_effort=job.catalog_best_effort)
    catalog_inherited = 0
    if job.previous_index is not None:
        catalog_inherited += inherit_confirmed_catalog_matches(job.previous_boxes, boxes)
    matched = perf_counter()
    relaxed_count = 0
    if (marked_pin_count(boxes) < job.expected_pins or
            (job.allow_empty_fallback and not any(
                box.review_state != 'rejected' for box in boxes))):
        relaxed = detect_pin_boxes_relaxed(frame, feedback_model=job.feedback_model)
        relaxed_count = len(relaxed)
        boxes = add_unique_boxes(boxes, add_new_detections([], relaxed))
        identify_candidates(frame, boxes, job.matcher,
                            best_effort=job.catalog_best_effort)
        if job.previous_index is not None:
            catalog_inherited += inherit_confirmed_catalog_matches(job.previous_boxes, boxes)
    finished = perf_counter()
    getLog().info(
        'frame=%d worker decode=%.1fms detect=%.1fms track=%.1fms catalog=%.1fms fallback=%.1fms total=%.1fms proposals=%d relaxed=%d boxes=%d catalog_inherited=%d',
        job.index, (decoded-started)*1000, (detected-decoded)*1000,
        (associated-detected)*1000, (matched-associated)*1000,
        (finished-matched)*1000, (finished-started)*1000,
        len(proposals), relaxed_count, len(boxes), catalog_inherited)
    return FrameResult(job.index, frame, boxes, len(job.existing_boxes),
                       (finished-started)*1000, len(proposals), relaxed_count,
                       job.catalog_best_effort)

import json
import os
import tempfile
from collections import defaultdict, deque
from contextlib import contextmanager
from copy import copy, deepcopy
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
from threading import Event
from time import perf_counter
from typing import Callable, Iterator, List, Dict
from uuid import NAMESPACE_URL, uuid4, uuid5

import cv2
import numpy as np
import wx

from boxdata import (BoxData, Coordinate, IdentityRejection, is_placeholder_tag,
                     has_specific_identity, rejects_identity, same_specific_name)
from classifier_review import ReviewArea, saved_review_areas
from annotation_history import BoxSnapshot, HistoryEntry, make_entry, snapshot
from detection_feedback import (FeedbackExample, FeedbackModel, FeedbackStore,
                                make_feedback_example)
from frame_processing import FrameJob, FrameResult, process_frame
from frame_presence import FramePresence, checkbox_values, marked_pin_count
from controlspanel import ControlsPanel
from settings_dialog import (SettingsDialog, load_detection_resolution,
                             save_detection_resolution)
from shutdown_progress import ShutdownProgressDialog
from trained_detection_model import (TrainedDetectionModel, examples_digest,
                                     metadata_path, model_path_for_feedback,
                                     PortableDetectionModel)
from cloud_sync import (CloudConfig, CloudSyncError, cloud_cursor, fetch_model, fetch_review_page,
                        apply_reviews, publish_model)
from events.BoxSelectedEvent import BoxSelectedEvent
from events.events import EVT_BOX_SELECTED
from controls.imagepanel import ImagePanel
from controls.classifierreviewdialog import ClassifierReviewDialog
from logutil import getLog
from controls.markerpanel import MarkerPanel  # Adjust import as needed
from controls.tagpanel import TagPanel
from pin_catalog import PinCatalog, PinMatch, PinMatcher
from session_training import (LINKED_POSITIVE_PREFIX, TrainingDelta,
                              TrainingObservation, build_training_examples,
                              observation_metadata, plan_training_delta,
                              reviewed_observations)
from pin_detection import (add_adjacent_tracked_boxes, add_new_detections,
                           add_unique_boxes, associate_frame_boxes,
                           can_link_saved_target,
                           consolidate_overlapping_boxes,
                           detect_pin_boxes, detect_pin_boxes_relaxed,
                           filter_new_candidates_by_pin_size,
                           identify_candidates, inherit_confirmed_catalog_matches,
                           inherit_corrected_bounds,
                           needs_catalog_match,
                           match_existing_candidates,
                           match_rejected_candidates)

def remove_empty(tags: List[str]) -> List[str]:
    """Remove empty tags from the list."""
    return [tag for tag in tags if tag.strip()]

def save_boxes_to_stream(stream, frame_boxes: dict[int, list[BoxData]],
                         presence: dict[int, FramePresence] | None = None,
                         progress: Callable[[int], None] | None = None) -> None:
    # frame_boxes: {frame_number: [BoxData, ...]}
    serializable = {}
    total_frames = len(frame_boxes)
    for position, (frame, boxes) in enumerate(frame_boxes.items(), 1):
        serializable[frame] = [
            {"coords": box.coords, "tags": remove_empty(box.tags), "source": box.source,
             "prev_box_id": box.prev_box_id, "track_id": box.track_id,
             "review_state": box.review_state,
             **({"bounds_adjustment": box.bounds_adjustment}
                if box.bounds_adjustment is not None else {}),
             **({"bounds_direct": True} if box.bounds_direct and box.source != 'user'
                else {}),
             **({"feedback_id": box.feedback_id, "feedback_label": box.feedback_label}
                if box.feedback_id and box.feedback_label else {}),
             **({"pin_id": box.pin_id} if box.pin_id else {}),
             **({"match_confidence": box.match_confidence}
                if box.match_confidence is not None else {}),
             "identity_confirmed": box.identity_confirmed,
             "identity_inferred": box.identity_inferred,
             **({"identity_rejections": [
                 {"example_id": item.example_id, "label": "wrong_identity",
                  "pin_id": item.pin_id,
                  "tags": list(item.tags), "coords": item.coords,
                  "frame_index": item.frame_index}
                 for item in box.identity_rejections]}
                if box.identity_rejections else {})}
            for box in boxes
        ]
        if progress:
            progress(round(85 * position / max(1, total_frames)))
    if presence:
        assertions = {str(frame): {'contains_pin': flags.pin_asserted,
                                    'contains_set': flags.set_asserted}
                      for frame, flags in presence.items()
                      if flags.pin_asserted or flags.set_asserted}
        if assertions:
            serializable['_frame_presence'] = assertions
    json.dump(serializable, stream)
    if progress:
        progress(90)

def save_boxes_to_file(filename: str, frame_boxes: dict[int, list[BoxData]],
                       presence: dict[int, FramePresence] | None = None,
                       progress: Callable[[int], None] | None = None) -> None:
    """Replace the sidecar only after the complete JSON has reached disk."""
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
                mode='w', encoding='utf-8', delete=False,
                dir=os.path.dirname(os.path.abspath(filename)),
                prefix='.pinident-', suffix='.tmp') as stream:
            temporary_name = stream.name
            if progress is None:
                save_boxes_to_stream(stream, frame_boxes, presence)
            else:
                save_boxes_to_stream(stream, frame_boxes, presence, progress)
            stream.flush()
            os.fsync(stream.fileno())
        if progress:
            progress(95)
        os.replace(temporary_name, filename)
        if progress:
            progress(100)
    finally:
        if temporary_name is not None and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def save_review_feedback(path: str, source: str,
                         changes: dict[str, FeedbackExample | None]) -> None:
    """Commit reviewed crops with a worker-owned SQLite connection."""
    store = FeedbackStore(path)
    try:
        store.apply_source_delta(
            source, [example for example in changes.values() if example is not None],
            {example_id for example_id, example in changes.items() if example is None})
    finally:
        store.close()

def merge_duplicate_boxes(boxes: List[BoxData]) -> List[BoxData]:
    """Merge boxes with the same coordinates and tags."""
    merged: Dict[Coordinate, BoxData] = {}
    for box in boxes:
        key = box.coords # , tuple(sorted(box.tags)))
        if key not in merged:
            merged[key] = BoxData(coords=box.coords, tags=copy(box.tags), source=box.source,
                                  track_id=box.track_id, pin_id=box.pin_id,
                                  review_state=box.review_state,
                                  feedback_id=box.feedback_id,
                                  feedback_label=box.feedback_label,
                                  match_confidence=box.match_confidence,
                                  identity_confirmed=box.identity_confirmed,
                                  identity_inferred=box.identity_inferred,
                                  identity_rejections=box.identity_rejections,
                                  bounds_adjustment=box.bounds_adjustment,
                                  bounds_direct=box.bounds_direct)
        else:
            merged[key].tags.extend(box.tags)
            merged[key].tags = remove_empty(merged[key].tags)

        merged[key].prev_box_id = box.prev_box_id

    # Remove duplicates in tags
    for box in merged.values():
        box.tags = list(set(box.tags))  # Remove duplicate tags

    return list(merged.values())

def load_boxes_from_stream(stream) -> dict[int, list[BoxData]]:
    data = json.load(stream)
    result = dict[int, list[BoxData]]()
    for frame, boxes in data.items():
        if frame == '_frame_presence':
            continue
        f: int = int(frame)
        result[f] = []
        for jsonbox in boxes:
            box = BoxData(
                    coords=tuple(jsonbox["coords"]),
                    tags=remove_empty(jsonbox["tags"]),
                    source=jsonbox.get("source", "automatic"),
                    track_id=jsonbox.get("track_id"),
                    pin_id=jsonbox.get("pin_id"),
                    review_state=jsonbox.get("review_state", "unconfirmed"),
                    feedback_id=jsonbox.get("feedback_id"),
                    feedback_label=jsonbox.get("feedback_label"),
                    match_confidence=jsonbox.get("match_confidence"),
                    identity_confirmed=jsonbox.get("identity_confirmed"),
                    identity_inferred=jsonbox.get("identity_inferred", False),
                    bounds_adjustment=(tuple(jsonbox['bounds_adjustment'])
                                       if 'bounds_adjustment' in jsonbox else None),
                    bounds_direct=jsonbox.get('bounds_direct'),
                    identity_rejections=tuple(IdentityRejection(
                        item['example_id'], item.get('pin_id'),
                        tuple(item.get('tags', [])), tuple(item['coords']),
                        int(item.get('frame_index', f)))
                        for item in jsonbox.get('identity_rejections', []))
                )
            box.prev_box_id = jsonbox.get("prev_box_id", None)
            result[f].append(box)

    return result


def load_presence_from_file(filename: str) -> dict[int, FramePresence]:
    with open(filename, 'r', encoding='utf-8') as stream:
        data = json.load(stream)
    raw = data.get('_frame_presence', {})
    if not isinstance(raw, dict):
        return {}
    return {int(frame): FramePresence(bool(value.get('contains_pin')),
                                      bool(value.get('contains_set')))
            for frame, value in raw.items() if isinstance(value, dict)}


def propagate_confirmations(frame_boxes: dict[int, list[BoxData]]) -> None:
    """Mark associated detections orange when their track has a confirmed frame."""
    propagate_identity_rejections(frame_boxes)
    # Inherited identity trust is derived from a direct confirmation. Rebuild it
    # after edits so a corrected direct label cannot leave stale trusted copies.
    for boxes in frame_boxes.values():
        for box in boxes:
            if box.review_state != 'confirmed' or box.identity_inferred:
                box.identity_confirmed = False
    propagate_track_identity(frame_boxes)
    confirmed_tracks = {box.track_id for boxes in frame_boxes.values()
                        for box in boxes if box.review_state == 'confirmed'}
    for boxes in frame_boxes.values():
        for box in boxes:
            if box.review_state == 'rejected':
                continue
            if box.track_id in confirmed_tracks and box.review_state != 'confirmed':
                box.review_state = 'inherited'
            elif box.review_state == 'inherited' and box.track_id not in confirmed_tracks:
                box.review_state = 'unconfirmed'


def link_adjacent_direct_confirmations(
    index: int,
    num_frames: int,
    frame_boxes: dict[int, list[BoxData]],
    frame_at: Callable[[int], np.ndarray | None],
    cancelled: Callable[[], bool],
) -> bool:
    """Join visually matching direct reviews while retaining each frame's decision."""
    current = [box for box in frame_boxes.get(index, [])
               if box.review_state == 'confirmed']
    if not current:
        return True
    linked = False
    for neighbour in (index - 1, index + 1):
        if cancelled():
            return False
        if not 0 <= neighbour < num_frames:
            continue
        adjacent = [box for box in frame_boxes.get(neighbour, [])
                    if box.review_state == 'confirmed']
        if not adjacent:
            continue
        source_frame, target_frame = frame_at(neighbour), frame_at(index)
        if source_frame is None or target_frame is None:
            continue
        for source, target in match_existing_candidates(
                source_frame, target_frame, adjacent, current):
            if (source.review_state != 'confirmed' or target.review_state != 'confirmed'
                    or source.track_id == target.track_id):
                continue
            members = [box for boxes in frame_boxes.values() for box in boxes
                       if box.track_id in (source.track_id, target.track_id)]
            verified = [box for box in members if box.review_state == 'confirmed'
                        and box.identity_confirmed and not box.identity_inferred]
            if any((left.pin_id and right.pin_id and left.pin_id != right.pin_id)
                   or (not (left.pin_id and right.pin_id) and
                       not same_specific_name(left, right))
                   for left in verified for right in verified):
                continue
            old_track = source.track_id
            for member in members:
                if member.track_id == old_track:
                    member.track_id = target.track_id
                    linked = True
    if linked:
        propagate_confirmations(frame_boxes)
    return True


def propagate_identity_rejections(frame_boxes: dict[int, list[BoxData]]) -> None:
    """Carry a direct wrong-name review to linked observations of that object."""
    tracks: dict[str, list[BoxData]] = {}
    for boxes in frame_boxes.values():
        for box in boxes:
            if box.review_state != 'rejected':
                tracks.setdefault(box.track_id, []).append(box)
    for members in tracks.values():
        reviews = {item.example_id: item for box in members
                   for item in box.identity_rejections}
        if not reviews:
            continue
        for box in members:
            if (box.review_state == 'confirmed' and box.identity_confirmed and
                    not box.identity_inferred):
                continue
            box.identity_rejections = tuple(reviews.values())
            if rejects_identity(box, box.pin_id, box.tags):
                box.pin_id = None
                box.tags = []
                box.match_confidence = None
                box.identity_confirmed = False
                box.identity_inferred = False


def propagate_track_identity(frame_boxes: dict[int, list[BoxData]]) -> None:
    """Carry a verified identity to compatible observations on the same track."""
    tracks: dict[str, list[BoxData]] = {}
    for boxes in frame_boxes.values():
        for box in boxes:
            if box.review_state != 'rejected':
                tracks.setdefault(box.track_id, []).append(box)
    for boxes in tracks.values():
        identified = [box for box in boxes if has_specific_identity(box)]
        verified = [box for box in identified if box.identity_confirmed
                    and not box.identity_inferred
                    and box.review_state == 'confirmed']
        choices = verified or identified
        identities = {(box.pin_id, tuple(tag for tag in box.tags or [] if tag.strip()))
                      for box in choices}
        if len(identities) != 1:
            continue
        source = choices[0]
        for box in boxes:
            if rejects_identity(box, source.pin_id, source.tags):
                continue
            if box.identity_confirmed or (box.review_state == 'confirmed'
                                          and not box.identity_inferred):
                continue
            same_name = same_specific_name(source, box)
            if (same_name and source.identity_confirmed and box.source in ('automatic', 'user')):
                box.pin_id = source.pin_id
                box.match_confidence = source.match_confidence
                box.identity_confirmed = True
                box.identity_inferred = box.review_state == 'confirmed'
            elif (box.source in ('automatic', 'user') and
                  (box.source == 'automatic' or source.identity_confirmed) and
                  (not has_specific_identity(box) or
                   (box.source == 'automatic' and source.identity_confirmed
                    and box.pin_id))):
                box.pin_id = source.pin_id
                box.tags = list(source.tags or [])
                box.match_confidence = source.match_confidence
                box.identity_confirmed = source.identity_confirmed
                box.identity_inferred = box.review_state == 'confirmed' and source.identity_confirmed


def propagate_rejections(frame_boxes: dict[int, list[BoxData]]) -> None:
    """Keep rejected automatic tracks hidden across already annotated frames."""
    rejected_tracks = {box.track_id for boxes in frame_boxes.values()
                       for box in boxes if box.review_state == 'rejected'}
    manual_tracks = {box.track_id for boxes in frame_boxes.values()
                     for box in boxes if box.source == 'user'}
    suppress_tracks = rejected_tracks - manual_tracks
    for boxes in frame_boxes.values():
        for box in boxes:
            if (box.track_id in suppress_tracks
                    and box.source == 'automatic'
                    and box.review_state != 'rejected'):
                box.review_state = 'rejected'
                box.identity_confirmed = False
                box.identity_inferred = False
                if box.feedback_label == 'positive':
                    box.feedback_id = None
                    box.feedback_label = None


def reconcile_saved_rejections(index: int, num_frames: int,
                               frame_boxes: dict[int, list[BoxData]],
                               frame_at: Callable[[int], np.ndarray | None],
                               cancelled: Callable[[], bool]) -> bool:
    """Follow a removed automatic candidate through contiguous saved frames."""
    if not frame_boxes.get(index):
        return True
    first = last = index
    while first > 0 and frame_boxes.get(first - 1):
        first -= 1
    while last + 1 < num_frames and frame_boxes.get(last + 1):
        last += 1

    members_by_track: dict[str, list[tuple[int, BoxData]]] = defaultdict(list)
    for frame_index, boxes in frame_boxes.items():
        for box in boxes:
            members_by_track[box.track_id].append((frame_index, box))
    protected = {track for track, members in members_by_track.items()
                 if any(box.source == 'user' or box.review_state == 'confirmed'
                        for _, box in members)}
    queue = deque((frame_index, box)
                  for frame_index in range(first, last + 1)
                  for box in frame_boxes[frame_index]
                  if box.review_state == 'rejected' and box.track_id not in protected)
    while queue:
        if cancelled():
            return False
        source_index, source = queue.popleft()
        for target_index in (source_index - 1, source_index + 1):
            if target_index < first or target_index > last:
                continue
            targets = [box for box in frame_boxes[target_index]
                       if box.source == 'automatic' and box.review_state == 'unconfirmed'
                       and box.track_id not in protected]
            if not targets:
                continue
            source_frame, target_frame = frame_at(source_index), frame_at(target_index)
            if source_frame is None or target_frame is None:
                continue
            for _, target in match_rejected_candidates(
                    source_frame, target_frame, [source], targets):
                old_track = target.track_id
                if old_track == source.track_id or old_track in protected:
                    continue
                members = members_by_track.pop(old_track, [])
                members_by_track[source.track_id].extend(members)
                for member_index, member in members:
                    member.track_id = source.track_id
                    if member.review_state == 'unconfirmed':
                        member.review_state = 'rejected'
                        member.identity_confirmed = False
                        queue.append((member_index, member))
    return True


def reconcile_saved_frames(index: int, num_frames: int,
                           frame_boxes: dict[int, list[BoxData]],
                           read_frame: Callable[[int], np.ndarray | None],
                           cancelled: Callable[[], bool] = lambda: False,
                           reference_index: int | None = None) -> bool:
    """Link a saved chain, using only supplied data and decoder."""
    frames: dict[int, np.ndarray | None] = {}

    def frame_at(frame_index: int) -> np.ndarray | None:
        if frame_index not in frames:
            frames[frame_index] = read_frame(frame_index)
        return frames[frame_index]

    propagate_rejections(frame_boxes)
    propagate_confirmations(frame_boxes)
    if not reconcile_saved_rejections(index, num_frames, frame_boxes,
                                      frame_at, cancelled):
        return False
    propagate_confirmations(frame_boxes)
    if not link_adjacent_direct_confirmations(
            index, num_frames, frame_boxes, frame_at, cancelled):
        return False
    if (reference_index is not None and abs(reference_index - index) == 1
            and not cancelled()):
        if inherit_confirmed_catalog_matches(frame_boxes.get(reference_index, []),
                                             frame_boxes.get(index, [])):
            propagate_confirmations(frame_boxes)
        sources = [box for box in frame_boxes.get(reference_index, [])
                   if box.review_state != 'rejected' and
                   (box.review_state in ('confirmed', 'inherited') or
                    has_specific_identity(box) or
                    box.bounds_adjustment is not None or
                    any(target.track_id == box.track_id and
                        target.bounds_adjustment is not None and
                        not target.bounds_direct
                        for target in frame_boxes.get(index, [])))]
        if sources:
            source_frame, target_frame = frame_at(reference_index), frame_at(index)
            if source_frame is not None and target_frame is not None:
                for source, target in match_existing_candidates(
                        source_frame, target_frame, sources,
                        frame_boxes.get(index, [])):
                    if inherit_corrected_bounds(source_frame, target_frame,
                                                source, target):
                        target.track_id = source.track_id
                        if reference_index < index:
                            target.prev_box_id = source.id
                proposed = associate_frame_boxes(source_frame, target_frame, sources, [])
                frame_boxes[index] = add_adjacent_tracked_boxes(
                    source_frame, target_frame, sources,
                    frame_boxes.get(index, []), proposed)
                propagate_confirmations(frame_boxes)
    current = frame_boxes.get(index, [])
    if not any(box.source in ('automatic', 'user') and box.review_state != 'rejected'
               and (box.review_state == 'unconfirmed' or not box.identity_confirmed)
               for box in current):
        return True

    for direction in (-1, 1):
        chain = [index]
        neighbour = index + direction
        while 0 <= neighbour < num_frames and frame_boxes.get(neighbour):
            if cancelled():
                return False
            chain.append(neighbour)
            if any(box.review_state in ('confirmed', 'inherited') or
                   has_specific_identity(box)
                   for box in frame_boxes[neighbour]):
                disconnected = False
                for position in range(len(chain) - 1, 0, -1):
                    if cancelled():
                        return False
                    source_index, target_index = chain[position], chain[position - 1]
                    source_frame, target_frame = frame_at(source_index), frame_at(target_index)
                    if source_frame is None or target_frame is None:
                        disconnected = True
                        break
                    connected = False
                    for source, target in match_existing_candidates(
                            source_frame, target_frame,
                            frame_boxes[source_index], frame_boxes[target_index]):
                        old_track = target.track_id
                        if old_track == source.track_id:
                            inherit_corrected_bounds(source_frame, target_frame,
                                                     source, target)
                            if (source.identity_confirmed and
                                    target.review_state == 'confirmed' and
                                    not target.identity_confirmed and
                                    not rejects_identity(target, source.pin_id, source.tags)):
                                target.pin_id = source.pin_id
                                target.tags = list(source.tags or [])
                                target.match_confidence = source.match_confidence
                                target.identity_confirmed = True
                                target.identity_inferred = True
                            connected = True
                            continue
                        members = [box for boxes in frame_boxes.values()
                                   for box in boxes if box.track_id == old_track]
                        if any(not can_link_saved_target(source, box) for box in members):
                            continue
                        if (source.review_state == 'unconfirmed' and
                                any(box.pin_id and box.pin_id != source.pin_id
                                    for box in members)):
                            continue
                        inherit_corrected_bounds(source_frame, target_frame,
                                                 source, target)
                        if (source.identity_confirmed and
                                target.review_state == 'confirmed' and
                                not target.identity_confirmed and
                                not rejects_identity(target, source.pin_id, source.tags)):
                            target.pin_id = source.pin_id
                            target.tags = list(source.tags or [])
                            target.match_confidence = source.match_confidence
                            target.identity_confirmed = True
                            target.identity_inferred = True
                        for member in members:
                            member.track_id = source.track_id
                            if (source.identity_confirmed and member.source == 'automatic'
                                    and member.pin_id and (source.pin_id or source.tags)):
                                if rejects_identity(member, source.pin_id, source.tags):
                                    continue
                                member.pin_id = source.pin_id
                                member.tags = list(source.tags or [])
                                member.match_confidence = source.match_confidence
                                member.identity_confirmed = True
                        if source_index < target_index:
                            target.prev_box_id = source.id
                        connected = True
                    if not connected:
                        disconnected = True
                        break
                    propagate_confirmations(frame_boxes)
                if disconnected:
                    break
                if not any(box.source in ('automatic', 'user')
                           and box.review_state != 'rejected'
                           and (box.review_state == 'unconfirmed' or
                                not box.identity_confirmed) for box in current):
                    break
            neighbour += direction
    return True

def load_boxes_from_file(filename: str) -> dict[int, list[BoxData]]:
    with open(filename, "r", encoding="utf-8") as f:
        return load_boxes_from_stream(f)

def filter_zero_sized_boxes(boxes: dict[int, list[BoxData]]) -> dict[int, list[BoxData]]:
    """Filter out boxes with zero width or height."""
    filtered_boxes: Dict[int, List[BoxData]] = {}
    for frame_index, box_list in boxes.items():
        for box in box_list:
            if not isinstance(box, BoxData):
                getLog().warning(f"Skipping non-BoxData object: {box}")
                continue
            if not isinstance(box.coords, tuple) or len(box.coords) != 4:
                getLog().warning(f"Skipping box with invalid coords: {box.coords}")
                continue
            if not box.is_non_zero_sized():
                getLog().debug(f"Skipping zero-sized box: {box.coords}")
                continue

            if frame_index not in filtered_boxes:
                filtered_boxes[frame_index] = []
            filtered_boxes[frame_index].append(box)

    return filtered_boxes


class ScrubberFrame(wx.Frame):
    __image_panel: ImagePanel
    __tag_panel: TagPanel
    __button_panel: ControlsPanel
    __box_data_filename: str | None = None
    __num_frames: int

    @staticmethod
    def create_box_data_name_from_filename(file_name: str) -> str:
        """Create a box data name from the file name."""
        """Return the filename with its extension replaced by .json."""
        base, _ = os.path.splitext(file_name)
        return base + ".json"

    @property
    def box_data_filename(self) -> str | None:
        """Get the filename for saving/loading box data."""
        return self.__box_data_filename

    @box_data_filename.setter
    def box_data_filename(self, filename: str | None) -> None:
        """Set the filename for saving/loading box data."""
        if filename is not None and not filename.endswith('.json'):
            raise ValueError(f'Box data filename {filename} must end with .json')
        self.__box_data_filename = filename
        self.Bind(wx.EVT_CLOSE, self.on_close)

    def close_video_file(self) -> None:
        # Check if the video file is open and close it
        # if hasattr(self, "video_capture") and self.video_capture is not None:
        #     if self.video_capture.isOpened():
        #         self.video_capture.release()
        #         getLog().info("Video file closed.")
        #     self.video_capture = None

        # Write the JSON metadata file to disk
        if self.box_data_filename and self.__frame_boxes:
            save_boxes_to_file(self.box_data_filename, self.__frame_boxes,
                               self._frame_presence)
            getLog().info(f"Metadata saved to {self.box_data_filename}")

    def load_box_data(self) -> Dict[int, List[BoxData]]:
        """Load box data from the specified file."""
        if self.box_data_filename and os.path.exists(self.box_data_filename):
            try:
                boxes = load_boxes_from_file(self.box_data_filename)
                self._invalidate_reconciliation()
                self._frame_presence = load_presence_from_file(self.box_data_filename)
                self.__frame_boxes = filter_zero_sized_boxes(boxes)
                self.__matched_frames.clear()
                self._cancel_catalog_matches()
                self._undo_history.clear()
                self._redo_history.clear()
                self._history_pending = None
                self._pending_rejection_history = None
                propagate_rejections(self.__frame_boxes)
                propagate_confirmations(self.__frame_boxes)
                self._consolidate_frames()
                self.__detected_frames = {
                    index for index in boxes
                    if self.__frame_boxes.get(index) and
                    marked_pin_count(self.__frame_boxes[index]) >= self._frame_presence.get(
                        index, FramePresence()).minimum_pins}
                self._primary_only_frames.clear()
                self._seed_feedback_from_annotations()
                self._sync_identity_feedback()
                # self.__frame_boxes = merge_duplicate_boxes(boxes)
                count = self.count_boxes()
                getLog().info(f"Loaded {count} boxes in data from {self.box_data_filename}")
                return self.__frame_boxes
            except Exception:
                getLog().exception('Error loading box data from %s', self.box_data_filename)
        else:
            getLog().warning("No box data file specified or file does not exist.")
        return {}

    @property
    def current_index(self) -> int:
        return self._current_index

    def __init__(self, parent: wx.Panel | None, title: str, opencv_exe_path: str,
                 num_frames: int, catalog_path: str | None = None,
                 load_catalog: bool = True, feedback_path: str | None = None,
                 feedback_source: str | None = None):
        super().__init__(parent=parent, title=title, size=wx.Size(1100, 700))

        self.__frame_boxes: Dict[int, List[BoxData]] = {}
        self._frame_presence: dict[int, FramePresence] = {}
        self.__detected_frames: set[int] = set()
        self._primary_only_frames: set[int] = set()
        self.__matched_frames: set[int] = set()
        self._pending_catalog: dict[int, Future[list[BoxData]]] = {}
        self._pending_catalog_targets: dict[int, tuple[BoxData, bool]] = {}
        self._pending_attempt: Future[BoxData | None] | None = None
        self._undo_history: dict[int, list[HistoryEntry]] = {}
        self._redo_history: dict[int, list[HistoryEntry]] = {}
        self._history_pending: tuple[int, dict[int, BoxSnapshot], int,
                                     dict[int, FramePresence]] | None = None
        self._pending_rejection_history: tuple[int, int, dict[int, BoxSnapshot],
                                               HistoryEntry] | None = None
        self._feedback_source = feedback_source or f'session:{uuid4().hex}'
        self._feedback_store = FeedbackStore(feedback_path)
        self._feedback_deferred = (self._feedback_store.path != ':memory:' and
                                   self._feedback_store.example_count() >= 1000)
        self._feedback_model = FeedbackModel(self._feedback_store,
                                             load=not self._feedback_deferred)
        self._feedback_load_executor = (ThreadPoolExecutor(
            max_workers=1, thread_name_prefix='pin-feedback')
            if self._feedback_deferred else None)
        self._feedback_load_future: Future[tuple[FeedbackModel, str | None]] | None = None
        self._feedback_load_cancel = Event()
        self._pin_catalog = PinCatalog(catalog_path) if catalog_path else PinCatalog()
        self._training_future: Future | None = None
        self._model_training_future: Future | None = None
        self._trained_model_path = model_path_for_feedback(self._feedback_store.path)
        self._cloud_config = CloudConfig.from_environment()
        self._cloud_model_version = 0
        self._cloud_status = 'Not checked'
        self._cloud_sync_future: Future | None = None
        self._cloud_publish_future: Future | None = None
        self._cloud_train_after_sync = False
        self._cloud_ready_for_publish = False
        self._training_base_cloud_version = 0
        self._sync_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='pin-cloud')
        self._review_save_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix='pin-review-save')
        self._review_save_future: Future[None] | None = None
        self._review_save_call: wx.CallLater | None = None
        self._review_save_pending = False
        self._review_autosave_active = False
        self._review_action_active = False
        self._review_feedback_changes: dict[str, FeedbackExample | None] = {}
        self._review_feedback_future: Future[None] | None = None
        self._review_feedback_inflight: dict[str, FeedbackExample | None] = {}
        self._review_feedback_call: wx.CallLater | None = None
        self._review_feedback_failed = False
        if (self._trained_model_path is not None and
                self._trained_model_path.is_file()):
            try:
                saved_model = TrainedDetectionModel.load(self._trained_model_path)
                self._feedback_model.trained_model = saved_model
                if self._feedback_deferred:
                    getLog().info('Saved classifier freshness check queued with feedback load')
                elif not (saved_model.catalog_state ==
                        self._pin_catalog.stock_training_state() and
                        saved_model.review_digest ==
                        examples_digest(self._feedback_store.examples())):
                    getLog().warning(
                        'Using saved classifier trained on older reviews or Pinpanion '
                        'catalog; run training again to include the latest')
            except (OSError, ValueError, KeyError, cv2.error):
                getLog().exception('Could not load trained classifier from %s',
                                   self._trained_model_path)
        if (self._trained_model_path is not None and
                not isinstance(self._feedback_model.trained_model,
                               TrainedDetectionModel)):
            cache_path = self._trained_model_path.with_suffix('.cloud.json')
            if (cache_path.is_file() and
                    (not self._trained_model_path.is_file() or
                     cache_path.stat().st_mtime >= self._trained_model_path.stat().st_mtime)):
                try:
                    cached = json.loads(cache_path.read_text(encoding='utf-8'))
                    self._cloud_model_version = cached['cloud_version']
                    self._feedback_model.trained_model = PortableDetectionModel.from_payload(
                        cached['model'])
                except (OSError, KeyError, ValueError, TypeError):
                    getLog().exception('Could not load cached Cloudflare classifier')
        self._playback_step = False
        self._playing = False
        self._play_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='pin-detection')
        self._detection_resolution = load_detection_resolution()
        self._pending_frame_job: Future[FrameResult] | None = None
        self._pending_reconcile: Future[bool] | None = None
        self._pending_reconcile_index: int | None = None
        self._reconcile_cancel = Event()
        self._reconcile_done: set[int] = set()
        self._reconciled_pairs: set[tuple[int, int]] = set()
        self._annotation_revision = 0
        self._play_generation = 0
        self._closing = False
        self._destroy_requested = False
        self._shutdown_dialog: ShutdownProgressDialog | None = None
        self._shutdown_stage = 0
        self._shutdown_prepared = False
        self._shutdown_play_barrier: Future | None = None
        self._shutdown_cloud_barrier: Future | None = None
        self._shutdown_save_future: Future | None = None
        self._shutdown_reader_future: Future | None = None
        self._shutdown_save_executor: ThreadPoolExecutor | None = None
        self._shutdown_timer: wx.Timer | None = None
        self._display_source_frame: np.ndarray | None = None
        self._tags_dirty = False
        self._play_timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self._on_play_timer, self._play_timer)
        self._pin_matcher = PinMatcher(self._pin_catalog.references() if load_catalog else [])

        self._current_index = 0
        self._rotation_angle = 0
        self.__num_frames = num_frames
        self.Bind(wx.EVT_SIZE, self.on_resize)

        main_panel = wx.Panel(self)
        vbox = wx.BoxSizer(wx.VERTICAL)

        image_and_tag_sizer = wx.BoxSizer(wx.HORIZONTAL)

        self.__image_panel = ImagePanel(main_panel)
        self.__image_panel.on_confirm_box = self.confirm_box
        self.__image_panel.on_confirm_boxes = self.confirm_boxes
        self.__image_panel.on_confirm_identities = self._confirm_identities
        self.__image_panel.on_confirm_identity = self.confirm_pin_identity
        self.__image_panel.on_wrong_identity = self.mark_wrong_identity
        self.__image_panel.on_attempt_detection = self.attempt_detection
        self.__image_panel.on_rank_replacements = self.rank_pin_replacements
        self.__image_panel.on_replace_pin = self.replace_with_pin
        self.__image_panel.on_reject_replacements = self.reject_pin_replacements
        self.__image_panel.on_box_added = self._on_image_box_added
        self.__image_panel.on_remove_box = self.remove_box
        self.__image_panel.on_remove_boxes = self.remove_boxes
        self.__image_panel.on_not_pin = self.mark_not_pin
        self.__image_panel.on_not_pin_boxes = self.mark_not_pin_boxes
        self.__image_panel.on_interact = self.pause_playback
        self.__image_panel.on_change_begin = self._begin_history
        self.__image_panel.on_change_end = self._finish_history
        self.__image_panel.on_undo = self.undo_action
        self.__image_panel.on_redo = self.redo_action
        # self.image_panel.Bind(EVT_BOX_ADDED, self.image_box_added)

        image_and_tag_sizer.Add(self.__image_panel, 1, wx.EXPAND | wx.ALL, 10)

        self.__tag_panel = TagPanel(
            main_panel,
            on_confirm=self.confirm_box,
            on_confirm_identity=self.confirm_pin_identity,
            on_remove=self.remove_box,
            on_interact=self.pause_playback,
            on_change_begin=self._begin_history,
            on_change_end=self._finish_history,
            on_presence_change=self._on_presence_change,
            catalog_options=self._pin_catalog.list_pins() if load_catalog else [],
        )
        self.__image_panel.on_before_context_menu = self.__tag_panel.commit_pending_labels
        self.__image_panel.Bind(EVT_BOX_SELECTED, self.__tag_panel.on_box_selected)
        frame_boxes = self.__get_frame_boxes(self._current_index)
        self.__tag_panel.boxes = frame_boxes
        # print(f'ScrubberFrame.__init__: TagPanel referencing {hex(id(self.__boxes))}=>{self.__boxes}')
        self.__tag_panel.bind_box_events(self.__image_panel)
        # self.image_panel.Bind(EVT_BOX_ADDED, self.tag_panel.Refresh)

        image_and_tag_sizer.Add(self.__tag_panel, 0, wx.EXPAND | wx.ALL, 10)

        vbox.Add(image_and_tag_sizer, 1, wx.EXPAND | wx.ALL, 5)

        # Add ControlsPanel below image_panel
        self.__button_panel = ControlsPanel(main_panel)
        vbox.Add(self.__button_panel, 0, wx.CENTER, 0)
        self.__button_panel.bind_buttons(self.on_prev, self.on_next_async, self.on_next_empty, self.on_rotate_ccw, self.on_rotate_cw, self.__on_remove_selected, self.on_detect_async, self.on_play, self.confirm_all_suggestions)
        self._create_menu_bar()

        # Add MarkerPanel below image_panel
        self.marker_panel = MarkerPanel(main_panel, num_frames)
        vbox.Add(self.marker_panel, 0, wx.EXPAND | wx.ALL, 5)

        self.slider = wx.Slider(main_panel, value=0, minValue=0, maxValue=max(1, num_frames-1),
                                style=wx.SL_HORIZONTAL | wx.SL_LABELS)
        self.slider.Enable(num_frames > 1)
        self.slider.Bind(wx.EVT_SLIDER, self.on_slider)
        vbox.Add(self.slider, 0, wx.EXPAND | wx.ALL, 10)

        main_panel.SetSizer(vbox)

        status = self.CreateStatusBar(2)
        status.SetStatusWidths([-3, -2])
        self._update_feedback_status()
        if self._feedback_deferred:
            status.SetStatusText('Loading reviewed-crop descriptors after window opens...', 0)

        self.Bind(wx.EVT_CHAR_HOOK, self.on_key_down)
        self.Bind(wx.EVT_SHOW, self.on_show)
        self._cloud_timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self._on_cloud_timer, self._cloud_timer)
        self._cloud_timer.Start(5 * 60 * 1000)
        self._start_cloud_sync()

    def _create_menu_bar(self) -> None:
        file_menu = wx.Menu()
        save_item = file_menu.Append(wx.ID_ANY, 'Save')
        settings_item = file_menu.Append(wx.ID_ANY, 'Settings')
        load_item = file_menu.Append(wx.ID_ANY, 'Load Annotations...')

        training_menu = wx.Menu()
        review_areas_item = training_menu.Append(wx.ID_ANY, 'Review saved areas...')
        training_menu.AppendSeparator()
        feedback_item = training_menu.Append(wx.ID_ANY, 'Feedback model...')
        self._training_add_item = training_menu.Append(wx.ID_ANY, 'Add to training data')
        self._training_run_item = training_menu.Append(wx.ID_ANY, 'Run training now')
        self._training_add_item.SetHelp('Save reviewed crops and refresh the example matcher.')
        self._training_run_item.SetHelp('Index reviews, then train and save the pin classifier.')

        menu_bar = wx.MenuBar()
        menu_bar.Append(file_menu, 'File')
        menu_bar.Append(training_menu, 'Training')
        self.SetMenuBar(menu_bar)

        self.Bind(wx.EVT_MENU, self.file_select, load_item)
        self.Bind(wx.EVT_MENU, self.save_annotations, save_item)
        self.Bind(wx.EVT_MENU, self.show_settings, settings_item)
        self.Bind(wx.EVT_MENU, self.open_classifier_review, review_areas_item)
        self.Bind(wx.EVT_MENU, lambda _event: self.show_feedback_info(), feedback_item)
        self.Bind(wx.EVT_MENU, lambda _event: self.add_to_training_data(), self._training_add_item)
        self.Bind(wx.EVT_MENU, lambda _event: self.run_training_now(), self._training_run_item)

    def _set_training_progress(self, percent: int, status: str, running: bool) -> None:
        self.__tag_panel.set_training_progress(percent, status, running)
        self._training_add_item.Enable(not running)
        self._training_run_item.Enable(not running)

    def get_frame(self, index: int, rotation_angle: int = 0):
        raise NotImplementedError

    def read_frame_for_processing(self, index: int):
        raise NotImplementedError

    def close_processing_reader(self) -> None:
        pass

    def __get_frame_boxes(self, index: int) -> List[BoxData]:
        """Get the boxes for the current frame index."""
        fb = self.__frame_boxes
        if not index in fb:
            fb[index] = []

        return fb[index]

    def _on_image_box_added(self, box: BoxData) -> None:
        boxes = self.__current_boxes
        if all(existing is not box for existing in boxes):
            boxes.append(box)
        if box.source == 'user' and box.review_state == 'confirmed':
            box.feedback_id = box.feedback_id or uuid4().hex
            box.feedback_label = 'positive'
        self.__image_panel.boxes = boxes
        self.__tag_panel.boxes = boxes
        self._sync_presence_ui()
        wx.CallAfter(self._match_newly_drawn_box, self._current_index, box)

    def _match_newly_drawn_box(self, index: int, box: BoxData) -> None:
        if (self._closing or box.source != 'user' or
                all(existing is not box for existing in self.__frame_boxes.get(index, []))):
            return
        self.__matched_frames.discard(index)
        pending = self._pending_catalog.pop(index, None)
        if pending is not None:
            pending.cancel()
        frame = (self._display_source_frame if index == self._current_index and
                 self._rotation_angle == 0 else None)
        self._schedule_catalog_match(index, frame, None, target_box=box)

    def _reconcile_frame(self, index: int) -> None:
        """Carry trusted identities through saved adjacent candidate chains."""
        reconcile_saved_frames(index, self.__num_frames, self.__frame_boxes,
                               lambda frame: self.get_frame(frame, 0))

    def _cancel_reconciliation(self) -> None:
        self._reconcile_cancel.set()
        if self._pending_reconcile is not None:
            self._pending_reconcile.cancel()
            self._pending_reconcile = None
        self._pending_reconcile_index = None

    def _cancel_catalog_matches(self) -> None:
        for future in self._pending_catalog.values():
            future.cancel()
        self._pending_catalog.clear()
        self._pending_catalog_targets.clear()

    def _cancel_attempt_detection(self) -> None:
        if self._pending_attempt is not None:
            self._pending_attempt.cancel()
            self._pending_attempt = None

    def _invalidate_reconciliation(self) -> None:
        self._cancel_reconciliation()
        self._reconcile_done.clear()
        self._reconciled_pairs.clear()
        self._annotation_revision += 1
        self._pending_rejection_history = None

    @staticmethod
    def _copy_box_for_worker(box: BoxData) -> BoxData:
        """Give a worker an independent observation without recursive copying."""
        result = copy(box)
        result.tags = list(box.tags) if box.tags is not None else None
        return result

    def _schedule_reconciliation(self, index: int,
                                 reference_index: int | None = None) -> None:
        if self._closing:
            return
        has_reference = (reference_index is not None and abs(reference_index - index) == 1
                         and any(box.review_state != 'rejected' and
                                 (box.review_state in ('confirmed', 'inherited') or
                                  has_specific_identity(box) or
                                  box.bounds_adjustment is not None or
                                  any(target.track_id == box.track_id and
                                      target.bounds_adjustment is not None and
                                      not target.bounds_direct
                                      for target in self.__frame_boxes.get(index, [])))
                                 for box in self.__frame_boxes.get(reference_index, [])))
        pair = (reference_index, index) if has_reference else None
        if index in self._reconcile_done and (pair is None or pair in self._reconciled_pairs):
            return
        if self._pending_reconcile is not None and self._pending_reconcile_index == index:
            return
        current = self.__frame_boxes.get(index, [])
        direct_tracks = {box.track_id for box in current
                         if box.review_state == 'confirmed'}
        adjacent_direct = bool(direct_tracks) and any(
            box.review_state == 'confirmed' and box.track_id not in direct_tracks
            for neighbour in (index - 1, index + 1)
            for box in self.__frame_boxes.get(neighbour, []))
        if not any((box.source in ('automatic', 'user') and
                    box.review_state != 'rejected' and
                    (box.review_state == 'unconfirmed' or
                     not box.identity_confirmed)) or
                   box.review_state == 'rejected' for box in current) and pair is None \
                and not adjacent_direct:
            self._reconcile_done.add(index)
            return
        if self._pending_reconcile is not None:
            self._cancel_reconciliation()
        # Reconciliation only mutates scalar box fields and tags. Copy each
        # observation and its tag list rather than recursively copying the
        # entire annotation graph on the wx thread for every review.
        snapshot = {
            frame_index: [self._copy_box_for_worker(box) for box in boxes]
            for frame_index, boxes in self.__frame_boxes.items()
        }
        snapshot.setdefault(index, [])
        initial_counts = {frame_index: len(boxes) for frame_index, boxes in snapshot.items()}
        cancelled = Event()
        self._reconcile_cancel = cancelled
        revision = self._annotation_revision
        def run_reconciliation() -> bool:
            started = perf_counter()
            result = reconcile_saved_frames(
                index, self.__num_frames, snapshot,
                self.read_frame_for_processing, cancelled.is_set,
                reference_index if has_reference else None)
            getLog().info('frame=%d reconcile=%.1fms cancelled=%s thread=worker',
                          index, (perf_counter() - started) * 1000, not result)
            return result
        future = self._play_executor.submit(
            run_reconciliation)
        self._pending_reconcile = future
        self._pending_reconcile_index = index
        future.add_done_callback(lambda finished: wx.CallAfter(
            self._complete_reconciliation, finished, index, revision, snapshot,
            initial_counts, pair))

    def _complete_reconciliation(self, future: Future[bool], index: int,
                                  revision: int,
                                  analyzed_boxes: dict[int, list[BoxData]],
                                  initial_counts: dict[int, int],
                                  pair: tuple[int, int] | None) -> None:
        if (self._closing or future is not self._pending_reconcile or
                revision != self._annotation_revision or self._playing):
            return
        self._pending_reconcile = None
        self._pending_reconcile_index = None
        try:
            if not future.result():
                return
        except Exception:
            getLog().exception('Background saved-frame reconciliation failed')
            return
        for frame_index, updated in analyzed_boxes.items():
            original = self.__frame_boxes.get(frame_index)
            count = initial_counts[frame_index]
            if original is None or len(original) != count or len(updated) < count:
                return
        for frame_index, updated in analyzed_boxes.items():
            count = initial_counts[frame_index]
            for original, analyzed in zip(self.__frame_boxes[frame_index], updated[:count]):
                if (original.source == 'automatic' and
                        original.review_state in ('unconfirmed', 'inherited') and
                        not original.bounds_direct and original.feedback_id is None and
                        (original.coords != analyzed.coords or
                         original.bounds_adjustment != analyzed.bounds_adjustment)):
                    original.coords = analyzed.coords
                    original.bounds_adjustment = analyzed.bounds_adjustment
                original.track_id = analyzed.track_id
                original.pin_id = analyzed.pin_id
                original.tags = list(analyzed.tags or [])
                original.match_confidence = analyzed.match_confidence
                original.identity_confirmed = analyzed.identity_confirmed
                original.identity_inferred = analyzed.identity_inferred
                original.review_state = analyzed.review_state
                original.identity_rejections = analyzed.identity_rejections
            self.__frame_boxes[frame_index].extend(updated[count:])
        self._sync_automatic_consolidation(invalidate=False)
        pending_history = self._pending_rejection_history
        if pending_history is not None and pending_history[1] == revision:
            origin_frame, _, before, entry = pending_history
            history = self._undo_history.get(origin_frame, [])
            if history and history[-1] is entry:
                history[-1] = make_entry(before, snapshot(self.__frame_boxes),
                                         entry.before_rotation, entry.after_rotation,
                                         entry.before_presence, entry.after_presence)
            self._pending_rejection_history = None
        self._reconcile_done.add(index)
        if pair is not None:
            self._reconciled_pairs.add(pair)
        if self._current_index == index:
            self.__image_panel.boxes = self.__current_boxes
            self.__tag_panel.boxes = self.__current_boxes
            self._sync_presence_ui()
            frame = (self._display_source_frame if self._display_source_frame is not None
                     else self.get_frame(index, 0))
            self._schedule_catalog_match(index, frame, None)
        if self._review_autosave_active:
            self.queue_review_save()

    def display_image(self, source_frame: np.ndarray | None = None,
                      reconcile: bool = True,
                      reference_index: int | None = None,
                      catalog_matching: bool = True) -> None:
        started = perf_counter()
        if source_frame is None:
            img = self.get_frame(self._current_index, self._rotation_angle)
            self._display_source_frame = img if self._rotation_angle == 0 else None
        else:
            self._display_source_frame = source_frame
            img = source_frame
            if self._rotation_angle == 90:
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            elif self._rotation_angle == 180:
                img = cv2.rotate(img, cv2.ROTATE_180)
            elif self._rotation_angle == 270:
                img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        if img is None:
            self.slider.SetValue(self._current_index)
            self.__button_panel.set_prev_enabled(self._current_index > 0)
            self.__button_panel.set_next_enabled(self._current_index < self.__num_frames - 1)
            self.SetStatusText(f'Frame {self._current_index}: could not decode image', 1)
            getLog().warning('frame=%d ui display failed: no decoded image', self._current_index)
            return
        self._sync_presence_ui()
        panel_size = self.__image_panel.GetSize()
        if panel_size.GetWidth() < 10 or panel_size.GetHeight() < 10:
            return  # Panel not yet sized, skip

        decoded = perf_counter()
        reconciled = perf_counter()
        self.__image_panel.set_image(img, self._rotation_angle)
        self.__image_panel.boxes = self.__current_boxes
        bitmap_ready = perf_counter()

        if reconcile and not self._playing:
            self._schedule_reconciliation(self._current_index, reference_index)
        if catalog_matching and self._pending_reconcile_index != self._current_index:
            self._schedule_catalog_match(
                self._current_index,
                img if self._rotation_angle == 0 else self.get_frame(self._current_index, 0),
                reference_index if reconcile else None)

        frame_boxes = self.__get_frame_boxes(self._current_index)
        self._sync_presence_ui()
        if self._playing:
            self._tags_dirty = True
        else:
            self.__tag_panel.boxes = frame_boxes
            self._tags_dirty = False
        tags_ready = perf_counter()

        self.__button_panel.set_prev_enabled(self._current_index > 0)
        self.__button_panel.set_next_enabled(self._current_index < self.__num_frames - 1)

        self.slider.SetValue(self._current_index)
        getLog().info('frame=%d ui decode=%.1fms reconcile=%.1fms bitmap=%.1fms tags=%.1fms total=%.1fms',
                      self._current_index, (decoded-started)*1000,
                      (reconciled-decoded)*1000, (bitmap_ready-reconciled)*1000,
                      (tags_ready-bitmap_ready)*1000, (perf_counter()-started)*1000)

    def _schedule_catalog_match(self, index: int, frame: np.ndarray | None,
                                reference_index: int | None,
                                target_box: BoxData | None = None,
                                skip_box: BoxData | None = None,
                                best_effort: bool = False) -> bool:
        if index in self._pending_catalog:
            if target_box is None:
                return True
            self._pending_catalog.pop(index).cancel()
            self._pending_catalog_targets.pop(index, None)
        if target_box is not None:
            self.__matched_frames.discard(index)
        if self._closing or index in self.__matched_frames:
            return False
        if (not getattr(self._pin_matcher, 'has_references', True) and
                not self._feedback_model.identity_sample_count):
            self.__matched_frames.add(index)
            return False
        candidates = [(box, deepcopy(box), deepcopy(box))
                      for box in sorted(
                          ([target_box] if target_box is not None else
                           self.__frame_boxes.get(index, [])),
                          key=lambda item: ({'confirmed': 0, 'inherited': 1}.get(
                              item.review_state, 2), item.source != 'user'))
                      if box is not skip_box and
                      needs_catalog_match(box, best_effort=best_effort)]
        if not candidates:
            self.__matched_frames.add(index)
            return False
        matcher = self._pin_matcher
        feedback = self._feedback_model.snapshot_for_processing()
        size_context = ([deepcopy(box) for box in self.__frame_boxes.get(index, [])]
                        if target_box is None else None)
        revision = self._annotation_revision
        started = perf_counter()
        def scan() -> list[BoxData] | None:
            image = frame.copy() if frame is not None else self.read_frame_for_processing(index)
            if image is None:
                return None
            result = identify_candidates(
                image, [copy for _, _, copy in candidates], matcher,
                best_effort=best_effort, feedback_model=feedback,
                prefer_reviewed_identity=best_effort,
                size_context=size_context)
            getLog().info('frame=%d catalog=%.1fms candidates=%d thread=worker',
                          index, (perf_counter() - started) * 1000, len(result))
            return result
        future = self._play_executor.submit(scan)
        self._pending_catalog[index] = future
        if target_box is not None:
            self._pending_catalog_targets[index] = (target_box, best_effort)
        future.add_done_callback(lambda completed: wx.CallAfter(
            self._complete_catalog_match, completed, index, revision, matcher,
            candidates, reference_index, target_box is not None, best_effort))
        return True

    def _complete_catalog_match(self, future: Future[list[BoxData] | None], index: int,
                                revision: int, matcher: PinMatcher,
                                candidates: list[tuple[BoxData, BoxData, BoxData]],
                                reference_index: int | None,
                                target_only: bool = False,
                                best_effort: bool = False) -> None:
        if future is not self._pending_catalog.get(index):
            return
        del self._pending_catalog[index]
        self._pending_catalog_targets.pop(index, None)
        if (self._closing or revision != self._annotation_revision or
                matcher is not self._pin_matcher or index in self.__matched_frames):
            getLog().debug('frame=%d catalog result stale closing=%s revision=%d/%d matcher=%s matched=%s',
                           index, self._closing, revision, self._annotation_revision,
                           matcher is self._pin_matcher, index in self.__matched_frames)
            if target_only and not self._closing and index == self._current_index:
                original, before, _ = candidates[0]
                if (any(existing is original for existing in self.__current_boxes) and
                        original.coords == before.coords and
                        original.track_id == before.track_id and
                        original.review_state == before.review_state and
                        original.identity_confirmed == before.identity_confirmed and
                        original.pin_id == before.pin_id and
                        original.tags == before.tags):
                    self._schedule_catalog_match(
                        index, None, reference_index, target_box=original,
                        best_effort=best_effort)
            return
        try:
            analyzed = future.result()
        except Exception:
            getLog().exception('Background catalog matching failed for frame %d', index)
            if target_only and best_effort and self._current_index == index:
                self.SetStatusText(f'Frame {index}: identity retry failed; see console log', 1)
            return
        if analyzed is None:
            getLog().warning('Background catalog matching could not decode frame %d', index)
            if target_only and best_effort and self._current_index == index:
                self.SetStatusText(f'Frame {index}: could not decode selected area', 1)
            return
        current = self.__frame_boxes.get(index, [])
        if any((all(box is not original for box in current) or
                original.coords != before.coords or
                original.track_id != before.track_id or
                original.review_state != before.review_state or
                original.identity_confirmed != before.identity_confirmed or
                original.pin_id != before.pin_id or original.tags != before.tags)
               for original, before, _worker_copy in candidates):
            getLog().debug('frame=%d catalog result skipped after box changed', index)
            return
        changed = False
        for (original, before, _worker_copy), result in zip(candidates, analyzed):
            if result.pin_id is None:
                if (before.pin_id is not None and not result.identity_confirmed and
                        not result.tags and original.pin_id is not None):
                    original.pin_id = None
                    original.tags = []
                    original.match_confidence = None
                    changed = True
                continue
            if (original.pin_id, original.tags, original.match_confidence) != (
                    result.pin_id, result.tags, result.match_confidence):
                original.pin_id = result.pin_id
                original.tags = list(result.tags or [])
                original.match_confidence = result.match_confidence
                changed = True
        if not target_only:
            self.__matched_frames.add(index)
        if changed:
            self._invalidate_reconciliation()
            self._sync_automatic_consolidation(invalidate=False)
            if self._review_autosave_active:
                self.queue_review_save()
        if self._current_index == index:
            if changed:
                self.__image_panel.Refresh()
                self.__tag_panel.boxes = self.__current_boxes
                self._sync_presence_ui()
            if target_only and best_effort:
                self.SetStatusText(
                    (f'Frame {index}: suggested {(analyzed[0].tags or [analyzed[0].pin_id])[0]}'
                     if analyzed[0].pin_id else
                     f'Frame {index}: cleared unverified identity suggestion')
                    if changed else
                    f'Frame {index}: no other pin identity match for selected area', 1)
            if not self._playing:
                self._schedule_reconciliation(index, reference_index)
        if target_only:
            self._schedule_catalog_match(index, None, reference_index,
                                         skip_box=candidates[0][0])

    def attempt_detection(self, box: BoxData) -> None:
        """Rescan one saved area for a provisional pin identity."""
        self.pause_playback()
        if (self._closing or box.review_state == 'rejected' or
                all(existing is not box for existing in self.__current_boxes)):
            return
        index = self._current_index
        before = deepcopy(box)
        probe = deepcopy(box)
        probe.pin_id = None
        probe.tags = []
        probe.match_confidence = None
        probe.identity_confirmed = False
        probe.identity_inferred = False
        if probe.source == 'processed':
            probe.source = 'automatic'
        # The probe is only a matching input. Best-effort matching already
        # checks reviewed crops for an unconfirmed area; keep its review state
        # so the confirmed-presence ranking rule applies only after a review.
        matcher = self._pin_matcher
        feedback = self._feedback_model.snapshot_for_processing()
        if (not getattr(matcher, 'has_references', True) and
                not feedback.identity_sample_count):
            self.SetStatusText(
                f'Frame {index}: no catalog images or reviewed pin identities are loaded', 1)
            return
        revision = self._annotation_revision
        generation = self._play_generation
        self.SetStatusText(f'Frame {index}: attempting identity detection for selected area', 1)

        def scan() -> BoxData | None:
            image = self.read_frame_for_processing(index)
            if image is None:
                return None
            return identify_candidates(image, [probe], matcher, best_effort=True,
                                       feedback_model=feedback,
                                       prefer_reviewed_identity=True)[0]

        future = self._play_executor.submit(scan)
        self._pending_attempt = future
        future.add_done_callback(lambda completed: wx.CallAfter(
            self._complete_attempt_detection, completed, index, revision,
            generation, matcher, box, before))

    def _complete_attempt_detection(self, future: Future[BoxData | None], index: int,
                                    revision: int, generation: int,
                                    matcher: PinMatcher, box: BoxData,
                                    before: BoxData) -> None:
        if future is not self._pending_attempt:
            return
        self._pending_attempt = None
        if (self._closing or index != self._current_index or
                revision != self._annotation_revision or
                generation != self._play_generation or
                matcher is not self._pin_matcher or
                all(existing is not box for existing in self.__current_boxes) or
                any(getattr(box, name) != getattr(before, name) for name in (
                    'coords', 'tags', 'pin_id', 'match_confidence', 'review_state',
                    'identity_confirmed', 'identity_inferred', 'identity_rejections',
                    'track_id', 'source'))):
            return
        try:
            result = future.result()
        except Exception:
            getLog().exception('Frame %d: selected-area identity detection failed', index)
            self.SetStatusText(f'Frame {index}: identity detection failed; see console log', 1)
            return
        if result is None:
            self.SetStatusText(f'Frame {index}: could not decode selected area', 1)
            return
        if result.pin_id is None:
            if box.pin_id is not None and not box.identity_confirmed and not result.tags:
                self._begin_history()
                box.pin_id = None
                box.tags = []
                box.match_confidence = None
                self._finish_history()
                self.__image_panel.Refresh()
                self.__tag_panel.boxes = self.__current_boxes
                self._sync_presence_ui()
                self._schedule_reconciliation(index)
                self.SetStatusText(f'Frame {index}: cleared unverified identity suggestion', 1)
            else:
                self.SetStatusText(f'Frame {index}: no pin identity match for selected area', 1)
            return
        label = (result.tags or [result.pin_id])[0]
        if (box.identity_confirmed or
                (box.pin_id is None and has_specific_identity(box))):
            self.SetStatusText(
                f'Frame {index}: model suggests {label}; kept the reviewed or manually entered identity', 1)
            return
        if (box.pin_id, box.tags, box.match_confidence) == (
                result.pin_id, result.tags, result.match_confidence):
            self.SetStatusText(
                f'Frame {index}: {label} is already the best match; '
                'use Not this pin to exclude it', 1)
            return
        if (box.pin_id, box.tags, box.match_confidence) != (
                result.pin_id, result.tags, result.match_confidence):
            self._begin_history()
            box.pin_id = result.pin_id
            box.tags = list(result.tags or [])
            box.match_confidence = result.match_confidence
            self._finish_history()
            self.__image_panel.Refresh()
            self.__tag_panel.boxes = self.__current_boxes
            self._sync_presence_ui()
            self._schedule_reconciliation(index)
        self.SetStatusText(f'Frame {index}: updated suggested pin to {label}', 1)

    def on_resize(self, event: wx.CommandEvent):
        if self._display_source_frame is not None and hasattr(self, 'slider'):
            self.display_image(self._display_source_frame, reconcile=False)
        event.Skip()

    def on_prev(self, _event: wx.CommandEvent):
        if self._current_index > 0:
            self.goto_frame(self._current_index - 1)

    def frame_has_boxes(self, index: int) -> bool:
        """Check if the current frame has boxes."""
        return any(box.review_state != 'rejected'
                   for box in self.__frame_boxes.get(index, []))

    def on_next(self, _event: wx.CommandEvent):
        if not self._playback_step:
            self.pause_playback()
        if self._current_index < self.__num_frames - 1:
            reference_index = self._current_index
            next_index = self._current_index + 1
            if next_index not in self.__detected_frames:
                current_frame = self.get_frame(self._current_index, 0)
                next_frame = self.get_frame(next_index, 0)
                if next_frame is not None:
                    detections = detect_pin_boxes(next_frame, feedback_model=self._feedback_model)
                    previous_boxes = self.__current_boxes
                    if current_frame is not None and previous_boxes:
                        proposed = associate_frame_boxes(current_frame, next_frame, previous_boxes, detections)
                    else:
                        proposed = add_new_detections([], detections)
                    existing = self.__get_frame_boxes(next_index)
                    proposed = filter_new_candidates_by_pin_size(next_frame, existing,
                                                                 proposed)
                    self.__frame_boxes[next_index] = add_unique_boxes(existing, proposed)
                    self._invalidate_reconciliation()
                    self._reconcile_frame(next_index)
                    self._reconcile_done.add(next_index)
                    self.__frame_boxes[next_index] = identify_candidates(
                        next_frame, self.__frame_boxes[next_index], self._pin_matcher,
                        feedback_model=self._feedback_model)
                    inherit_confirmed_catalog_matches(previous_boxes,
                                                      self.__frame_boxes[next_index])
                    if (marked_pin_count(self.__frame_boxes[next_index]) <
                            self._frame_presence.get(next_index, FramePresence()).minimum_pins
                            or not any(box.review_state != 'rejected'
                                       for box in self.__frame_boxes[next_index])):
                        extra = detect_pin_boxes_relaxed(next_frame, self._feedback_model)
                        proposed_extra = filter_new_candidates_by_pin_size(
                            next_frame, self.__frame_boxes[next_index],
                            add_new_detections([], extra))
                        self.__frame_boxes[next_index] = add_unique_boxes(
                            self.__frame_boxes[next_index], proposed_extra)
                        identify_candidates(next_frame, self.__frame_boxes[next_index],
                                            self._pin_matcher,
                                            feedback_model=self._feedback_model)
                        inherit_confirmed_catalog_matches(previous_boxes,
                                                          self.__frame_boxes[next_index])
                    propagate_confirmations(self.__frame_boxes)
                    self._sync_automatic_consolidation()
                    self.__matched_frames.add(next_index)
                    self.__detected_frames.add(next_index)
            elif any(box.review_state in ('confirmed', 'inherited') and
                     needs_catalog_match(box)
                     for box in self.__frame_boxes.get(next_index, [])):
                self.__matched_frames.discard(next_index)
            self._current_index = next_index
            self.display_image(reference_index=reference_index)

    def goto_frame(self, index: int, set_slider: bool = True) -> bool:
        """Go to a specific frame index."""
        self.pause_playback()
        if 0 <= index < self.__num_frames:
            if (index != self._current_index and any(
                    box.review_state in ('confirmed', 'inherited') and
                    needs_catalog_match(box)
                    for box in self.__frame_boxes.get(index, []))):
                self.__matched_frames.discard(index)
            reference_index = self._current_index if abs(index - self._current_index) == 1 else None
            reference_frame = self._display_source_frame if reference_index is not None else None
            self._current_index = index
            if set_slider and self.slider is not None:
                self.slider.SetValue(index)
            if self._needs_manual_detection(index) and (
                    reference_index is not None or not self.frame_has_boxes(index)):
                self.display_image(reconcile=False)
                if self._display_source_frame is not None:
                    self._submit_frame_job(index, requires_play=False,
                                           reference_index=reference_index,
                                           reference_frame=reference_frame)
            else:
                self.display_image(reference_index=reference_index)
            return True
        else:
            getLog().warning(f"Index {index} out of bounds for {self.__num_frames} frames.")

        return False

    def on_slider(self, _event: wx.CommandEvent) -> bool:
        return self.goto_frame(self.slider.GetValue(), set_slider=False)

    def on_next_empty(self, _event: wx.CommandEvent) -> bool:
        current_index = self._current_index
        total_frames = self.__num_frames

        for idx in range(current_index + 1, total_frames):
            if not self.frame_has_boxes(idx):  # implement has_boxes(idx) to check for boxes
                self.goto_frame(idx)  # implement goto_frame(idx) to show the frame
                return True

        wx.MessageBox("No more empty frames found.", "Info", wx.OK | wx.ICON_INFORMATION)
        return False

    def on_rotate_cw(self, _event: wx.CommandEvent) -> None:
        self.pause_playback()
        self._begin_history()
        self._rotation_angle = (self._rotation_angle + 90) % 360
        self.__image_panel.rotate_boxes(self._rotation_angle)
        self._finish_history()
        self.display_image()

    def on_rotate_ccw(self, _event: wx.CommandEvent) -> None:
        self.pause_playback()
        self._begin_history()
        self._rotation_angle = (self._rotation_angle - 90) % 360
        self.__image_panel.rotate_boxes(self._rotation_angle)
        self._finish_history()
        self.display_image()

    def on_show(self, event: wx.ShowEvent):
        if event.IsShown():
            self.GetChildren()[0].Layout()  # main_panel.Layout()
            self.display_image()
            if self._feedback_deferred:
                wx.CallAfter(self._start_feedback_load)
        event.Skip()

    def _start_feedback_load(self) -> None:
        if (self._closing or not self._feedback_deferred or
                self._feedback_load_future is not None or
                self._feedback_load_executor is None):
            return
        path = self._feedback_store.path
        ui_changes = self._feedback_store.db.total_changes
        trained = self._feedback_model.trained_model
        self.SetStatusText('Loading reviewed-crop descriptors...', 0)

        def progress(done: int, total: int) -> None:
            if self._feedback_load_cancel.is_set():
                raise CancelledError()
            wx.CallAfter(self._feedback_load_progress, done, total)

        def load() -> tuple[FeedbackModel, str | None]:
            worker_store = FeedbackStore(path)
            try:
                model = FeedbackModel(worker_store, load=False)
                model.reload(progress)
                digest = (examples_digest(worker_store.examples())
                          if isinstance(trained, TrainedDetectionModel) else None)
                return model, digest
            finally:
                worker_store.close()

        future = self._feedback_load_executor.submit(load)
        self._feedback_load_future = future
        future.add_done_callback(lambda done: wx.CallAfter(
            self._finish_feedback_load, done, ui_changes))

    def _feedback_load_progress(self, done: int, total: int) -> None:
        if not self._closing and self._feedback_deferred:
            self.SetStatusText(f'Loading reviewed crops: {done:,}/{total:,}', 0)

    def _finish_feedback_load(self, future: Future[tuple[FeedbackModel, str | None]],
                              ui_changes: int) -> None:
        if self._closing or future is not self._feedback_load_future:
            return
        self._feedback_load_future = None
        try:
            model, digest = future.result()
        except Exception:
            getLog().exception('Could not load reviewed-crop descriptors')
            self.SetStatusText('Feedback loading failed; see console log', 0)
            return
        if self._feedback_store.db.total_changes != ui_changes:
            self._start_feedback_load()
            return
        model.store = self._feedback_store
        model.trained_model = self._feedback_model.trained_model
        model.version = max(model.version, self._feedback_model.version) + 1
        self.pause_playback()
        self._feedback_model = model
        self._feedback_deferred = False
        if (isinstance(model.trained_model, TrainedDetectionModel) and
                (digest != model.trained_model.review_digest or
                 model.trained_model.catalog_state !=
                 self._pin_catalog.stock_training_state())):
            getLog().warning(
                'Using saved classifier trained on older reviews or Pinpanion catalog; '
                'run training again to include the latest')
        self._update_feedback_status()
        if self._feedback_load_executor is not None:
            self._feedback_load_executor.shutdown(wait=False)
            self._feedback_load_executor = None

    def on_key_down(self, event: wx.KeyEvent):
        keycode = event.GetKeyCode()
        control_down = event.ControlDown()
        shift_down = event.ShiftDown()
        modified = (control_down or getattr(event, 'AltDown', lambda: False)() or
                    getattr(event, 'MetaDown', lambda: False)())
        focused = wx.Window.FindFocus()
        editing_text = isinstance(focused, (wx.TextCtrl, wx.ComboBox))
        selected = self.__image_panel.selected_boxes
        delete_keys = (wx.WXK_DELETE, getattr(wx, 'WXK_NUMPAD_DELETE', -1))
        wrong_identity_keys = (ord('X'), ord('x'))
        not_pin_keys = (ord('N'), ord('n'))
        confirm_identity_keys = (ord('Y'), ord('y'))
        shortcut = keycode in (*delete_keys, *wrong_identity_keys,
                               *not_pin_keys, *confirm_identity_keys)
        if keycode in delete_keys:
            actionable = selected
        elif keycode in wrong_identity_keys:
            actionable = selected if len(selected) == 1 and has_specific_identity(selected[0]) else []
        elif keycode in confirm_identity_keys:
            actionable = selected if (
                len(selected) == 1 and has_specific_identity(selected[0])
                and selected[0].review_state != 'rejected'
                and (selected[0].review_state != 'confirmed'
                     or not selected[0].identity_confirmed
                     or selected[0].identity_inferred)) else []
        else:
            actionable = [box for box in selected if box.source == 'automatic']
        if shortcut and actionable and not modified:
            if editing_text:
                event.Skip()
            elif keycode in delete_keys:
                self.remove_boxes(selected)
            elif keycode in wrong_identity_keys:
                self.mark_wrong_identity(actionable[0])
            elif keycode in confirm_identity_keys:
                self.confirm_pin_identity(actionable[0])
            else:
                self.mark_not_pin_boxes(selected)
            return
        if keycode in (ord('F'), ord('D')) and not modified and not editing_text:
            if keycode == ord('F'):
                self.on_next_async(event)
            else:
                self.on_prev(event)
            return
        # Ctrl+Z for undo
        if control_down and keycode == ord('Z') and not shift_down:
            self.undo_action()
        # Ctrl+Shift+Z for redo
        elif control_down and keycode == ord('Z') and shift_down:
            self.redo_action()
        else:
            event.Skip()

    @property
    def __current_boxes(self) -> List[BoxData]:
        """Get the boxes for the current frame index."""
        return self.__get_frame_boxes(self._current_index)

    def on_box_update(self) -> None:
        """Called when a box is updated, e.g., after adding or removing a tag."""
        self.__tag_panel.boxes = self.__current_boxes
        self._sync_presence_ui()
        self.Refresh()

    def _sync_presence_ui(self) -> None:
        flags = self._frame_presence.get(self._current_index, FramePresence())
        contains_pin, contains_set = checkbox_values(flags, self.__current_boxes)
        self.__tag_panel.set_presence(contains_pin, contains_set)

    def _on_presence_change(self, kind: str, checked: bool) -> None:
        self.pause_playback()
        self._begin_history()
        old = self._frame_presence.get(self._current_index, FramePresence())
        if kind == 'set':
            flags = FramePresence(old.pin_asserted or checked, checked)
        else:
            flags = FramePresence(checked, old.set_asserted if checked else False)
        if flags.pin_asserted or flags.set_asserted:
            self._frame_presence[self._current_index] = flags
        else:
            self._frame_presence.pop(self._current_index, None)
        self._finish_history()
        if checked and marked_pin_count(self.__current_boxes) < flags.minimum_pins:
            self._submit_frame_job(self._current_index, requires_play=False)

    def _begin_history(self) -> None:
        if self._history_pending is None:
            self._history_pending = (self._current_index, snapshot(self.__frame_boxes),
                                     self._rotation_angle, dict(self._frame_presence))

    def _finish_history(self) -> None:
        pending = self._history_pending
        self._history_pending = None
        if pending is None or pending[0] != self._current_index:
            return
        frame, before, rotation, presence = pending
        entry = make_entry(before, snapshot(self.__frame_boxes), rotation,
                           self._rotation_angle, presence,
                           dict(self._frame_presence))
        consolidated = False
        if any({'tags', 'pin_id', 'identity_confirmed', 'review_state', 'track_id',
                 'identity_rejections'}
               & change.changed_fields.keys() for change in entry.boxes):
            propagate_confirmations(self.__frame_boxes)
            affected_tracks: set[str] = set()
            affected_frames: set[int] = set()
            for change in entry.boxes:
                affected_tracks.add(change.box.track_id)
                if change.before_location is not None:
                    affected_frames.add(change.before_location[0])
                if change.after_location is not None:
                    affected_frames.add(change.after_location[0])
                old_track, new_track = change.changed_fields.get(
                    'track_id', (change.box.track_id, change.box.track_id))
                if isinstance(old_track, str):
                    affected_tracks.add(old_track)
                if isinstance(new_track, str):
                    affected_tracks.add(new_track)
            affected_frames.update(
                frame_index for frame_index, boxes in self.__frame_boxes.items()
                if any(box.track_id in affected_tracks for box in boxes))
            consolidated = self._consolidate_frames(affected_frames)
            entry = make_entry(before, snapshot(self.__frame_boxes), rotation,
                               self._rotation_angle, presence,
                               dict(self._frame_presence))
            self.__image_panel.Refresh()
            self.__tag_panel.boxes = self.__current_boxes
        if entry.changed:
            self._undo_history.setdefault(frame, []).append(entry)
            self._redo_history.pop(frame, None)
            self._sync_feedback_for_entry(entry)
            if consolidated:
                self._prune_linked_feedback()
            if any(change.box.identity_rejections or
                   'identity_rejections' in change.changed_fields
                   for change in entry.boxes):
                self._sync_identity_feedback_for_entry(entry)
            self._invalidate_reconciliation()
            if self._review_autosave_active:
                self.queue_review_save()
        self._sync_presence_ui()

    def _consolidate_frames(self, frame_indices: set[int] | None = None) -> bool:
        """Remove overlapping automatic duplicates and same-pin observations."""
        changed = False
        indices = self.__frame_boxes if frame_indices is None else frame_indices
        for frame_index in indices:
            boxes = self.__frame_boxes.get(frame_index)
            if boxes is None:
                continue
            kept, removed = consolidate_overlapping_boxes(boxes)
            if removed:
                boxes[:] = kept
                changed = True
        return changed

    def _sync_automatic_consolidation(self, *, invalidate: bool = True,
                                      frame_index: int | None = None) -> None:
        if frame_index is None:
            before = snapshot(self.__frame_boxes)
            if not self._consolidate_frames():
                return
            after = snapshot(self.__frame_boxes)
        else:
            # Playback changes only its destination frame. Avoid copying and
            # scanning thousands of unrelated saved observations each frame.
            boxes = self.__frame_boxes.get(frame_index, [])
            kept, removed = consolidate_overlapping_boxes(boxes)
            if not removed:
                return
            before = snapshot({frame_index: boxes})
            boxes[:] = kept
            after = snapshot({frame_index: boxes})
        entry = make_entry(before, after,
                           self._rotation_angle, self._rotation_angle)
        self._sync_feedback_for_entry(entry)
        self._prune_linked_feedback()
        if any(change.box.identity_rejections for change in entry.boxes):
            self._sync_identity_feedback_for_entry(entry)
        if invalidate:
            self._invalidate_reconciliation()

    def _prune_linked_feedback(self) -> None:
        stored = self._feedback_store.metadata_for_source(self._feedback_source)
        active = {item.example_id for item in reviewed_observations(
            self.__frame_boxes, self._feedback_source)
            if item.example_id.startswith(LINKED_POSITIVE_PREFIX)}
        removed = {example_id for example_id in stored
                   if example_id.startswith(LINKED_POSITIVE_PREFIX)
                   and example_id not in active}
        if removed:
            self._feedback_store.apply_source_delta(self._feedback_source, [], removed)
            self._feedback_model.refresh(removed)
            self._update_feedback_status()

    def undo_action(self) -> None:
        self.pause_playback()
        history = self._undo_history.get(self._current_index, [])
        if not history:
            return
        entry = history.pop()
        self._rotation_angle = entry.apply(self.__frame_boxes, undo=True,
                                           presence=self._frame_presence)
        self._invalidate_reconciliation()
        self._sync_feedback_for_entry(entry)
        self._sync_identity_feedback_for_entry(entry)
        self._redo_history.setdefault(self._current_index, []).append(entry)
        if self._review_autosave_active:
            self.queue_review_save()
        self.display_image()

    def redo_action(self) -> None:
        self.pause_playback()
        history = self._redo_history.get(self._current_index, [])
        if not history:
            return
        entry = history.pop()
        self._rotation_angle = entry.apply(self.__frame_boxes, undo=False,
                                           presence=self._frame_presence)
        self._invalidate_reconciliation()
        self._sync_feedback_for_entry(entry)
        self._sync_identity_feedback_for_entry(entry)
        self._undo_history.setdefault(self._current_index, []).append(entry)
        if self._review_autosave_active:
            self.queue_review_save()
        self.display_image()

    def _sync_feedback_for_entry(self, entry: HistoryEntry) -> None:
        affected: list[BoxData] = []
        ids: set[str] = set()
        for change in entry.boxes:
            old_id, new_id = change.changed_fields.get(
                'feedback_id', (change.box.feedback_id, change.box.feedback_id))
            ids.update(value for value in (old_id, new_id) if value)
            if change.box.feedback_id:
                ids.add(change.box.feedback_id)
                affected.append(change.box)
        if not ids:
            return
        if ((self._review_action_active or self._review_autosave_active) and
                self._feedback_store.path != ':memory:'):
            changes: dict[str, FeedbackExample | None] = {
                example_id: None for example_id in ids}
            for box in affected:
                example_id = box.feedback_id
                label = box.feedback_label
                if example_id is None or label not in ('positive', 'negative'):
                    continue
                for frame_index, boxes in self.__frame_boxes.items():
                    if any(existing is box for existing in boxes):
                        frame = self.get_frame(frame_index, 0)
                        if frame is not None:
                            try:
                                changes[example_id] = make_feedback_example(
                                    example_id, label, frame,
                                    box.coords, self._feedback_source, frame_index,
                                    box.track_id,
                                    box.pin_id if label == 'positive'
                                    and box.identity_confirmed else None,
                                    tuple(tag for tag in box.tags or [] if tag.strip())
                                    if label == 'positive' else ())
                            except ValueError as error:
                                getLog().warning('Skipping invalid feedback crop: %s', error)
                        break
            self._queue_review_feedback_changes(changes)
            return
        for example_id in ids:
            self._feedback_store.delete(example_id)
        for box in affected:
            if box.feedback_label not in ('positive', 'negative'):
                continue
            for frame_index, boxes in self.__frame_boxes.items():
                if any(existing is box for existing in boxes):
                    frame = self.get_frame(frame_index, 0)
                    if frame is not None:
                        try:
                            self._feedback_store.save(make_feedback_example(
                                box.feedback_id, box.feedback_label, frame, box.coords,
                                self._feedback_source, frame_index, box.track_id,
                                box.pin_id if box.feedback_label == 'positive'
                                and box.identity_confirmed else None,
                                tuple(tag for tag in box.tags or [] if tag.strip())
                                if box.feedback_label == 'positive' else ()))
                        except ValueError as error:
                            getLog().warning(f'Skipping invalid feedback crop: {error}')
                    break
        self._feedback_model.refresh(ids)
        self._update_feedback_status()

    def _sync_identity_feedback(self) -> None:
        """Index direct wrong-name reviews as crops, including after undo/load."""
        stored = {item.example_id: item for item in
                  self._feedback_store.identity_rejections_for_source(self._feedback_source)}
        active: set[str] = set()
        changed: set[str] = set()
        frames: dict[int, np.ndarray | None] = {}
        for frame_index, boxes in self.__frame_boxes.items():
            for box in boxes:
                for review in box.identity_rejections:
                    if review.example_id in active:
                        continue
                    active.add(review.example_id)
                    old = stored.get(review.example_id)
                    source_index = (review.frame_index if review.frame_index >= 0
                                    else frame_index)
                    if (old is not None and old.frame_index == source_index and
                            old.track_id == box.track_id and
                            old.coords == review.coords and old.pin_id == review.pin_id and
                            old.tags == review.tags):
                        continue
                    if source_index not in frames:
                        frames[source_index] = self.get_frame(source_index, 0)
                    frame = frames[source_index]
                    if frame is None:
                        continue
                    try:
                        self._feedback_store.save_identity_rejection(make_feedback_example(
                            review.example_id, 'wrong_identity', frame, review.coords,
                            self._feedback_source, source_index, box.track_id,
                            review.pin_id, review.tags))
                        changed.add(review.example_id)
                    except ValueError as error:
                        getLog().warning('Skipping invalid wrong-identity crop: %s', error)
        changed.update(set(stored) - active)
        self._feedback_store.delete_missing_identity_rejections(self._feedback_source,
                                                                active)
        if changed:
            self._feedback_model.refresh(changed)
            self._update_feedback_status()
        getLog().info('identity feedback: %d wrong-name examples indexed', len(active))

    def _sync_identity_feedback_for_entry(self, entry: HistoryEntry) -> None:
        if ((self._review_action_active or self._review_autosave_active) and
                self._feedback_store.path != ':memory:'):
            self._queue_review_identity_feedback(entry)
        else:
            self._sync_identity_feedback()

    def _queue_review_identity_feedback(self, entry: HistoryEntry) -> None:
        """Stage only wrong-ID reviews touched by this classification action."""
        ids: set[str] = set()
        for change in entry.boxes:
            ids.update(review.example_id for review in change.box.identity_rejections)
            values = change.changed_fields.get('identity_rejections')
            if values is not None:
                for value in values:
                    if isinstance(value, tuple):
                        ids.update(item.example_id for item in value
                                   if isinstance(item, IdentityRejection))
        if not ids:
            return
        changes: dict[str, FeedbackExample | None] = {
            example_id: None for example_id in ids}
        remaining = set(ids)
        frames: dict[int, np.ndarray | None] = {}
        for frame_index, boxes in self.__frame_boxes.items():
            for box in boxes:
                for review in box.identity_rejections:
                    if review.example_id not in remaining:
                        continue
                    remaining.remove(review.example_id)
                    source_index = (review.frame_index if review.frame_index >= 0
                                    else frame_index)
                    if source_index not in frames:
                        frames[source_index] = self.get_frame(source_index, 0)
                    frame = frames[source_index]
                    if frame is None:
                        changes.pop(review.example_id, None)
                        continue
                    try:
                        changes[review.example_id] = make_feedback_example(
                            review.example_id, 'wrong_identity', frame,
                            review.coords, self._feedback_source, source_index,
                            box.track_id, review.pin_id, review.tags)
                    except ValueError as error:
                        getLog().warning('Skipping invalid wrong-identity crop: %s', error)
                        changes.pop(review.example_id, None)
        self._queue_review_feedback_changes(changes)

    def _queue_review_feedback_changes(
            self, changes: dict[str, FeedbackExample | None]) -> None:
        if not changes:
            return
        self._review_feedback_changes.update(changes)
        self._review_feedback_failed = False
        if self._review_feedback_call is not None:
            self._review_feedback_call.Stop()
        self._review_feedback_call = wx.CallLater(1200, self._flush_review_feedback)

    def _flush_review_feedback(self) -> None:
        self._review_feedback_call = None
        if not self._review_feedback_changes or self._review_feedback_future is not None:
            return
        changes = self._review_feedback_changes
        self._review_feedback_changes = {}
        self._review_feedback_inflight = changes
        future = self._review_save_executor.submit(
            save_review_feedback, self._feedback_store.path,
            self._feedback_source, changes)
        self._review_feedback_future = future
        future.add_done_callback(lambda done: wx.CallAfter(
            self._complete_review_feedback, done))

    def _complete_review_feedback(self, future: Future[None]) -> bool:
        if self._destroy_requested or future is not self._review_feedback_future:
            return True
        self._review_feedback_future = None
        changes = self._review_feedback_inflight
        self._review_feedback_inflight = {}
        try:
            future.result()
        except Exception:
            getLog().exception('Could not save reviewed feedback crops')
            self._review_feedback_changes = {
                **changes, **self._review_feedback_changes}
            self._review_feedback_failed = True
            self.SetStatusText('Background feedback save failed; use File → Save', 1)
            return False
        self._feedback_model.refresh(set(changes))
        self._update_feedback_status()
        self._review_feedback_failed = False
        if self._review_feedback_changes and self._review_feedback_call is None:
            self._review_feedback_call = wx.CallLater(
                300, self._flush_review_feedback)
        return True

    def _seed_feedback_from_annotations(self) -> None:
        """Import explicit and legacy reviewed boxes once per source observation."""
        rejected_tracks = {box.track_id for boxes in self.__frame_boxes.values()
                           for box in boxes if box.feedback_label == 'negative'}
        protected_tracks = {box.track_id for boxes in self.__frame_boxes.values()
                            for box in boxes if box.source == 'user' or
                            box.review_state == 'confirmed'}
        stored = self._feedback_store.metadata_for_source(self._feedback_source)
        active_ids: set[str] = set()
        changed_ids: set[str] = set()
        for frame_index, boxes in sorted(self.__frame_boxes.items()):
            for box in boxes:
                if box.feedback_label is None:
                    if box.review_state == 'confirmed':
                        box.feedback_label = 'positive'
                    elif (box.review_state == 'rejected' and
                          box.track_id not in rejected_tracks and
                          box.track_id not in protected_tracks):
                        box.feedback_label = 'negative'
                        rejected_tracks.add(box.track_id)
                if box.feedback_label not in ('positive', 'negative'):
                    continue
                if not box.feedback_id:
                    key = f'{self._feedback_source}|{frame_index}|{box.track_id}|{box.feedback_label}'
                    box.feedback_id = uuid5(NAMESPACE_URL, key).hex
                if stored.get(box.feedback_id) == (box.feedback_label, frame_index,
                                                   box.track_id,
                                                   box.pin_id if box.feedback_label == 'positive'
                                                   and box.identity_confirmed else None,
                                                   box.coords,
                                                   tuple(tag for tag in box.tags or []
                                                         if tag.strip())
                                                   if box.feedback_label == 'positive' else ()):
                    active_ids.add(box.feedback_id)
                    continue
                frame = self.get_frame(frame_index, 0)
                if frame is None:
                    continue
                try:
                    self._feedback_store.save(make_feedback_example(
                        box.feedback_id, box.feedback_label, frame, box.coords,
                        self._feedback_source, frame_index, box.track_id,
                        box.pin_id if box.feedback_label == 'positive'
                        and box.identity_confirmed else None,
                        tuple(tag for tag in box.tags or [] if tag.strip())
                        if box.feedback_label == 'positive' else ()))
                    active_ids.add(box.feedback_id)
                    changed_ids.add(box.feedback_id)
                except ValueError as error:
                    getLog().warning(f'Skipping invalid feedback crop: {error}')
        for item in reviewed_observations(self.__frame_boxes, self._feedback_source):
            if (item.example_id.startswith(LINKED_POSITIVE_PREFIX) and
                    stored.get(item.example_id) == observation_metadata(item)):
                active_ids.add(item.example_id)
        changed_ids.update(set(stored) - active_ids)
        self._feedback_store.delete_missing_source(self._feedback_source, active_ids)
        if changed_ids:
            self._feedback_model.refresh(changed_ids)
            self._update_feedback_status()

    def _discard_stale_local_classifier(self) -> None:
        trained = self._feedback_model.trained_model
        if (isinstance(trained, TrainedDetectionModel) and
                (trained.catalog_state != self._pin_catalog.stock_training_state() or
                 trained.review_digest !=
                 examples_digest(self._feedback_store.examples()))):
            self._feedback_model.trained_model = None
            getLog().warning('Saved classifier became stale after review or catalog changes')

    def _update_feedback_status(self) -> None:
        stored = self._feedback_store.counts()
        usable = self._feedback_model.sample_counts
        message = (
            f"Feedback v{self._feedback_model.version}: "
            f"pin {usable['positive']}/{stored['positive']}, "
            f"non-pin {usable['negative']}/{stored['negative']} usable, "
            f"identity {self._feedback_model.identity_sample_count} usable, "
            f"wrong ID {self._feedback_store.identity_rejection_count(self._feedback_source)} saved, "
            f"OpenCV classifier {'active' if self._feedback_model.trained_model else 'not active'}"
        )
        status = self.GetStatusBar()
        if status is not None:
            status.SetStatusText(message, 0)
        getLog().info('feedback model updated %s', message)

    def show_feedback_info(self) -> None:
        stored = self._feedback_store.counts()
        usable = self._feedback_model.sample_counts
        message = (
            f"In-memory example matcher: revision {self._feedback_model.version}\n"
            f"Confirmed pins: {stored['positive']} saved, {usable['positive']} usable\n"
            f"Rejected automatic candidates: {stored['negative']} saved, "
            f"{usable['negative']} usable\n"
            f"Verified identity examples: {self._feedback_model.identity_sample_count} usable\n"
            f"Wrong pin identities: {self._feedback_store.identity_rejection_count(self._feedback_source)} "
            "saved for this source in the separate identity_rejections table\n\n"
            "Confirm is a pin adds a positive crop. Not a pin on an automatic "
            "area adds a negative crop. Remove area does not label a location "
            "as non-pin. Undo and redo update "
            "the saved examples and refresh the feedback model immediately. "
            "Rejecting a suggested identity immediately saves one wrong-ID crop per pin, "
            "including every candidate in a None of these menu page. "
            "Add to training data indexes new or changed session reviews "
            "and removes examples no longer present. Run training now also fits "
            "and saves an OpenCV pin/non-pin classifier from usable reviewed crops "
            "and Pinpanion stock images.\n\n"
            "The in-memory nearest-neighbour baseline is not a trained neural detector. "
            "It ranks pin candidates and compares confirmed "
            "pin IDs with reviewed image crops when catalog matching is weak. "
            "Wrong-ID crops block a visually similar wrong suggestion. "
            "Identity suggestions remain unconfirmed and their scores are "
            "similarities, not calibrated probabilities. The OpenCV classifier "
            "downranks predicted non-pin proposals when active; a prediction alone "
            "does not discard a rectangle. Neither method is a "
            "neural object detector or can find a pin that the proposal stage "
            "misses. Low-detail crops remain saved but are not usable by these models."
        )
        wx.MessageBox(message, 'Detection feedback', wx.OK | wx.ICON_INFORMATION)

    def show_settings(self, _event: wx.CommandEvent | None = None) -> None:
        trained = self._feedback_model.trained_model
        dialog = SettingsDialog(
            self,
            feedback_path=self._feedback_store.path,
            catalog_path=self._pin_catalog.path,
            annotations_path=self.box_data_filename,
            feedback_counts=self._feedback_store.counts(),
            wrong_identity_count=self._feedback_store.identity_rejection_count(),
            model_revision=self._feedback_model.version,
            model_path=self._trained_model_path,
            model_active=trained is not None,
            model_current=(isinstance(trained, TrainedDetectionModel) and
                           trained.review_digest ==
                           examples_digest(self._feedback_store.examples()) and
                           trained.catalog_state ==
                           self._pin_catalog.stock_training_state()),
            catalog_training_summary=(
                f'{trained.catalog_usable}/{trained.catalog_total} Pinpanion stock '
                f'images used; {trained.catalog_unverified} with unverified rights. '
                'Original Pinpanion names are saved as tags with source and pin IDs '
                'in model metadata.'
                if isinstance(trained, TrainedDetectionModel) else
                'No locally trained catalog-inclusive classifier is active.'),
            cloud_origin=self._cloud_config.origin,
            cloud_status=self._cloud_status,
            cloud_cache_path=(self._trained_model_path.with_suffix('.cloud.json')
                              if self._trained_model_path is not None else None),
            detection_resolution=self._detection_resolution,
        )
        try:
            dialog.ShowModal()
            chosen = dialog.detection_resolution
            if chosen in ('regular', 'low') and chosen != self._detection_resolution:
                self._detection_resolution = chosen
                save_detection_resolution(chosen)
        finally:
            dialog.Destroy()

    def save_annotations(self, _event: wx.CommandEvent | None = None) -> bool:
        """Save the current annotations without ending the editing session."""
        filename = self.box_data_filename
        if filename is None:
            self.SetStatusText('No annotation file selected to save', 1)
            return False
        try:
            self.__tag_panel.commit_pending_labels()
            if self._review_feedback_call is not None:
                self._review_feedback_call.Stop()
                self._review_feedback_call = None
            while self._review_feedback_changes or self._review_feedback_future is not None:
                if self._review_feedback_future is None:
                    if self._review_feedback_call is not None:
                        self._review_feedback_call.Stop()
                        self._review_feedback_call = None
                    self._flush_review_feedback()
                pending_feedback = self._review_feedback_future
                if pending_feedback is None:
                    break
                if not self._complete_review_feedback_after_wait(pending_feedback):
                    return False
            if self._review_save_call is not None:
                self._review_save_call.Stop()
                self._review_save_call = None
            if self._review_save_future is not None:
                try:
                    self._review_save_future.result()
                except Exception:
                    # The explicit save below retries from the current state.
                    getLog().exception('Earlier background annotation save failed')
                self._review_save_future = None
            save_boxes_to_file(filename, self.__frame_boxes, self._frame_presence)
        except Exception:
            getLog().exception('Could not save annotations to %s', filename)
            self.SetStatusText('Save failed; see console log', 1)
            return False
        self._review_save_pending = False
        count = self.count_boxes()
        getLog().info('%d boxes saved to %s', count, filename)
        self.SetStatusText(f'Saved {count} areas to {os.path.basename(filename)}', 1)
        return True

    def _complete_review_feedback_after_wait(self, future: Future[None]) -> bool:
        """Finish an explicit save after its earlier feedback write finishes."""
        try:
            future.result()
        except Exception:
            pass  # The completion handler logs and retains the unsaved changes.
        return self._complete_review_feedback(future)

    def queue_review_save(self) -> None:
        """Coalesce quick reviews and persist an immutable snapshot off the UI thread."""
        if self.box_data_filename is None or self._closing:
            return
        self._review_autosave_active = True
        self._review_save_pending = True
        if self._review_save_call is not None:
            self._review_save_call.Stop()
        self._review_save_call = wx.CallLater(1200, self._flush_review_save)

    def _flush_review_save(self) -> None:
        self._review_save_call = None
        if (self._closing or not self._review_save_pending or
                self.box_data_filename is None or
                self._review_save_future is not None):
            return
        snapshot = {
            frame_index: [self._copy_box_for_worker(box) for box in boxes]
            for frame_index, boxes in self.__frame_boxes.items()
        }
        presence = dict(self._frame_presence)
        filename = self.box_data_filename
        self._review_save_pending = False
        future = self._review_save_executor.submit(
            save_boxes_to_file, filename, snapshot, presence)
        self._review_save_future = future
        future.add_done_callback(lambda done: wx.CallAfter(
            self._complete_review_save, done, filename))

    def _complete_review_save(self, future: Future[None], filename: str) -> None:
        if self._destroy_requested or future is not self._review_save_future:
            return
        self._review_save_future = None
        try:
            future.result()
        except Exception:
            getLog().exception('Could not save reviewed annotations to %s', filename)
            self._review_save_pending = True
            self.SetStatusText('Background save failed; use File → Save', 1)
            return
        if self._review_save_pending:
            if self._review_save_call is None:
                self._review_save_call = wx.CallLater(300, self._flush_review_save)
        else:
            self.SetStatusText(
                f'Reviewed areas saved to {os.path.basename(filename)}', 1)

    def classifier_review_areas(self) -> list[ReviewArea]:
        return saved_review_areas(self.__frame_boxes)

    @contextmanager
    def classifier_review_action(self) -> Iterator[None]:
        """Stage feedback I/O while one review action updates annotations."""
        self._review_action_active = True
        try:
            yield
        finally:
            self._review_action_active = False

    def classifier_review_catalog_options(self) -> list[tuple[str, str]]:
        return self._pin_catalog.list_pins()

    def classifier_review_rank(self, crop: np.ndarray, box: BoxData,
                               limit: int = 10) -> list[PinMatch]:
        """Rank the crop itself, without the replacement menu's ID injection."""
        return self._pin_matcher.rank_matches(
            crop, limit=limit,
            excluded_pin_ids=frozenset(item.pin_id for item in box.identity_rejections
                                       if item.pin_id),
            excluded_names=frozenset(tag.casefold()
                                     for item in box.identity_rejections
                                     if item.pin_id is None for tag in item.tags))

    def classifier_review_open(self, area: ReviewArea) -> np.ndarray | None:
        """Show only an already saved frame and return this area's RGB crop."""
        index, box = area.frame_index, area.box
        if not any(existing is box for existing in self.__frame_boxes.get(index, [])):
            return None
        frame = self.get_frame(index, 0)
        if frame is None:
            return None
        x, y, width, height = box.coords
        if (x < 0 or y < 0 or width <= 0 or height <= 0 or
                x + width > frame.shape[1] or y + height > frame.shape[0]):
            return None
        self._current_index = index
        self.slider.SetValue(index)
        self.display_image(frame, reconcile=False, catalog_matching=False)
        return frame[y:y + height, x:x + width].copy()

    def classifier_review_pending(self, index: int) -> bool:
        """Wait for a review's catalog retry and linked-frame reconciliation."""
        return (index in self._pending_catalog or
                (self._pending_reconcile_index == index and
                 self._pending_reconcile is not None))

    def open_classifier_review(self, _event: wx.CommandEvent | None = None) -> None:
        self.pause_playback()
        self.__tag_panel.commit_pending_labels()
        if self.box_data_filename is None:
            self.SetStatusText('Select an annotation JSON file before reviewing saved areas', 1)
            return
        areas = self.classifier_review_areas()
        if not areas:
            self.SetStatusText('No saved rectangles to review', 1)
            return
        dialog = ClassifierReviewDialog(self, areas,
                                        self.classifier_review_catalog_options())
        try:
            dialog.ShowModal()
        finally:
            dialog.Destroy()

    def _on_cloud_timer(self, _event: wx.TimerEvent) -> None:
        if self._training_future is None and self._model_training_future is None:
            self._start_cloud_sync()

    def _start_cloud_sync(self, *, then_train: bool = False) -> None:
        if (self._closing and not then_train) or self._trained_model_path is None:
            if then_train:
                self.add_to_training_data(then_train=True)
            return
        if self._cloud_sync_future is not None:
            if then_train:
                self._cloud_train_after_sync = True
            return
        self._cloud_train_after_sync = then_train
        cursor = cloud_cursor(self._feedback_store)
        config = self._cloud_config

        def download() -> tuple[tuple[int, PortableDetectionModel] | None,
                                list[dict], int, bool]:
            model = fetch_model(config)
            events: list[dict] = []
            after = cursor
            if config.token:
                for _ in range(10):  # Bound memory to 50 events per UI commit.
                    page, next_cursor = fetch_review_page(config, after)
                    events.extend(page)
                    if next_cursor <= after or len(page) < 5:
                        after = next_cursor
                        break
                    after = next_cursor
                else:
                    return model, events, after, True
            return model, events, after, False

        future = self._sync_executor.submit(download)
        self._cloud_sync_future = future
        future.add_done_callback(lambda done: wx.CallAfter(self._finish_cloud_sync, done))

    def _finish_cloud_sync(self, future: Future) -> None:
        if (getattr(self, '_destroy_requested', False) or not self or
                future is not self._cloud_sync_future):
            return
        self._cloud_sync_future = None
        then_train = self._cloud_train_after_sync
        self._cloud_train_after_sync = False
        try:
            model, events, cursor, has_more = future.result()
            changed = apply_reviews(self._feedback_store, events, cursor) if self._cloud_config.token else 0
            if changed:
                if self._feedback_deferred:
                    # The background loader will see the UI connection's changed
                    # revision and retry. A synchronous reload here would decode
                    # the full store before the startup window can respond.
                    self._feedback_model.trained_model = None
                else:
                    self._feedback_model.reload()
                    self._discard_stale_local_classifier()
                self._update_feedback_status()
            if model is not None and model[0] > self._cloud_model_version:
                version, classifier = model
                cache_path = self._trained_model_path.with_suffix('.cloud.json')
                temporary = cache_path.with_suffix('.cloud.tmp')
                try:
                    temporary.write_text(json.dumps({
                        'cloud_version': version, 'model': classifier.web_payload()
                    }, separators=(',', ':')), encoding='utf-8')
                    os.replace(temporary, cache_path)
                finally:
                    temporary.unlink(missing_ok=True)
                self._cloud_model_version = version
                if not isinstance(self._feedback_model.trained_model,
                                  TrainedDetectionModel):
                    self._feedback_model.trained_model = classifier
                    self._feedback_model.version += 1
                    self._update_feedback_status()
            self._cloud_ready_for_publish = bool(self._cloud_config.token)
            self._cloud_status = (f'Model v{self._cloud_model_version}; {changed} web training crops '
                                  f'changed; through revision {cursor}')
            self.SetStatusText(f'Cloud sync: {self._cloud_status}', 1)
            if has_more:
                self._start_cloud_sync(then_train=then_train)
                return
        except Exception as exc:
            self._cloud_ready_for_publish = False
            self._cloud_status = 'Unavailable; local annotations and model remain usable'
            if isinstance(exc, CloudSyncError):
                getLog().warning('Cloudflare model/review sync unavailable: %s', exc)
            else:
                getLog().exception('Cloudflare model/review sync failed')
            self.SetStatusText(f'Cloud sync: {self._cloud_status}', 1)
        if then_train:
            self.add_to_training_data(then_train=True)

    def _finish_cloud_publish(self, future: Future) -> None:
        if self._closing or future is not self._cloud_publish_future:
            return
        self._cloud_publish_future = None
        try:
            version = future.result()
            self._cloud_model_version = version
            self._cloud_status = f'Model v{version} published for the web app'
            self.SetStatusText(self._cloud_status, 1)
        except Exception:
            self._cloud_status = 'Model trained locally; cloud publish failed'
            getLog().exception('Could not publish trained model to Cloudflare')
            self.SetStatusText(self._cloud_status, 1)

    def run_training_now(self) -> None:
        """Index session reviews, then fit and persist the OpenCV classifier."""
        if self._trained_model_path is None:
            self._set_training_progress(
                0, 'Training needs a persistent feedback database.', False)
            return
        if self._cloud_config.token:
            self._start_cloud_sync(then_train=True)
        else:
            self.add_to_training_data(then_train=True)

    def add_to_training_data(self, then_train: bool = False) -> None:
        """Index only new and changed direct reviews from this session."""
        if (self._training_future is not None or
                self._model_training_future is not None):
            return
        self.pause_playback()
        if self.box_data_filename and not self.save_annotations():
            self._set_training_progress(
                0, 'Could not save reviews before training; see console log.', False)
            return
        observations = reviewed_observations(self.__frame_boxes, self._feedback_source)
        observations_by_id = {item.example_id: item for item in observations}
        try:
            delta = plan_training_delta(
                observations,
                self._feedback_store.training_metadata_for_source(self._feedback_source))
        except ValueError:
            getLog().exception('Could not plan incremental training update')
            self._set_training_progress(
                0, 'Training data has duplicate example IDs; see console log.', False)
            return
        if not delta.pending:
            if delta.removed_ids:
                self._feedback_store.apply_source_delta(
                    self._feedback_source, [], delta.removed_ids)
                self._feedback_model.refresh(delta.removed_ids)
                self._update_feedback_status()
                message = (f'100% · {len(delta.removed_ids)} removed\n'
                           f'{self._training_index_summary(observations_by_id, delta.active_ids)}')
            else:
                message = ('100% · Up to date · 0 added now; reviews saved when marked\n'
                           f'{self._training_index_summary(observations_by_id, delta.active_ids)}')
            self._set_training_progress(100, message, False)
            if then_train:
                self._start_model_training()
            return
        revision = self._annotation_revision
        self._set_training_progress(
            0, f'Preparing {len(delta.pending)} new or changed examples...', True)

        def progress(done: int, total: int, eta: float | None) -> None:
            percent = round(90 * done / total) if total else 90
            remaining = 'estimating...' if eta is None else f'{eta:.0f}s remaining'
            wx.CallAfter(self._show_training_progress, percent,
                         f'{percent}% · {done}/{total} crops · {remaining}')

        future = self._play_executor.submit(
            build_training_examples, delta.pending, self._feedback_source,
            self.read_frame_for_processing, progress)
        self._training_future = future
        future.add_done_callback(lambda finished: wx.CallAfter(
            self._finish_training, finished, revision, delta, observations_by_id,
            then_train))

    @staticmethod
    def _training_index_summary(observations_by_id: dict[str, TrainingObservation],
                                ids: set[str]) -> str:
        indexed = [observations_by_id[item] for item in ids]
        pin_crops = [item for item in indexed if item.label == 'positive']
        linked_views = sum(item.example_id.startswith(LINKED_POSITIVE_PREFIX)
                           for item in pin_crops)
        without_verified_id = sum(item.pin_id is None for item in pin_crops)
        non_pin_crops = sum(item.label == 'negative' for item in indexed)
        wrong_ids = sum(item.label == 'wrong_identity' for item in indexed)
        return (f'{len(ids)} examples indexed: {len(pin_crops)} pin crops '
                f'(linked frames: {linked_views}, '
                f'{without_verified_id} without verified ID), '
                f'{non_pin_crops} non-pin crops, '
                f'{wrong_ids} wrong IDs')

    def _show_training_progress(self, percent: int, message: str) -> None:
        if (self._closing and self._shutdown_dialog is not None and
                self._shutdown_stage == 1):
            self._shutdown_dialog.set_percent(round(45 * percent / 100))
        elif self._training_future is not None:
            self._set_training_progress(percent, message, True)

    def _finish_training(self, future: Future, revision: int,
                         delta: TrainingDelta,
                         observations_by_id: dict[str, TrainingObservation],
                         then_train: bool = False) -> None:
        if future is not self._training_future:
            return
        self._training_future = None
        try:
            examples, skipped = future.result()
            if revision != self._annotation_revision:
                self._set_training_progress(
                    0, 'Annotations changed; run training again.', False)
                return
            self._set_training_progress(
                95, '95% · refreshing example matcher · estimating...', True)
            indexed_ids = {example.example_id for example in examples}
            failed_ids = {item.example_id for item in delta.pending} - indexed_ids
            removed_ids = delta.removed_ids | (failed_ids & delta.stored_ids)
            changed_ids = indexed_ids | removed_ids
            if changed_ids:
                self._feedback_store.apply_source_delta(
                    self._feedback_source, examples, removed_ids)
                self._feedback_model.refresh(changed_ids)
                self._update_feedback_status()
            added = len(indexed_ids - delta.stored_ids)
            updated = len(indexed_ids & delta.stored_ids)
            self._set_training_progress(
                100, f'100% · {added} added now, {updated} updated, {len(removed_ids)} removed'
                + (f' · {skipped} skipped' if skipped else '')
                + f'\n{self._training_index_summary(observations_by_id, delta.active_ids - failed_ids)}',
                False)
            if then_train:
                self._start_model_training()
        except Exception:
            getLog().exception('Could not add session annotations to training data')
            self._set_training_progress(
                0, 'Adding training data failed; see console log.', False)

    def _start_model_training(self) -> None:
        """Pass an immutable crop list to OpenCV on the worker, never SQLite."""
        if self._trained_model_path is None:
            return
        examples = self._feedback_store.examples()
        counts = self._feedback_store.counts()
        stock_count = self._pin_catalog.stock_training_state()[0]
        if not counts['negative'] or (not counts['positive'] and not stock_count):
            self._set_training_progress(
                0, 'Need at least one usable pin image and one reviewed non-pin crop.', False)
            return
        digest = examples_digest(examples)
        previous_model = None
        if self._trained_model_path.is_file():
            try:
                previous_model = TrainedDetectionModel.load(self._trained_model_path)
            except (OSError, ValueError, KeyError, cv2.error):
                getLog().warning('Saved classifier cannot be continued; fitting from stored crops',
                                 exc_info=True)
        self._training_base_cloud_version = self._cloud_model_version
        self._set_training_progress(
            0, f"Fitting OpenCV classifier from {counts['positive']} reviewed pins, "
               f"{stock_count} Pinpanion stock images and "
               f"{counts['negative']} non-pins...", True)
        def progress(done: int, total: int) -> None:
            wx.CallAfter(self._show_model_training_progress, done, total)
        future = self._play_executor.submit(
            TrainedDetectionModel.train, examples, self._pin_catalog.path, progress,
            previous_model)
        self._model_training_future = future
        future.add_done_callback(lambda finished: wx.CallAfter(
            self._finish_model_training, finished, digest,
            previous_model.training_digest if previous_model else None,
            previous_model.incremental_updates if previous_model else 0))

    def _show_model_training_progress(self, done: int, total: int) -> None:
        if (self._closing and self._shutdown_dialog is not None and
                self._shutdown_stage == 1):
            self._shutdown_dialog.set_percent(
                50 + (round(40 * done / total) if total else 40))
        elif self._model_training_future is not None:
            percent = round(80 * done / total) if total else 80
            self._set_training_progress(
                percent, f'{percent}% · {done}/{total} Pinpanion stock images '
                         'prepared for classifier...', True)

    def _finish_model_training(self, future: Future, digest: str,
                               previous_digest: str | None,
                               previous_updates: int) -> None:
        if future is not self._model_training_future:
            return
        self._model_training_future = None
        try:
            model = future.result()
            if digest != examples_digest(self._feedback_store.examples()):
                self._set_training_progress(
                    0, 'Reviews changed during training; run training again.', False)
                return
            if model.catalog_state != self._pin_catalog.stock_training_state():
                self._set_training_progress(
                    0, 'Pinpanion catalog changed during training; run again.', False)
                return
            model.save(self._trained_model_path)
            # A detection queued while training used an earlier feedback snapshot.
            # Invalidate it before publishing the new model to future jobs.
            self.pause_playback()
            self._feedback_model.trained_model = model
            self._feedback_model.version += 1
            self._update_feedback_status()
            mode = ('already current' if model.training_digest == previous_digest else
                    'incrementally updated' if model.incremental_updates > previous_updates
                    else 'rebuilt from current crops')
            self._set_training_progress(
                100, f"Pin/non-pin classifier {mode}: "
                     f"{model.sample_counts['positive'] - model.catalog_usable} "
                     f"reviewed pin crops + {model.catalog_usable}/{model.catalog_total} "
                     f"Pinpanion stock images, "
                     f"{model.sample_counts['negative']} non-pins.", False)
            self.SetStatusText('Trained pin/non-pin classifier is active for future detections', 1)
            if model.catalog_unverified:
                self.SetStatusText(
                    'Classifier trained locally; Pinpanion image rights are unverified, '
                    'so web publication is held', 1)
            elif (not self._closing and self._cloud_ready_for_publish and
                  self._cloud_config.token):
                payload = model.web_payload()
                payload['web_review_revision'] = cloud_cursor(self._feedback_store)
                future = self._sync_executor.submit(
                    publish_model, self._cloud_config, payload,
                    self._training_base_cloud_version)
                self._cloud_publish_future = future
                future.add_done_callback(lambda done: wx.CallAfter(
                    self._finish_cloud_publish, done))
                self.SetStatusText('Classifier trained; publishing web model...', 1)
            elif not self._cloud_config.token:
                self.SetStatusText(
                    'Classifier trained locally; set PINIDENT_SYNC_TOKEN to publish online', 1)
        except ValueError as exc:
            self._set_training_progress(0, str(exc), False)
        except Exception:
            getLog().exception('Could not train or save the OpenCV classifier')
            self._set_training_progress(
                0, 'Classifier training failed; see console log.', False)

    def set_pin_matcher(self, matcher: PinMatcher) -> None:
        """Install a rebuilt reference index on the GUI thread."""
        pending_targets = list(self._pending_catalog_targets.items())
        self._pin_matcher = matcher
        self._discard_stale_local_classifier()
        self.__tag_panel.set_catalog_options(self._pin_catalog.list_pins())
        self._cancel_catalog_matches()
        self.__matched_frames.clear()
        self._invalidate_reconciliation()
        for index, (box, best_effort) in pending_targets:
            if (index == self._current_index and
                    any(existing is box for existing in self.__current_boxes)):
                if not self._schedule_catalog_match(
                        index, None, None, target_box=box,
                        best_effort=best_effort):
                    self.SetStatusText(
                        f'Frame {index}: no catalog images or reviewed pin identities are loaded', 1)
        if not self.IsBeingDeleted():
            self.display_image()

    def file_select(self, evt: wx.CommandEvent) -> None:
        """Load an annotation JSON file for the currently open video."""
        self.pause_playback()
        with wx.FileDialog(self, "Select annotation JSON", wildcard="JSON files|*.json",
                           style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as fileDialog:
            if fileDialog.ShowModal() == wx.ID_OK:
                path = fileDialog.GetPath()
                if os.path.exists(path):
                    self.box_data_filename = path
                    self.__frame_boxes = self.load_box_data()
                    self.display_image()
                else:
                    wx.MessageBox("File does not exist.", "Error", wx.OK | wx.ICON_ERROR)

    @current_index.setter
    def current_index(self, index: int):
        if 0 <= index < self.__num_frames:
            reference_index = self._current_index if abs(index - self._current_index) == 1 else None
            self._current_index = index
            if self.slider is not None:
                self.slider.SetValue(index)
                self.display_image(reference_index=reference_index)
        else:
            raise ValueError("Index out of bounds")

    def count_boxes(self):
        count = 0
        for box_list in self.__frame_boxes.values():
            count += len(box_list)
        """Count the number of boxes in the current frame."""
        return count

    def on_close(self, event: wx.CloseEvent) -> None:
        if self._shutdown_dialog is not None:
            if event.CanVeto():
                event.Veto()
            return
        if not event.CanVeto():
            event.Skip()
            return
        event.Veto()
        self._closing = True
        if self._review_save_call is not None:
            self._review_save_call.Stop()
            self._review_save_call = None
        if self._review_feedback_call is not None:
            self._review_feedback_call.Stop()
            self._review_feedback_call = None
        self._feedback_load_cancel.set()
        self._cloud_timer.Stop()
        self._shutdown_dialog = ShutdownProgressDialog(self)
        self._shutdown_timer = wx.Timer(self._shutdown_dialog)
        self._shutdown_dialog.Bind(wx.EVT_TIMER, self._shutdown_tick,
                                   self._shutdown_timer)
        self._shutdown_dialog.Show()
        self.Enable(False)
        self._shutdown_timer.Start(150)
        wx.CallAfter(self._shutdown_tick)

    def Destroy(self) -> bool:
        """Stop callbacks when a test or parent destroys the frame directly."""
        self._destroy_requested = True
        self._closing = True
        self._feedback_load_cancel.set()
        self._cloud_timer.Stop()
        self._play_timer.Stop()
        if self._review_save_call is not None:
            self._review_save_call.Stop()
            self._review_save_call = None
        if self._review_feedback_call is not None:
            self._review_feedback_call.Stop()
            self._review_feedback_call = None
        self._shutdown_stop_prefetch()
        self._play_executor.shutdown(wait=False, cancel_futures=True)
        self._sync_executor.shutdown(wait=False, cancel_futures=True)
        self._review_save_executor.shutdown(wait=False, cancel_futures=True)
        if self._feedback_load_executor is not None:
            self._feedback_load_executor.shutdown(wait=False, cancel_futures=True)
        return super().Destroy()

    def _shutdown_stop_prefetch(self) -> bool:
        """Overridden by VideoScrubber, which owns the prefetch thread."""
        return True

    def _shutdown_resume_prefetch(self) -> None:
        """Restore optional prefetch after a failed close save."""

    def _abort_failed_close(self, message: str) -> None:
        dialog = self._shutdown_dialog
        if dialog is None:
            return
        if self._shutdown_save_executor is not None:
            self._shutdown_save_executor.shutdown(wait=False)
            self._shutdown_save_executor = None
        if self._shutdown_timer is not None:
            self._shutdown_timer.Stop()
        self.Enable(True)
        dialog.Destroy()
        self._shutdown_dialog = None
        self._closing = False
        self._feedback_load_cancel.clear()
        if self._feedback_deferred:
            self._feedback_load_future = None
            self._start_feedback_load()
        self._shutdown_stage = 0
        self._shutdown_prepared = False
        self._shutdown_save_future = None
        self._cloud_timer.Start(5 * 60 * 1000)
        self._shutdown_resume_prefetch()
        wx.MessageBox(message, 'Close failed', wx.OK | wx.ICON_ERROR)

    def _shutdown_tick(self, _event: wx.TimerEvent | None = None) -> None:
        dialog = self._shutdown_dialog
        if dialog is None:
            return
        if self._shutdown_stage == 0:
            if not self._shutdown_prepared:
                self.pause_playback()
                self.__tag_panel.commit_pending_labels()
                self._shutdown_prepared = True
            if not self._shutdown_stop_prefetch():
                return
            dialog.set_percent(100)
            dialog.advance()
            self._shutdown_stage = 1
        if self._shutdown_stage == 1:
            if self._review_feedback_failed:
                self._abort_failed_close(
                    'Reviewed feedback could not be saved. Pinident remains open; '
                    'use File → Save and try closing again.')
                return
            if self._review_feedback_future is None and self._review_feedback_changes:
                self._flush_review_feedback()
            if self._review_feedback_future is not None:
                if not self._review_feedback_future.done():
                    dialog.set_percent(10)
                    return
                self._complete_review_feedback(self._review_feedback_future)
                if self._review_feedback_failed or self._review_feedback_changes:
                    return
            # Let a requested training/indexing run finish and commit on the wx
            # thread before placing the final queue barriers.
            if (self._cloud_sync_future is not None or
                    self._training_future is not None or
                    self._model_training_future is not None or
                    (self._feedback_load_future is not None and
                     not self._feedback_load_future.done())):
                dialog.set_percent(10)
                return
            if self._shutdown_play_barrier is None:
                self._shutdown_play_barrier = self._play_executor.submit(lambda: None)
                self._shutdown_cloud_barrier = self._sync_executor.submit(lambda: None)
            complete = sum(future.done() for future in
                           (self._shutdown_play_barrier, self._shutdown_cloud_barrier))
            review_saved = (self._review_save_future is None or
                            self._review_save_future.done())
            dialog.set_percent(10 + 30 * (complete + review_saved))
            if complete != 2 or not review_saved:
                return
            dialog.set_percent(100)
            dialog.advance()
            self._shutdown_stage = 2
        if self._shutdown_stage == 2:
            if self._shutdown_save_future is None:
                if self.box_data_filename is None:
                    dialog.set_percent(100)
                    dialog.advance()
                    self._shutdown_stage = 3
                else:
                    self._shutdown_save_executor = ThreadPoolExecutor(
                        max_workers=1, thread_name_prefix='pin-save')
                    def update(percent: int) -> None:
                        wx.CallAfter(self._shutdown_save_progress, percent)
                    self._shutdown_save_future = self._shutdown_save_executor.submit(
                        save_boxes_to_file, self.box_data_filename,
                        self.__frame_boxes, self._frame_presence, update)
                    return
            else:
                if not self._shutdown_save_future.done():
                    return
                try:
                    self._shutdown_save_future.result()
                except Exception:
                    getLog().exception('Could not save annotations during close')
                    self._abort_failed_close(
                        'Annotations could not be saved. Pinident remains open; '
                        'use File → Save and try closing again.')
                    return
                getLog().info('%d boxes saved to %s', self.count_boxes(),
                              self.box_data_filename)
                dialog.set_percent(100)
                dialog.advance()
                self._shutdown_stage = 3
        if self._shutdown_stage == 3:
            if self._shutdown_reader_future is None:
                self._shutdown_reader_future = self._play_executor.submit(
                    self.close_processing_reader)
                return
            if not self._shutdown_reader_future.done():
                return
            self._shutdown_reader_future.result()
            self._play_executor.shutdown(wait=False)
            self._sync_executor.shutdown(wait=False)
            self._review_save_executor.shutdown(wait=False)
            if self._feedback_load_executor is not None:
                self._feedback_load_executor.shutdown(wait=False)
            if self._shutdown_save_executor is not None:
                self._shutdown_save_executor.shutdown(wait=False)
            self._feedback_store.close()
            dialog.set_percent(100)
            dialog.advance()
            self._shutdown_timer.Stop()
            dialog.Destroy()
            self._shutdown_dialog = None
            self.Destroy()

    def _shutdown_save_progress(self, percent: int) -> None:
        if self._shutdown_dialog is not None and self._shutdown_stage == 2:
            self._shutdown_dialog.set_percent(percent)

    def on_box_selected(self, event: BoxSelectedEvent) -> None:
        selected_box = event.box
        self.__tag_panel.selected_box = selected_box
        getLog().debug(f'Selected box {selected_box.coords} with tags {selected_box.tags}')
        # self.__tag_panel.set_selected(event.box)

    @staticmethod
    def find_object_in_next_frame(
        prev_frame: np.ndarray,
        next_frame: np.ndarray,
        bbox: BoxData
    ) -> BoxData | None:
        x: int
        y: int
        w: int
        h: int
        x, y, w, h = bbox.coords
        template: np.ndarray = prev_frame[y:y + h, x:x + w]

        orb: cv2.ORB = cv2.ORB_create()
        kp1: list[cv2.KeyPoint]
        des1: np.ndarray | None
        kp1, des1 = orb.detectAndCompute(template, None)
        kp2: list[cv2.KeyPoint]
        des2: np.ndarray | None
        kp2, des2 = orb.detectAndCompute(next_frame, None)

        if des1 is None or des2 is None or len(kp1) == 0 or len(kp2) == 0:
            return None

        bf: cv2.BFMatcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches: list[cv2.DMatch] = bf.match(des1, des2)
        matches = sorted(matches, key=lambda m: m.distance)

        # src_pts: np.ndarray = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        # dst_pts: np.ndarray = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        src_pts = np.float32([np.array(kp1[m.queryIdx].pt) for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([np.array(kp2[m.trainIdx].pt) - np.array([x, y]) for m in matches]).reshape(-1, 1, 2)

        if src_pts.shape[0] >= 3:
            M: np.ndarray | None
            mask: np.ndarray | None
            M, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts)
            if M is not None:
                corners: np.ndarray = np.float32([
                    [x, y],
                    [x + w, y],
                    [x + w, y + h],
                    [x, y + h]
                ]).reshape(-1, 1, 2)
                new_corners: np.ndarray = cv2.transform(corners, M)
                new_bbox: tuple[int, int, int, int] = cv2.boundingRect(new_corners)

                new_data = BoxData(
                    coords=(new_bbox[0], new_bbox[1], new_bbox[2], new_bbox[3]),
                    tags=copy(bbox.tags),
                    source='automatic'
                )
                new_data.prev_box_id = bbox.id
                return new_data
        return None

    def get_selected_box(self) -> BoxData | None:
        """Get the currently selected box from the tag panel."""
        return self.__image_panel.selected_box if self.__image_panel else None

    def __on_remove_selected(self, event: wx.CommandEvent) -> None:
        self.remove_boxes(self.__image_panel.selected_boxes)

    def remove_box(self, box: BoxData) -> None:
        """Delete this annotation without making a pin/non-pin judgement."""
        self.remove_boxes([box])

    def remove_boxes(self, boxes: list[BoxData]) -> None:
        self._remove_or_reject_boxes(boxes, not_pin=False)

    def mark_not_pin(self, box: BoxData) -> None:
        """Keep a hidden negative example and suppress matching automatic areas."""
        self.mark_not_pin_boxes([box])

    def mark_not_pin_boxes(self, boxes: list[BoxData]) -> None:
        self._remove_or_reject_boxes(boxes, not_pin=True)

    def _remove_or_reject_boxes(self, selected: list[BoxData], not_pin: bool) -> None:
        self.pause_playback()
        boxes = self.__current_boxes
        targets: list[BoxData] = []
        for box in selected:
            if (any(existing is box for existing in boxes) and
                    box.review_state != 'rejected' and
                    (not not_pin or box.source == 'automatic') and
                    all(existing is not box for existing in targets)):
                targets.append(box)
        if not targets:
            return
        self._begin_history()
        before = self._history_pending[1]
        for box in targets:
            if not_pin:
                same_track = [existing for frame in self.__frame_boxes.values()
                              for existing in frame if existing is not box
                              and existing.track_id == box.track_id]
                if any(existing.source == 'user' for existing in same_track):
                    # A manual correction wins over an automatic track's rejection.
                    box.track_id = uuid4().hex
                box.review_state = 'rejected'
                box.identity_confirmed = False
                box.identity_inferred = False
                box.feedback_id = box.feedback_id or uuid4().hex
                box.feedback_label = 'negative'
            else:
                boxes.remove(box)
        if not_pin:
            propagate_rejections(self.__frame_boxes)
        propagate_confirmations(self.__frame_boxes)
        remaining = marked_pin_count(boxes)
        old_flags = self._frame_presence.get(self._current_index, FramePresence())
        new_flags = FramePresence(old_flags.pin_asserted and remaining >= 1,
                                  old_flags.set_asserted and remaining >= 2)
        if new_flags.pin_asserted or new_flags.set_asserted:
            self._frame_presence[self._current_index] = new_flags
        else:
            self._frame_presence.pop(self._current_index, None)
        self.__detected_frames.add(self._current_index)
        self.__image_panel.boxes = boxes
        self.__tag_panel.boxes = boxes
        self.__image_panel.Refresh()
        self._finish_history()
        if not_pin:
            history = self._undo_history.get(self._current_index, [])
            if history:
                self._pending_rejection_history = (
                    self._current_index, self._annotation_revision, before, history[-1])
            self._schedule_reconciliation(self._current_index)

    def confirm_box(self, box: BoxData) -> None:
        """Confirm pin presence without approving a suggested name."""
        self.confirm_boxes([box])

    def classifier_review_restore_pin(self, box: BoxData) -> bool:
        """Correct one negative crop into a positive pin-presence review."""
        if (box.source != 'automatic' or box.review_state != 'rejected' or
                all(existing is not box for existing in self.__current_boxes)):
            return False
        self.pause_playback()
        self._begin_history()
        box.track_id = uuid4().hex
        box.prev_box_id = None
        box.review_state = 'confirmed'
        box.pin_id = None
        box.tags = []
        box.match_confidence = None
        box.identity_confirmed = False
        box.identity_inferred = False
        box.feedback_id = box.feedback_id or uuid4().hex
        box.feedback_label = 'positive'
        self.__image_panel.boxes = self.__current_boxes
        self.__tag_panel.boxes = self.__current_boxes
        self.__image_panel.Refresh()
        self._finish_history()
        self.__matched_frames.discard(self._current_index)
        self._schedule_catalog_match(
            self._current_index, self.get_frame(self._current_index, 0), None,
            target_box=box)
        return True

    def confirm_boxes(self, selected: list[BoxData]) -> None:
        """Confirm multiple areas as pins in one undoable review."""
        self.pause_playback()
        self.__tag_panel.commit_pending_labels()
        targets: list[BoxData] = []
        for box in selected:
            if (any(existing is box for existing in self.__current_boxes) and
                    box.review_state not in ('confirmed', 'rejected') and
                    all(existing is not box for existing in targets)):
                targets.append(box)
        if not targets:
            return
        self._begin_history()
        for box in targets:
            box.review_state = 'confirmed'
            box.identity_confirmed = False
            box.identity_inferred = False
            box.feedback_id = box.feedback_id or uuid4().hex
            box.feedback_label = 'positive'
        self.__tag_panel.boxes = self.__current_boxes
        self.__image_panel.Refresh()
        self._finish_history()
        if any(self.__frame_boxes.get(neighbour)
               for neighbour in (self._current_index - 1, self._current_index + 1)):
            self._schedule_reconciliation(self._current_index)
        if any(needs_catalog_match(box) for box in targets):
            self.__matched_frames.discard(self._current_index)
            self._schedule_catalog_match(
                self._current_index, self.get_frame(self._current_index, 0), None)

    def confirm_pin_identity(self, box: BoxData) -> None:
        """Confirm the current specific ID or user-entered pin name."""
        self._confirm_identities([box])

    def confirm_all_suggestions(self, _event: wx.CommandEvent | None = None) -> None:
        """Directly verify catalog suggestions visible on the current frame."""
        self.__tag_panel.commit_pending_labels()
        suggestions = [
            box for box in self.__current_boxes
            if box.review_state != 'rejected'
            and isinstance(box.pin_id, str)
            and bool(box.pin_id.strip())
            and not is_placeholder_tag(box.pin_id)
            and (box.review_state != 'confirmed' or not box.identity_confirmed
                 or box.identity_inferred)
        ]
        count = self._confirm_identities(suggestions)
        noun = 'identity' if count == 1 else 'identities'
        self.SetStatusText(
            f'Confirmed {count} suggested pin {noun} on frame {self._current_index}' if count else
            'No unconfirmed pin suggestions on this frame', 1)

    def _confirm_identities(self, selected: list[BoxData]) -> int:
        """Confirm identities in one owner transaction and one undo entry."""
        self.pause_playback()
        self.__tag_panel.commit_pending_labels()
        targets: list[BoxData] = []
        for box in selected:
            if (any(existing is box for existing in self.__current_boxes)
                    and box.review_state != 'rejected'
                    and has_specific_identity(box)
                    and (box.review_state != 'confirmed' or not box.identity_confirmed
                         or box.identity_inferred)
                    and all(existing is not box for existing in targets)):
                targets.append(box)
        if not targets:
            return 0
        self._begin_history()
        for box in targets:
            box.review_state = 'confirmed'
            box.identity_confirmed = True
            box.identity_inferred = False
            box.feedback_id = box.feedback_id or uuid4().hex
            box.feedback_label = 'positive'
        self.__tag_panel.boxes = self.__current_boxes
        self.__image_panel.Refresh()
        self._finish_history()
        if any(self.__frame_boxes.get(neighbour)
               for neighbour in (self._current_index - 1, self._current_index + 1)):
            self._schedule_reconciliation(self._current_index)
        return len(targets)

    def mark_wrong_identity(self, box: BoxData) -> None:
        """Reject only the proposed identity, keeping the pin-presence decision."""
        if not has_specific_identity(box):
            return
        self._reject_pin_identities(box, [(box.pin_id, tuple(box.tags or []))],
                                    retry_catalog=True)

    def classifier_review_reject_identity(self, box: BoxData) -> bool:
        """Review this crop as a pin while recording its suggested ID as wrong."""
        if not has_specific_identity(box):
            return False
        return self._reject_pin_identities(
            box, [(box.pin_id, tuple(box.tags or []))], retry_catalog=True,
            confirm_presence=True)

    def rank_pin_replacements(self, box: BoxData, limit: int = 10) -> list[PinMatch]:
        """Rank catalog choices for the visible rectangle without changing it."""
        if (all(existing is not box for existing in self.__current_boxes) or
                box.review_state == 'rejected'):
            return []
        rank = getattr(self._pin_matcher, 'rank_matches', None)
        if rank is None:
            return []
        frame = (self._display_source_frame if self._display_source_frame is not None
                 else self.get_frame(self._current_index, 0))
        if frame is None:
            return []
        x, y, width, height = box.coords
        if (x < 0 or y < 0 or width <= 0 or height <= 0 or
                x + width > frame.shape[1] or y + height > frame.shape[0]):
            return []
        choices = rank(
            frame[y:y + height, x:x + width], limit=limit,
            excluded_pin_ids=frozenset(item.pin_id for item in box.identity_rejections
                                       if item.pin_id),
            excluded_names=frozenset(tag.casefold() for item in box.identity_rejections
                                     if item.pin_id is None for tag in item.tags))
        # A geometric or reviewed-crop suggestion can fall outside the colour
        # top ten. Include it so "none of these" actually excludes the visible
        # suggestion before the selected rectangle is scanned again.
        if (choices and box.pin_id and box.tags and
                not any(match.pin_id == box.pin_id for match in choices) and
                not rejects_identity(box, box.pin_id, box.tags)):
            current = PinMatch(box.pin_id, box.tags[0], 0,
                               box.match_confidence)
            return [current, *choices[:max(0, limit - 1)]]
        return choices

    def replace_with_pin(self, box: BoxData, match: PinMatch) -> bool:
        """Treat the user's menu choice as a direct identity confirmation."""
        self.pause_playback()
        if (all(existing is not box for existing in self.__current_boxes) or
                box.review_state == 'rejected' or
                rejects_identity(box, match.pin_id, [match.name])):
            return False
        self._begin_history()
        if (has_specific_identity(box) and
                (box.pin_id != match.pin_id or
                 (box.pin_id is None and box.tags != [match.name])) and
                not rejects_identity(box, box.pin_id, box.tags)):
            old_review = IdentityRejection(
                uuid4().hex, box.pin_id,
                tuple(tag for tag in box.tags or [] if tag.strip()),
                box.coords, self._current_index)
            self._apply_identity_rejections(box, [old_review])
        box.pin_id = match.pin_id
        box.tags = [match.name]
        box.match_confidence = match.confidence
        box.review_state = 'confirmed'
        box.identity_confirmed = True
        box.identity_inferred = False
        box.feedback_id = box.feedback_id or uuid4().hex
        box.feedback_label = 'positive'
        self.__tag_panel.boxes = self.__current_boxes
        self.__image_panel.Refresh()
        self._finish_history()
        if any(self.__frame_boxes.get(neighbour)
               for neighbour in (self._current_index - 1, self._current_index + 1)):
            self._schedule_reconciliation(self._current_index)
        return True

    def reject_pin_replacements(self, box: BoxData,
                                matches: list[PinMatch]) -> bool:
        """Reject one displayed page, then suggest the next identity for this area."""
        return self._reject_pin_identities(
            box, [(match.pin_id, (match.name,)) for match in matches],
            retry_catalog=True)

    def _reject_pin_identities(self, box: BoxData,
                               candidates: list[tuple[str | None, tuple[str, ...]]],
                               retry_catalog: bool,
                               confirm_presence: bool = False) -> bool:
        self.pause_playback()
        if (all(existing is not box for existing in self.__current_boxes) or
                box.review_state == 'rejected'):
            return False
        old_ids = {item.pin_id for item in box.identity_rejections if item.pin_id}
        old_names = {tag.casefold() for item in box.identity_rejections
                     if item.pin_id is None for tag in item.tags}
        reviews: list[IdentityRejection] = []
        for pin_id, tags in candidates:
            clean_tags = tuple(tag for tag in tags if tag.strip())
            if ((pin_id and pin_id in old_ids) or
                    (pin_id is None and any(tag.casefold() in old_names
                                            for tag in clean_tags))):
                continue
            reviews.append(IdentityRejection(uuid4().hex, pin_id, clean_tags,
                                             box.coords, self._current_index))
            if pin_id:
                old_ids.add(pin_id)
            else:
                old_names.update(tag.casefold() for tag in clean_tags)
        if not reviews:
            return False
        stored_before = self._feedback_store.identity_rejection_count(self._feedback_source)
        self._begin_history()
        if confirm_presence:
            box.review_state = 'confirmed'
            box.feedback_id = box.feedback_id or uuid4().hex
            box.feedback_label = 'positive'
            box.identity_confirmed = False
            box.identity_inferred = False
        self._apply_identity_rejections(box, reviews)
        self.__matched_frames.discard(self._current_index)
        self.__image_panel.Refresh()
        self.__tag_panel.boxes = self.__current_boxes
        self._finish_history()
        stored_total = self._feedback_store.identity_rejection_count(self._feedback_source)
        saved = stored_total - stored_before
        noun = 'example' if saved == 1 else 'examples'
        message = f'{saved} wrong-ID {noun} saved ({stored_total} total)'
        if saved < len(reviews):
            message += f'; {len(reviews) - saved} could not be saved'
        self.SetStatusText(message, 1)
        if self._training_future is None:
            self.__tag_panel.set_review_status(message)
        if retry_catalog:
            frame = (self._display_source_frame if self._display_source_frame is not None
                     else self.get_frame(self._current_index, 0))
            if self._schedule_catalog_match(self._current_index, frame, None,
                                            target_box=box, best_effort=True):
                self.SetStatusText(message + '; retrying identity for selected area', 1)
            elif (not getattr(self._pin_matcher, 'has_references', True) and
                  not self._feedback_model.identity_sample_count):
                self.SetStatusText(
                    message + '; no catalog images or reviewed pin identities are loaded', 1)
            else:
                self.SetStatusText(message + '; could not retry identity for selected area', 1)
        return True

    def _apply_identity_rejections(self, box: BoxData,
                                   reviews: list[IdentityRejection]) -> None:
        """Share direct wrong-ID evidence with the track inside a history edit."""
        # The same tracked object should not be assigned the rejected name in
        # another frame. Keep the review on its source box for one training crop.
        rejected_ids = {review.pin_id for review in reviews if review.pin_id}
        rejected_names = {tag.casefold() for review in reviews for tag in review.tags}
        for boxes in self.__frame_boxes.values():
            for member in boxes:
                if member.track_id != box.track_id or member.review_state == 'rejected':
                    continue
                if (member is not box and member.review_state == 'confirmed' and
                        member.identity_confirmed and not member.identity_inferred):
                    continue
                member.identity_rejections = (*member.identity_rejections, *reviews)
                if (member.pin_id in rejected_ids or
                        (member.pin_id is None and any(
                            tag.casefold() in rejected_names for tag in member.tags or []))):
                    member.pin_id = None
                    member.tags = []
                    member.match_confidence = None
                    member.identity_confirmed = False
                    member.identity_inferred = False

    def pause_playback(self) -> None:
        self._playing = False
        self._play_timer.Stop()
        self._cancel_reconciliation()
        self._cancel_catalog_matches()
        self._cancel_attempt_detection()
        self._play_generation += 1
        if self._pending_frame_job is not None:
            self._pending_frame_job.cancel()
            self._pending_frame_job = None
        if hasattr(self, '_ScrubberFrame__button_panel'):
            self.__button_panel.play_btn.SetLabel('Play')
        if hasattr(self, '_ScrubberFrame__tag_panel'):
            self.__tag_panel.Enable()
            if self._tags_dirty:
                self.__tag_panel.boxes = self.__current_boxes
                self._tags_dirty = False

    def on_play(self, _event: wx.CommandEvent) -> None:
        if self._playing:
            self.pause_playback()
            return
        if self._pending_frame_job is not None:
            self.pause_playback()
        self._cancel_reconciliation()
        self._cancel_catalog_matches()
        self._cancel_attempt_detection()
        if self._current_index >= self.__num_frames - 1 and self._current_index in self.__detected_frames:
            return
        self.__button_panel.play_btn.SetLabel('Pause')
        self.__tag_panel.Disable()
        self._playing = True
        self._play_timer.StartOnce(1)

    def _on_play_timer(self, _event: wx.TimerEvent) -> None:
        if not self._playing or self._pending_frame_job is not None:
            return
        current = self._current_index
        target = current if current not in self.__detected_frames else current + 1
        if target >= self.__num_frames:
            self.pause_playback()
            return
        self._submit_frame_job(target, requires_play=True)

    def on_next_async(self, _event: wx.CommandEvent) -> None:
        if self._current_index < self.__num_frames - 1:
            self.goto_frame(self._current_index + 1)

    def _needs_manual_detection(self, index: int) -> bool:
        """Run the full detector once when playback only made a primary pass."""
        return (index not in self.__detected_frames or
                (index in self._primary_only_frames and
                 not self.frame_has_boxes(index)))

    def on_detect_async(self, _event: wx.CommandEvent) -> None:
        self.pause_playback()
        self._submit_frame_job(self._current_index, requires_play=False,
                               catalog_best_effort=True,
                               diagnostic_run_id=uuid4().hex[:8])

    def _submit_frame_job(self, target: int, requires_play: bool,
                          reference_index: int | None = None,
                          reference_frame: np.ndarray | None = None,
                          catalog_best_effort: bool = False,
                          diagnostic_run_id: str | None = None) -> None:
        if self._pending_frame_job is not None:
            return
        started = perf_counter()
        current = self._current_index
        previous_index = (reference_index if reference_index is not None else
                          current if target != current else None)
        if previous_index is None and self._frame_presence.get(target, FramePresence()).minimum_pins:
            previous_index = next((neighbor for neighbor in (target - 1, target + 1)
                                   if 0 <= neighbor < self.__num_frames and
                                   marked_pin_count(self.__frame_boxes.get(neighbor, []))), None)
        job = FrameJob(
            index=target,
            previous_index=previous_index,
            previous_boxes=deepcopy(self.__frame_boxes.get(previous_index, []))
            if previous_index is not None else [],
            existing_boxes=deepcopy(self.__get_frame_boxes(target)),
            rejected_next=deepcopy([box for box in self.__frame_boxes.get(target + 1, [])
                                    if box.review_state == 'rejected']) if previous_index is None else [],
            displayed_frame=self._display_source_frame if target == current else None,
            previous_frame=(reference_frame if reference_index is not None else
                            self._display_source_frame if previous_index == current else None),
            feedback_model=self._feedback_model.snapshot_for_processing(),
            matcher=self._pin_matcher,
            expected_pins=self._frame_presence.get(target, FramePresence()).minimum_pins,
            allow_empty_fallback=not requires_play and target == current,
            allow_motion_fallback=previous_index is not None,
            catalog_best_effort=catalog_best_effort,
            diagnostic_run_id=diagnostic_run_id,
            low_resolution=self._detection_resolution == 'low',
        )
        generation = self._play_generation
        revision = self._annotation_revision
        future = self._play_executor.submit(process_frame, job, self.read_frame_for_processing)
        self._pending_frame_job = future
        future.add_done_callback(
            lambda completed: wx.CallAfter(self._complete_play_job, completed,
                                           generation, revision, requires_play,
                                           started))
        getLog().info('frame=%d ui submit=%.1fms', target, (perf_counter()-started)*1000)

    def _complete_play_job(self, future: Future[FrameResult], generation: int,
                           revision: int, requires_play: bool,
                           submitted_at: float) -> None:
        if self._closing or future is not self._pending_frame_job:
            return
        if (generation != self._play_generation or
                revision != self._annotation_revision or
                (requires_play and not self._playing)):
            self._pending_frame_job = None
            return
        self._pending_frame_job = None
        try:
            result = future.result()
        except Exception:
            getLog().exception('Background frame processing failed')
            self.pause_playback()
            return
        started = perf_counter()
        target = result.index
        existing = self.__get_frame_boxes(target)
        if len(existing) != result.existing_count:
            if requires_play:
                self.pause_playback()
            return
        protected_tracks: set[str] | None = None
        for original, analyzed in zip(existing, result.boxes[:result.existing_count]):
            if (original.source == 'automatic' and
                    original.review_state in ('unconfirmed', 'inherited') and
                    analyzed.review_state != 'rejected' and
                    not original.bounds_direct and original.feedback_id is None and
                    (original.coords != analyzed.coords or
                     original.bounds_adjustment != analyzed.bounds_adjustment)):
                original.coords = analyzed.coords
                original.bounds_adjustment = analyzed.bounds_adjustment
            if (original.source == 'automatic' and
                    original.review_state in ('unconfirmed', 'inherited') and
                    not original.identity_confirmed):
                if analyzed.review_state == 'rejected':
                    if protected_tracks is None:
                        protected_tracks = {box.track_id for boxes in self.__frame_boxes.values()
                                            for box in boxes if box.source == 'user' or
                                            box.review_state == 'confirmed'}
                    if original.track_id in protected_tracks:
                        continue
                original.track_id = analyzed.track_id
                original.review_state = analyzed.review_state
                original.pin_id = analyzed.pin_id
                original.tags = list(analyzed.tags or [])
                original.match_confidence = analyzed.match_confidence
                original.identity_confirmed = analyzed.identity_confirmed
                original.identity_inferred = analyzed.identity_inferred
                original.identity_rejections = analyzed.identity_rejections
            elif (needs_catalog_match(original, best_effort=result.catalog_best_effort) and
                  analyzed.pin_id is not None):
                if analyzed.identity_confirmed:
                    original.track_id = analyzed.track_id
                    original.identity_confirmed = True
                    original.identity_inferred = original.review_state == 'confirmed'
                original.pin_id = analyzed.pin_id
                original.tags = list(analyzed.tags or [])
                original.match_confidence = analyzed.match_confidence
            elif (original.source == 'user' and
                  original.review_state in ('unconfirmed', 'inherited')
                  and analyzed.identity_confirmed and
                  (same_specific_name(original, analyzed) or
                   not has_specific_identity(original))):
                original.track_id = analyzed.track_id
                original.review_state = analyzed.review_state
                original.pin_id = analyzed.pin_id
                if not has_specific_identity(original):
                    original.tags = list(analyzed.tags or [])
                original.match_confidence = analyzed.match_confidence
                original.identity_confirmed = True
        # The worker already merged proposals against its annotation snapshot.
        # Reapplying an overlap-only merge here would discard a visually tracked
        # box beside an unrelated saved automatic candidate.
        self.__frame_boxes[target] = existing + result.boxes[result.existing_count:]
        self._invalidate_reconciliation()
        if not requires_play:
            propagate_rejections(self.__frame_boxes)
            propagate_confirmations(self.__frame_boxes)
        self._sync_automatic_consolidation(
            frame_index=target if requires_play else None)
        self.__matched_frames.add(target)
        self.__detected_frames.add(target)
        if requires_play and not self.frame_has_boxes(target):
            self._primary_only_frames.add(target)
        else:
            self._primary_only_frames.discard(target)
        self._current_index = target
        committed = perf_counter()
        self.display_image(result.frame, reconcile=not requires_play)
        if result.diagnostic_run_id is not None:
            self._log_detection_display(result)
        visible = sum(box.review_state != 'rejected' for box in self.__frame_boxes[target])
        if visible:
            summary = (f'Frame {target}: {visible} candidates '
                       f'(primary {result.primary_count}, fallback {result.relaxed_count})')
        else:
            summary = (f'Frame {target}: no candidates found; draw a box or '
                       'check Contains pin')
        self.SetStatusText(summary, 1)
        getLog().info('frame=%d ui commit=%.1fms render_queue=%.1fms worker_total=%.1fms',
                      target, (committed-started)*1000,
                      (perf_counter()-committed)*1000, result.total_ms)
        if target >= self.__num_frames - 1:
            self.pause_playback()
        elif requires_play and self._playing:
            elapsed_ms = (perf_counter() - submitted_at) * 1000
            self._play_timer.StartOnce(self.playback_delay_ms(
                self.playback_frame_interval_ms(), elapsed_ms))

    def playback_frame_interval_ms(self) -> int:
        return 50

    @staticmethod
    def playback_delay_ms(frame_interval_ms: int, elapsed_ms: float) -> int:
        """Avoid an extra idle gap after a frame already missed its deadline."""
        return max(1, round(frame_interval_ms - elapsed_ms))

    def _log_detection_display(self, result: FrameResult) -> None:
        """Record both coordinate spaces for an explicit Detect Pins run."""
        panel = self.__image_panel
        panel_size = panel.GetSize()
        run = result.diagnostic_run_id
        if not hasattr(panel, 'img_size') or not hasattr(panel, 'bmp_size'):
            getLog().warning('PINIDENT_DETECTION display unavailable run=%s frame=%d',
                             run, result.index)
            return
        header = {
            'run': run,
            'frame': result.index,
            'source_size': [int(result.frame.shape[1]), int(result.frame.shape[0])],
            'rotation': self._rotation_angle,
            'display_size': list(panel.img_size),
            'bitmap_size': list(panel.bmp_size),
            'panel_size': [panel_size.GetWidth(), panel_size.GetHeight()],
            'bitmap_offset': list(panel.get_image_offset()),
            'xywh': 'x,y,width,height',
            'box_count': len(self.__frame_boxes[result.index]),
        }
        getLog().info('PINIDENT_DETECTION display %s',
                      json.dumps(header, separators=(',', ':')))
        source_width, source_height = result.frame.shape[1], result.frame.shape[0]
        for index, box in enumerate(self.__frame_boxes[result.index]):
            x, y, width, height = box.coords
            record = {
                'run': run,
                'index': index,
                'origin': 'existing' if index < result.existing_count else 'new',
                'source_xywh': list(box.coords),
                'panel_xywh': list(panel._box_panel_rect(box)),
                'in_source_bounds': (x >= 0 and y >= 0 and width > 0 and height > 0
                                     and x + width <= source_width
                                     and y + height <= source_height),
                'source': box.source,
                'review_state': box.review_state,
                'tags': list(box.tags or []),
                'pin_id': box.pin_id,
                'match_confidence': box.match_confidence,
                'identity_confirmed': box.identity_confirmed,
            }
            getLog().info('PINIDENT_DETECTION box %s',
                          json.dumps(record, separators=(',', ':')))
        getLog().info('PINIDENT_DETECTION end run=%s', run)

    def __on_process(self, event: wx.CommandEvent) -> None:
        """Find possible pins without overwriting existing annotations."""
        if not self._playback_step:
            self.pause_playback()
        frame_img = self.get_frame(self._current_index, 0)
        if frame_img is None:
            getLog().warning("No image data for current frame.")
            return
        existing = self.__current_boxes
        proposals = detect_pin_boxes(frame_img, feedback_model=self._feedback_model)
        rejected_next = [box for box in self.__frame_boxes.get(self._current_index + 1, [])
                         if box.review_state == 'rejected']
        next_frame = self.get_frame(self._current_index + 1, 0) if rejected_next else None
        if next_frame is not None:
            proposed_boxes = associate_frame_boxes(next_frame, frame_img,
                                                   rejected_next, proposals)
        else:
            proposed_boxes = add_new_detections([], proposals)
        proposed_boxes = filter_new_candidates_by_pin_size(frame_img, existing,
                                                            proposed_boxes)
        self.__frame_boxes[self._current_index] = add_unique_boxes(existing, proposed_boxes)
        self._invalidate_reconciliation()
        self._reconcile_frame(self._current_index)
        self._reconcile_done.add(self._current_index)
        detected = identify_candidates(frame_img, self.__current_boxes,
                                       self._pin_matcher, best_effort=True)
        if (marked_pin_count(self.__current_boxes) <
                self._frame_presence.get(self._current_index, FramePresence()).minimum_pins
                or not any(box.review_state != 'rejected'
                           for box in self.__current_boxes)):
            extra = detect_pin_boxes_relaxed(frame_img, self._feedback_model)
            proposed_extra = filter_new_candidates_by_pin_size(
                frame_img, self.__current_boxes, add_new_detections([], extra))
            self.__frame_boxes[self._current_index] = add_unique_boxes(
                self.__current_boxes, proposed_extra)
            identify_candidates(frame_img, self.__current_boxes,
                                self._pin_matcher, best_effort=True)
        propagate_rejections(self.__frame_boxes)
        self._sync_automatic_consolidation()
        self.__matched_frames.add(self._current_index)
        self.__detected_frames.add(self._current_index)
        self.display_image()
        getLog().info(f"Detected {len(detected) - len(existing)} new candidates in frame {self._current_index}")


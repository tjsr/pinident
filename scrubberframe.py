import json
import os
from collections import defaultdict, deque
from copy import copy, deepcopy
from concurrent.futures import Future, ThreadPoolExecutor
from threading import Event
from time import perf_counter
from typing import Callable, List, Dict
from uuid import NAMESPACE_URL, uuid4, uuid5

import cv2
import numpy as np
import wx

from boxdata import (BoxData, Coordinate, IdentityRejection,
                     has_specific_identity, rejects_identity, same_specific_name)
from annotation_history import BoxSnapshot, HistoryEntry, make_entry, snapshot
from detection_feedback import (FeedbackModel, FeedbackStore,
                                make_feedback_example)
from frame_processing import FrameJob, FrameResult, process_frame
from frame_presence import FramePresence, checkbox_values, marked_pin_count
from controlspanel import ControlsPanel
from events.BoxSelectedEvent import BoxSelectedEvent
from events.events import EVT_BOX_SELECTED
from controls.imagepanel import ImagePanel
from logutil import getLog
from controls.markerpanel import MarkerPanel  # Adjust import as needed
from controls.tagpanel import TagPanel
from pin_catalog import PinCatalog, PinMatch, PinMatcher
from session_training import (TrainingDelta, reviewed_observations,
                              plan_training_delta, build_training_examples)
from pin_detection import (add_new_detections, add_unique_boxes, associate_frame_boxes,
                           can_link_saved_target,
                           detect_pin_boxes, detect_pin_boxes_relaxed,
                           identify_candidates, inherit_confirmed_catalog_matches,
                           needs_catalog_match,
                           match_existing_candidates,
                           match_rejected_candidates)

def remove_empty(tags: List[str]) -> List[str]:
    """Remove empty tags from the list."""
    return [tag for tag in tags if tag.strip()]

def save_boxes_to_stream(stream, frame_boxes: dict[int, list[BoxData]],
                         presence: dict[int, FramePresence] | None = None) -> None:
    # frame_boxes: {frame_number: [BoxData, ...]}
    serializable = {
        frame: [
            {"coords": box.coords, "tags": remove_empty(box.tags), "source": box.source,
             "prev_box_id": box.prev_box_id, "track_id": box.track_id,
             "review_state": box.review_state,
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
        for frame, boxes in frame_boxes.items()
    }
    if presence:
        assertions = {str(frame): {'contains_pin': flags.pin_asserted,
                                    'contains_set': flags.set_asserted}
                      for frame, flags in presence.items()
                      if flags.pin_asserted or flags.set_asserted}
        if assertions:
            serializable['_frame_presence'] = assertions
    json.dump(serializable, stream)

def save_boxes_to_file(filename: str, frame_boxes: dict[int, list[BoxData]],
                       presence: dict[int, FramePresence] | None = None) -> None:
    with open(filename, "w", encoding="utf-8") as f:
        save_boxes_to_stream(f, frame_boxes, presence)
        f.close()

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
                                  identity_rejections=box.identity_rejections)
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
                    has_specific_identity(box))]
        if sources:
            source_frame, target_frame = frame_at(reference_index), frame_at(index)
            if source_frame is not None and target_frame is not None:
                proposed = associate_frame_boxes(source_frame, target_frame, sources, [])
                frame_boxes[index] = add_unique_boxes(frame_boxes.get(index, []), proposed)
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
            except Exception as e:
                getLog().error(f"Error loading box data: {e}")
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
        self._undo_history: dict[int, list[HistoryEntry]] = {}
        self._redo_history: dict[int, list[HistoryEntry]] = {}
        self._history_pending: tuple[int, dict[int, BoxSnapshot], int,
                                     dict[int, FramePresence]] | None = None
        self._pending_rejection_history: tuple[int, int, dict[int, BoxSnapshot],
                                               HistoryEntry] | None = None
        self._feedback_source = feedback_source or f'session:{uuid4().hex}'
        self._feedback_store = FeedbackStore(feedback_path)
        self._feedback_model = FeedbackModel(self._feedback_store)
        self._training_future: Future | None = None
        self._playback_step = False
        self._playing = False
        self._play_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='pin-detection')
        self._pending_frame_job: Future[FrameResult] | None = None
        self._pending_reconcile: Future[bool] | None = None
        self._pending_reconcile_index: int | None = None
        self._reconcile_cancel = Event()
        self._reconcile_done: set[int] = set()
        self._reconciled_pairs: set[tuple[int, int]] = set()
        self._annotation_revision = 0
        self._play_generation = 0
        self._closing = False
        self._display_source_frame: np.ndarray | None = None
        self._tags_dirty = False
        self._play_timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self._on_play_timer, self._play_timer)
        self._pin_catalog = PinCatalog(catalog_path) if catalog_path else PinCatalog()
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
        self.__image_panel.on_confirm_identity = self.confirm_pin_identity
        self.__image_panel.on_wrong_identity = self.mark_wrong_identity
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
            on_feedback_info=self.show_feedback_info,
            catalog_options=self._pin_catalog.list_pins() if load_catalog else [],
            on_add_training_data=self.add_to_training_data,
        )
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
        self.__button_panel.bind_buttons(self.on_prev, self.on_next_async, self.on_next_empty, self.on_rotate_ccw, self.on_rotate_cw, self.__on_remove_selected, self.on_detect_async, self.file_select, self.on_play)

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

        self.Bind(wx.EVT_CHAR_HOOK, self.on_key_down)
        self.Bind(wx.EVT_SHOW, self.on_show)

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
        self._schedule_catalog_match(index, frame, None)

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

    def _invalidate_reconciliation(self) -> None:
        self._cancel_reconciliation()
        self._reconcile_done.clear()
        self._reconciled_pairs.clear()
        self._annotation_revision += 1
        self._pending_rejection_history = None

    def _schedule_reconciliation(self, index: int,
                                 reference_index: int | None = None) -> None:
        if self._closing:
            return
        has_reference = (reference_index is not None and abs(reference_index - index) == 1
                         and any(box.review_state != 'rejected' and
                                 (box.review_state in ('confirmed', 'inherited') or
                                  has_specific_identity(box))
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
        snapshot = deepcopy(self.__frame_boxes)
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
                original.track_id = analyzed.track_id
                original.pin_id = analyzed.pin_id
                original.tags = list(analyzed.tags or [])
                original.match_confidence = analyzed.match_confidence
                original.identity_confirmed = analyzed.identity_confirmed
                original.identity_inferred = analyzed.identity_inferred
                original.review_state = analyzed.review_state
                original.identity_rejections = analyzed.identity_rejections
            self.__frame_boxes[frame_index].extend(updated[count:])
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

    def display_image(self, source_frame: np.ndarray | None = None,
                      reconcile: bool = True,
                      reference_index: int | None = None):
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
        if self._pending_reconcile_index != self._current_index:
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
                                reference_index: int | None) -> bool:
        if index in self._pending_catalog:
            return True
        if self._closing or index in self.__matched_frames:
            return False
        if not getattr(self._pin_matcher, 'has_references', True):
            self.__matched_frames.add(index)
            return False
        candidates = [(box, deepcopy(box), deepcopy(box))
                      for box in self.__frame_boxes.get(index, [])
                      if needs_catalog_match(box)]
        if not candidates:
            self.__matched_frames.add(index)
            return False
        matcher = self._pin_matcher
        revision = self._annotation_revision
        started = perf_counter()
        def scan() -> list[BoxData] | None:
            image = frame.copy() if frame is not None else self.read_frame_for_processing(index)
            if image is None:
                return None
            result = identify_candidates(image, [copy for _, _, copy in candidates], matcher)
            getLog().info('frame=%d catalog=%.1fms candidates=%d thread=worker',
                          index, (perf_counter() - started) * 1000, len(result))
            return result
        future = self._play_executor.submit(scan)
        self._pending_catalog[index] = future
        future.add_done_callback(lambda completed: wx.CallAfter(
            self._complete_catalog_match, completed, index, revision, matcher,
            candidates, reference_index))
        return True

    def _complete_catalog_match(self, future: Future[list[BoxData] | None], index: int,
                                revision: int, matcher: PinMatcher,
                                candidates: list[tuple[BoxData, BoxData, BoxData]],
                                reference_index: int | None) -> None:
        if future is not self._pending_catalog.get(index):
            return
        del self._pending_catalog[index]
        if (self._closing or revision != self._annotation_revision or
                matcher is not self._pin_matcher or index in self.__matched_frames):
            getLog().debug('frame=%d catalog result stale closing=%s revision=%d/%d matcher=%s matched=%s',
                           index, self._closing, revision, self._annotation_revision,
                           matcher is self._pin_matcher, index in self.__matched_frames)
            return
        try:
            analyzed = future.result()
        except Exception:
            getLog().exception('Background catalog matching failed for frame %d', index)
            return
        if analyzed is None:
            getLog().warning('Background catalog matching could not decode frame %d', index)
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
        for (original, _before, _worker_copy), result in zip(candidates, analyzed):
            if result.pin_id is None:
                continue
            if (original.pin_id, original.tags, original.match_confidence) != (
                    result.pin_id, result.tags, result.match_confidence):
                original.pin_id = result.pin_id
                original.tags = list(result.tags or [])
                original.match_confidence = result.match_confidence
                changed = True
        self.__matched_frames.add(index)
        if changed:
            self._invalidate_reconciliation()
        if self._current_index == index:
            if changed:
                self.__image_panel.Refresh()
                self.__tag_panel.boxes = self.__current_boxes
                self._sync_presence_ui()
            if not self._playing:
                self._schedule_reconciliation(index, reference_index)

    def on_resize(self, event: wx.CommandEvent):
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
                    self.__frame_boxes[next_index] = add_unique_boxes(
                        self.__get_frame_boxes(next_index), proposed)
                    self._invalidate_reconciliation()
                    self._reconcile_frame(next_index)
                    self._reconcile_done.add(next_index)
                    self.__frame_boxes[next_index] = identify_candidates(
                        next_frame, self.__frame_boxes[next_index], self._pin_matcher)
                    inherit_confirmed_catalog_matches(previous_boxes,
                                                      self.__frame_boxes[next_index])
                    if (marked_pin_count(self.__frame_boxes[next_index]) <
                            self._frame_presence.get(next_index, FramePresence()).minimum_pins
                            or not any(box.review_state != 'rejected'
                                       for box in self.__frame_boxes[next_index])):
                        extra = detect_pin_boxes_relaxed(next_frame, self._feedback_model)
                        self.__frame_boxes[next_index] = add_unique_boxes(
                            self.__frame_boxes[next_index], add_new_detections([], extra))
                        identify_candidates(next_frame, self.__frame_boxes[next_index],
                                            self._pin_matcher)
                        inherit_confirmed_catalog_matches(previous_boxes,
                                                          self.__frame_boxes[next_index])
                    propagate_confirmations(self.__frame_boxes)
                    self.__matched_frames.add(next_index)
                    self.__detected_frames.add(next_index)
            self._current_index = next_index
            self.display_image(reference_index=reference_index)

    def goto_frame(self, index: int, set_slider: bool = True) -> bool:
        """Go to a specific frame index."""
        self.pause_playback()
        if 0 <= index < self.__num_frames:
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
        event.Skip()

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
        shortcut = keycode in (*delete_keys, *wrong_identity_keys, *not_pin_keys)
        if keycode in delete_keys:
            actionable = selected
        elif keycode in wrong_identity_keys:
            actionable = selected if len(selected) == 1 and has_specific_identity(selected[0]) else []
        else:
            actionable = [box for box in selected if box.source == 'automatic']
        if shortcut and actionable and not modified:
            if editing_text:
                event.Skip()
            elif keycode in delete_keys:
                self.remove_boxes(selected)
            elif keycode in wrong_identity_keys:
                self.mark_wrong_identity(actionable[0])
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
        if any({'tags', 'pin_id', 'identity_confirmed', 'review_state', 'track_id',
                 'identity_rejections'}
               & change.changed_fields.keys() for change in entry.boxes):
            propagate_confirmations(self.__frame_boxes)
            entry = make_entry(before, snapshot(self.__frame_boxes), rotation,
                               self._rotation_angle, presence,
                               dict(self._frame_presence))
            self.__image_panel.Refresh()
            self.__tag_panel.boxes = self.__current_boxes
        if entry.changed:
            self._undo_history.setdefault(frame, []).append(entry)
            self._redo_history.pop(frame, None)
            self._sync_feedback_for_entry(entry)
            if any(change.box.identity_rejections or
                   'identity_rejections' in change.changed_fields
                   for change in entry.boxes):
                self._sync_identity_feedback()
            self._invalidate_reconciliation()
        self._sync_presence_ui()

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
        self._sync_identity_feedback()
        self._redo_history.setdefault(self._current_index, []).append(entry)
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
        self._sync_identity_feedback()
        self._undo_history.setdefault(self._current_index, []).append(entry)
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
                    except ValueError as error:
                        getLog().warning('Skipping invalid wrong-identity crop: %s', error)
        self._feedback_store.delete_missing_identity_rejections(self._feedback_source,
                                                                active)
        getLog().info('identity feedback: %d wrong-name examples indexed', len(active))

    def _seed_feedback_from_annotations(self) -> None:
        """Import explicit and legacy reviewed boxes once per source observation."""
        rejected_tracks = {box.track_id for boxes in self.__frame_boxes.values()
                           for box in boxes if box.feedback_label == 'negative'}
        protected_tracks = {box.track_id for boxes in self.__frame_boxes.values()
                            for box in boxes if box.source == 'user' or
                            box.review_state == 'confirmed'}
        stored = self._feedback_store.metadata_for_source(self._feedback_source)
        active_ids: set[str] = set()
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
                except ValueError as error:
                    getLog().warning(f'Skipping invalid feedback crop: {error}')
        self._feedback_store.delete_missing_source(self._feedback_source, active_ids)
        self._feedback_model.reload()
        self._update_feedback_status()

    def _update_feedback_status(self) -> None:
        stored = self._feedback_store.counts()
        usable = self._feedback_model.sample_counts
        message = (
            f"Feedback v{self._feedback_model.version}: "
            f"pin {usable['positive']}/{stored['positive']}, "
            f"non-pin {usable['negative']}/{stored['negative']} usable, "
            f"wrong ID {self._feedback_store.identity_rejection_count(self._feedback_source)} saved"
        )
        status = self.GetStatusBar()
        if status is not None:
            status.SetStatusText(message, 0)
        getLog().info('feedback model updated %s', message)

    def show_feedback_info(self) -> None:
        stored = self._feedback_store.counts()
        usable = self._feedback_model.sample_counts
        message = (
            f"Current feedback model: revision {self._feedback_model.version}\n"
            f"Confirmed pins: {stored['positive']} saved, {usable['positive']} usable\n"
            f"Rejected automatic candidates: {stored['negative']} saved, "
            f"{usable['negative']} usable\n"
            f"Wrong pin identities: {self._feedback_store.identity_rejection_count(self._feedback_source)} "
            "saved for this source\n\n"
            "Confirm is a pin adds a positive crop. Not a pin on an automatic "
            "area adds a negative crop. Remove area does not label a location "
            "as non-pin. Undo and redo update "
            "the saved examples and refresh the feedback model immediately. "
            "Rejecting a suggested identity saves one wrong-ID crop per pin, "
            "including every candidate in a None of these menu page. "
            "Add to training data indexes new or changed session reviews "
            "and removes examples no longer present.\n\n"
            "The current model is a local nearest-neighbour proposal ranker, "
            "not a trained neural detector. It can rank candidates and suppress "
            "close matches to rejected crops, but cannot find a pin that the "
            "proposal stage misses. Wrong-ID reviews exclude those identities "
            "from the current tracked area's suggestions; an identity model "
            "does not yet learn from them. Samples without enough visual detail "
            "remain saved for future training but are not usable by this ranker."
        )
        wx.MessageBox(message, 'Detection feedback', wx.OK | wx.ICON_INFORMATION)

    def add_to_training_data(self) -> None:
        """Index only new and changed direct reviews from this session."""
        if self._closing or self._training_future is not None:
            return
        self.pause_playback()
        if self.box_data_filename:
            save_boxes_to_file(self.box_data_filename, self.__frame_boxes,
                               self._frame_presence)
        observations = reviewed_observations(self.__frame_boxes, self._feedback_source)
        labels_by_id = {item.example_id: item.label for item in observations}
        try:
            delta = plan_training_delta(
                observations,
                self._feedback_store.training_metadata_for_source(self._feedback_source))
        except ValueError:
            getLog().exception('Could not plan incremental training update')
            self.__tag_panel.set_training_progress(
                0, 'Training data has duplicate example IDs; see console log.', False)
            return
        if not delta.pending:
            if delta.removed_ids:
                binary_ids = set(self._feedback_store.metadata_for_source(
                    self._feedback_source)) & delta.removed_ids
                self._feedback_store.apply_source_delta(
                    self._feedback_source, [], delta.removed_ids)
                if binary_ids:
                    self._feedback_model.refresh(binary_ids)
                self._update_feedback_status()
                message = (f'100% · {self._training_index_summary(labels_by_id, delta.active_ids)} · '
                           f'{len(delta.removed_ids)} removed')
            else:
                message = (f'100% · Up to date · '
                           f'{self._training_index_summary(labels_by_id, delta.active_ids)}')
            self.__tag_panel.set_training_progress(100, message, False)
            return
        revision = self._annotation_revision
        self.__tag_panel.set_training_progress(
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
            self._finish_training, finished, revision, delta, labels_by_id))

    @staticmethod
    def _training_index_summary(labels_by_id: dict[str, str], ids: set[str]) -> str:
        wrong_ids = sum(labels_by_id[item] == 'wrong_identity' for item in ids)
        return f'{len(ids)} examples indexed ({wrong_ids} wrong IDs)'

    def _show_training_progress(self, percent: int, message: str) -> None:
        if not self._closing and self._training_future is not None:
            self.__tag_panel.set_training_progress(percent, message, True)

    def _finish_training(self, future: Future, revision: int,
                         delta: TrainingDelta, labels_by_id: dict[str, str]) -> None:
        if self._closing or future is not self._training_future:
            return
        self._training_future = None
        try:
            examples, skipped = future.result()
            if revision != self._annotation_revision:
                self.__tag_panel.set_training_progress(
                    0, 'Annotations changed; run training again.', False)
                return
            self.__tag_panel.set_training_progress(95, '95% · updating model · estimating...', True)
            indexed_ids = {example.example_id for example in examples}
            failed_ids = {item.example_id for item in delta.pending} - indexed_ids
            removed_ids = delta.removed_ids | (failed_ids & delta.stored_ids)
            changed_ids = indexed_ids | removed_ids
            if changed_ids:
                binary_ids = ({item.example_id for item in examples
                               if item.label in ('positive', 'negative')} |
                              (removed_ids & set(self._feedback_store.metadata_for_source(
                                  self._feedback_source))))
                self._feedback_store.apply_source_delta(
                    self._feedback_source, examples, removed_ids)
                if binary_ids:
                    self._feedback_model.refresh(binary_ids)
                self._update_feedback_status()
            added = len(indexed_ids - delta.stored_ids)
            updated = len(indexed_ids & delta.stored_ids)
            self.__tag_panel.set_training_progress(
                100, f'100% · {self._training_index_summary(labels_by_id, delta.active_ids - failed_ids)} · '
                f'{added} added, {updated} updated, {len(removed_ids)} removed'
                + (f' · {skipped} skipped' if skipped else '') + ' · 0s remaining', False)
        except Exception:
            getLog().exception('Could not add session annotations to training data')
            self.__tag_panel.set_training_progress(0, 'Training failed; see console log.', False)

    def set_pin_matcher(self, matcher: PinMatcher) -> None:
        """Install a rebuilt reference index on the GUI thread."""
        self._pin_matcher = matcher
        self.__tag_panel.set_catalog_options(self._pin_catalog.list_pins())
        self._cancel_catalog_matches()
        self.__matched_frames.clear()
        self._invalidate_reconciliation()
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
        # Save boxes before exiting
        self.pause_playback()
        self._closing = True
        self._play_executor.submit(self.close_processing_reader)
        self._play_executor.shutdown(wait=False)
        count = self.count_boxes()
        save_boxes_to_file(self.__box_data_filename, self.__frame_boxes,
                           self._frame_presence)
        getLog().info(f'{count} boxes saved to {self.__box_data_filename}')
        event.Skip()  # Continue closing

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

    def confirm_boxes(self, selected: list[BoxData]) -> None:
        """Confirm multiple areas as pins in one undoable review."""
        self.pause_playback()
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
        propagate_confirmations(self.__frame_boxes)
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
        self.pause_playback()
        if (all(existing is not box for existing in self.__current_boxes)
                or not has_specific_identity(box)):
            return
        self._begin_history()
        box.review_state = 'confirmed'
        box.identity_confirmed = True
        box.identity_inferred = False
        box.feedback_id = box.feedback_id or uuid4().hex
        box.feedback_label = 'positive'
        propagate_confirmations(self.__frame_boxes)
        self.__tag_panel.boxes = self.__current_boxes
        self.__image_panel.Refresh()
        self._finish_history()
        if any(self.__frame_boxes.get(neighbour)
               for neighbour in (self._current_index - 1, self._current_index + 1)):
            self._schedule_reconciliation(self._current_index)

    def mark_wrong_identity(self, box: BoxData) -> None:
        """Reject only the proposed identity, keeping the pin-presence decision."""
        if not has_specific_identity(box):
            return
        self._reject_pin_identities(box, [(box.pin_id, tuple(box.tags or []))],
                                    retry_catalog=True)

    def rank_pin_replacements(self, box: BoxData, limit: int = 20) -> list[PinMatch]:
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
        return rank(
            frame[y:y + height, x:x + width], limit=limit,
            excluded_pin_ids=frozenset(item.pin_id for item in box.identity_rejections
                                       if item.pin_id),
            excluded_names=frozenset(tag.casefold() for item in box.identity_rejections
                                     if item.pin_id is None for tag in item.tags))

    def replace_with_pin(self, box: BoxData, match: PinMatch) -> None:
        """Treat the user's menu choice as a direct identity confirmation."""
        self.pause_playback()
        if (all(existing is not box for existing in self.__current_boxes) or
                box.review_state == 'rejected' or
                rejects_identity(box, match.pin_id, [match.name])):
            return
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
        propagate_confirmations(self.__frame_boxes)
        self.__tag_panel.boxes = self.__current_boxes
        self.__image_panel.Refresh()
        self._finish_history()
        if any(self.__frame_boxes.get(neighbour)
               for neighbour in (self._current_index - 1, self._current_index + 1)):
            self._schedule_reconciliation(self._current_index)

    def reject_pin_replacements(self, box: BoxData,
                                matches: list[PinMatch]) -> bool:
        """Reject one displayed page and let the menu show the next page."""
        return self._reject_pin_identities(
            box, [(match.pin_id, (match.name,)) for match in matches],
            retry_catalog=False)

    def _reject_pin_identities(self, box: BoxData,
                               candidates: list[tuple[str | None, tuple[str, ...]]],
                               retry_catalog: bool) -> bool:
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
        self._begin_history()
        self._apply_identity_rejections(box, reviews)
        self.__matched_frames.discard(self._current_index)
        self.__image_panel.Refresh()
        self.__tag_panel.boxes = self.__current_boxes
        self._finish_history()
        if retry_catalog:
            frame = (self._display_source_frame if self._display_source_frame is not None
                     else self.get_frame(self._current_index, 0))
            self._schedule_catalog_match(self._current_index, frame, None)
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
                               catalog_best_effort=True)

    def _submit_frame_job(self, target: int, requires_play: bool,
                          reference_index: int | None = None,
                          reference_frame: np.ndarray | None = None,
                          catalog_best_effort: bool = False) -> None:
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
            catalog_best_effort=catalog_best_effort,
        )
        generation = self._play_generation
        revision = self._annotation_revision
        future = self._play_executor.submit(process_frame, job, self.read_frame_for_processing)
        self._pending_frame_job = future
        future.add_done_callback(
            lambda completed: wx.CallAfter(self._complete_play_job, completed,
                                           generation, revision, requires_play))
        getLog().info('frame=%d ui submit=%.1fms', target, (perf_counter()-started)*1000)

    def _complete_play_job(self, future: Future[FrameResult], generation: int,
                           revision: int, requires_play: bool) -> None:
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
            elif (result.catalog_best_effort and
                  needs_catalog_match(original, best_effort=True) and
                  analyzed.pin_id is not None):
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
        self.__frame_boxes[target] = add_unique_boxes(
            existing, result.boxes[result.existing_count:])
        self._invalidate_reconciliation()
        if not requires_play:
            propagate_rejections(self.__frame_boxes)
            propagate_confirmations(self.__frame_boxes)
        self.__matched_frames.add(target)
        self.__detected_frames.add(target)
        if requires_play and not self.frame_has_boxes(target):
            self._primary_only_frames.add(target)
        else:
            self._primary_only_frames.discard(target)
        self._current_index = target
        committed = perf_counter()
        self.display_image(result.frame, reconcile=not requires_play)
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
            self._play_timer.StartOnce(50)

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
            self.__frame_boxes[self._current_index] = add_unique_boxes(
                self.__current_boxes, add_new_detections([], extra))
            identify_candidates(frame_img, self.__current_boxes,
                                self._pin_matcher, best_effort=True)
        propagate_rejections(self.__frame_boxes)
        self.__matched_frames.add(self._current_index)
        self.__detected_frames.add(self._current_index)
        self.display_image()
        getLog().info(f"Detected {len(detected) - len(existing)} new candidates in frame {self._current_index}")


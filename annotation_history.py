"""Small, identity-preserving changes for frame annotation undo and redo."""

from dataclasses import dataclass
from typing import Any

from boxdata import BoxData
from frame_presence import FramePresence


_FIELDS = ('coords', 'tags', 'source', 'track_id', 'pin_id',
           'review_state', 'prev_box_id', 'feedback_id', 'feedback_label',
           'match_confidence', 'identity_confirmed', 'identity_inferred',
           'identity_rejections')


@dataclass(frozen=True)
class BoxSnapshot:
    frame: int
    position: int
    box: BoxData
    values: dict[str, Any]

    @property
    def location(self) -> tuple[int, int]:
        return self.frame, self.position


@dataclass(frozen=True)
class BoxChange:
    box: BoxData
    before_location: tuple[int, int] | None
    after_location: tuple[int, int] | None
    changed_fields: dict[str, tuple[Any, Any]]


@dataclass(frozen=True)
class HistoryEntry:
    boxes: list[BoxChange]
    before_rotation: int
    after_rotation: int
    before_presence: dict[int, FramePresence] | None = None
    after_presence: dict[int, FramePresence] | None = None

    @property
    def changed(self) -> bool:
        return (bool(self.boxes) or self.before_rotation != self.after_rotation or
                self.before_presence != self.after_presence)

    def apply(self, frames: dict[int, list[BoxData]], undo: bool,
              presence: dict[int, FramePresence] | None = None) -> int:
        inserts: list[tuple[int, int, BoxData]] = []
        for change in self.boxes:
            destination = change.before_location if undo else change.after_location
            origin = change.after_location if undo else change.before_location
            if destination != origin:
                for boxes in frames.values():
                    boxes[:] = [box for box in boxes if box is not change.box]
                if destination is not None:
                    inserts.append((destination[0], destination[1], change.box))
            for field, (before, after) in change.changed_fields.items():
                value = before if undo else after
                setattr(change.box, field, list(value) if field == 'tags' and value is not None else value)
        for frame, position, box in sorted(inserts, key=lambda item: item[:2]):
            boxes = frames.setdefault(frame, [])
            boxes.insert(min(position, len(boxes)), box)
        if presence is not None:
            source = self.before_presence if undo else self.after_presence
            if source is not None:
                presence.clear()
                presence.update(source)
        return self.before_rotation if undo else self.after_rotation


def snapshot(frames: dict[int, list[BoxData]]) -> dict[int, BoxSnapshot]:
    result: dict[int, BoxSnapshot] = {}
    for frame, boxes in frames.items():
        for position, box in enumerate(boxes):
            values = {field: getattr(box, field) for field in _FIELDS}
            values['tags'] = tuple(box.tags) if box.tags is not None else None
            result[id(box)] = BoxSnapshot(frame, position, box, values)
    return result


def make_entry(before: dict[int, BoxSnapshot], after: dict[int, BoxSnapshot],
               before_rotation: int, after_rotation: int,
               before_presence: dict[int, FramePresence] | None = None,
               after_presence: dict[int, FramePresence] | None = None) -> HistoryEntry:
    changes: list[BoxChange] = []
    for identity in before.keys() | after.keys():
        old, new = before.get(identity), after.get(identity)
        old_location = old.location if old else None
        new_location = new.location if new else None
        fields = ({field: (old.values[field], new.values[field]) for field in _FIELDS
                   if old.values[field] != new.values[field]} if old and new else {})
        if old_location != new_location or fields:
            changes.append(BoxChange((old or new).box, old_location, new_location, fields))
    return HistoryEntry(changes, before_rotation, after_rotation,
                        before_presence, after_presence)

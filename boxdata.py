import re
from dataclasses import dataclass
from uuid import uuid4

Coordinate = tuple[int, int, int, int]
TagLabel = str


@dataclass(frozen=True)
class IdentityRejection:
    """A direct review of a wrong name; the area is still a pin."""

    example_id: str
    pin_id: str | None
    tags: tuple[str, ...]
    coords: Coordinate
    frame_index: int = -1


def is_placeholder_tag(tag: str) -> bool:
    return bool(re.fullmatch(r'unknown(?:-\d+)?', tag.strip(), re.IGNORECASE))


def has_specific_identity(box: 'BoxData') -> bool:
    return bool(box.pin_id or any(
        tag.strip() and not is_placeholder_tag(tag)
        for tag in box.tags or []))


def same_specific_name(first: 'BoxData', second: 'BoxData') -> bool:
    """Compare real pin names without treating legacy unknown labels as identities."""
    def names(box: 'BoxData') -> tuple[str, ...]:
        return tuple(tag.strip().casefold() for tag in box.tags or []
                     if tag.strip() and not is_placeholder_tag(tag))

    return bool(names(first)) and names(first) == names(second)


def rejects_identity(box: 'BoxData', pin_id: str | None,
                     tags: list[str] | None) -> bool:
    names = {tag.casefold() for tag in tags or [] if tag.strip()}
    return any((item.pin_id is not None and item.pin_id == pin_id) or
               (item.pin_id is None and bool(names & {
                   tag.casefold() for tag in item.tags}))
               for item in box.identity_rejections)

class BoxData:
    VALID_SOURCE_TYPES = ['user', 'automatic', 'processed']
    VALID_REVIEW_STATES = ('unconfirmed', 'inherited', 'confirmed', 'rejected')

    __id: str
    _tags: list[TagLabel]
    _coords: Coordinate
    _source: str | None = None
    __is_set: bool = False
    __has_backing_card: bool = False
    prev_box_id: str | None = None

    def __init__(self, coords: Coordinate, tags: list[TagLabel] | None, source: str,
                 track_id: str | None = None, pin_id: str | None = None,
                 review_state: str = 'unconfirmed', feedback_id: str | None = None,
                 feedback_label: str | None = None,
                 match_confidence: float | None = None,
                 identity_confirmed: bool | None = None,
                 identity_inferred: bool = False,
                 identity_rejections: tuple[IdentityRejection, ...] = ()):
        if source not in BoxData.VALID_SOURCE_TYPES:
            raise ValueError(f'Source must be "user" or "automatic" but was "{source}"')
        self._coords = coords
        self._tags = tags
        self._source = source
        self.__id = hex(id(self))
        self.track_id = track_id or uuid4().hex
        self.pin_id = pin_id
        self.review_state = review_state
        self.feedback_id = feedback_id
        self.feedback_label = feedback_label
        self.match_confidence = match_confidence
        # Older annotations used the green review state for both decisions.
        self.identity_confirmed = (review_state == 'confirmed' and has_specific_identity(self)
                                   if identity_confirmed is None else identity_confirmed)
        self.identity_inferred = identity_inferred
        self.identity_rejections = tuple(identity_rejections)

    @property
    def review_state(self) -> str:
        return self._review_state

    @review_state.setter
    def review_state(self, value: str) -> None:
        if value not in self.VALID_REVIEW_STATES:
            raise ValueError(f'Invalid review state: {value}')
        self._review_state = value

    @property
    def coords(self) -> Coordinate:
        return self._coords

    @coords.setter
    def coords(self, value: Coordinate) -> None:
        self._coords = value

    @property
    def tags(self) -> list[TagLabel] | None:
        return self._tags

    @tags.setter
    def tags(self, value: list[TagLabel] | None) -> None:
        self._tags = value

    def get_tag(self, index: int) -> TagLabel | None:
        if self._tags is None:
            return None

        """Get the tag at the specified index."""
        if 0 <= index < len(self._tags):
            return self._tags[index]
        raise IndexError("Tag index out of range")

    def set_tag(self, index: int, tag: TagLabel) -> None:
        """Set the tag at the specified index."""
        if 0 <= index < len(self._tags):
            self._tags[index] = tag
        else:
            raise IndexError("Tag index out of range")

    def add_tag(self, tag: TagLabel) -> int:
        if self._tags is None:
            self._tags = []

        """Add a new tag to the box."""
        self._tags.append(tag)
        return len(self._tags)

    def remove_tag(self, index: int) -> None:
        """Remove the tag at the specified index."""
        if self._tags is not None and 0 <= index < len(self._tags):
            del self._tags[index]
        else:
            raise IndexError("Tag index out of range")

    @property
    def source(self) -> str | None:
        """Get the source of the box data."""
        return self._source

    @source.setter
    def source(self, value: str | None) -> None:
        """Set the source of the box data."""
        if value is not None and not isinstance(value, str):
            raise ValueError("Source must be a string or None")
        self._source = value

    def __str__(self) -> str:
        return f'BoxData({self.source})<{hex(id(self))}>@{self.coords}={self.tags}'

    @property
    def is_set(self) -> bool:
        """Check if the box data is set."""
        return self.__is_set

    @is_set.setter
    def is_set(self, value: bool) -> None:
        """Set the is_set flag."""
        if not isinstance(value, bool):
            raise ValueError("is_set must be a boolean value")
        self.__is_set = value

    @property
    def has_backing_card(self) -> bool:
        """Check if the box has a backing card."""
        return self.__has_backing_card

    @has_backing_card.setter
    def has_backing_card(self, value: bool) -> None:
        """Set the has_backing_card flag."""
        if not isinstance(value, bool):
            raise ValueError("has_backing_card must be a boolean value")
        self.__has_backing_card = value

    def is_non_zero_sized(self) -> bool:
        return self.coords[2] > 0 and self.coords[3] > 0  # Check width and height

    @property
    def id(self):
        return self.__id

    def __eq__(self, other):
        """Check equality based on id, coords, tags, and source."""
        if not isinstance(other, BoxData):
            return False
        return (self.__id == other.id and
                self.coords == other.coords and
                self.tags == other.tags and
                self.source == other.source)

    def matches(self, other: 'BoxData') -> bool:
        """Check if this box matches another box based on coordinates and tags."""
        if not isinstance(other, BoxData):
            return False
        if self.__id == other.id:
            return True
        return (self.coords == other.coords and
                self.tags == other.tags and
                self.source == other.source)

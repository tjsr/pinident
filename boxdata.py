Coordinate = tuple[int, int, int, int]
TagLabel = str

class BoxData:
    _tags: list[TagLabel]
    _coords: Coordinate
    _source: str | None = None

    def __init__(self, coords: Coordinate, tags: list[TagLabel], source: str):
        if source not in ['user', 'automatic']:
            raise ValueError("Source must be 'user' or 'automatic'")
        self._coords = coords
        self._tags = tags
        self._source = source

    @property
    def coords(self) -> Coordinate:
        return self._coords

    @coords.setter
    def coords(self, value: Coordinate) -> None:
        self._coords = value

    @property
    def tags(self) -> list[TagLabel]:
        return self._tags

    @tags.setter
    def tags(self, value: list[TagLabel]) -> None:
        self._tags = value

    def get_tag(self, index: int) -> TagLabel:
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
        """Add a new tag to the box."""
        self._tags.append(tag)
        return len(self._tags)

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
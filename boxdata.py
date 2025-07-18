Coordinate = tuple[int, int, int, int]
TagLabel = str

class BoxData:
    _tags: list[TagLabel]
    _coords: Coordinate

    def __init__(self, coords: Coordinate, tags: list[TagLabel]):
        self._coords = coords
        self._tags = tags

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

    def GetTag(self, index: int) -> TagLabel:
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
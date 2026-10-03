"""Geometry for source-image boxes shown in a scaled or rotated view."""

from boxdata import Coordinate

ResizeHandle = str


def display_size(source_size: tuple[int, int], angle: int) -> tuple[int, int]:
    width, height = source_size
    return (height, width) if angle % 180 else (width, height)


def rotate_rect(rect: Coordinate, source_size: tuple[int, int], angle: int) -> Coordinate:
    x, y, width, height = rect
    source_width, source_height = source_size
    angle %= 360
    if angle == 90:
        return source_height - y - height, x, height, width
    if angle == 180:
        return source_width - x - width, source_height - y - height, width, height
    if angle == 270:
        return y, source_width - x - width, height, width
    return rect


def unrotate_rect(rect: Coordinate, source_size: tuple[int, int], angle: int) -> Coordinate:
    x, y, width, height = rect
    source_width, source_height = source_size
    angle %= 360
    if angle == 90:
        return y, source_height - x - width, height, width
    if angle == 180:
        return source_width - x - width, source_height - y - height, width, height
    if angle == 270:
        return source_width - y - height, x, height, width
    return rect


def hit_test_rect(rect: Coordinate, point: tuple[int, int], margin: int = 6) -> ResizeHandle | None:
    """Identify a corner before an edge, with tolerance in displayed pixels."""
    x, y, width, height = rect
    px, py = point
    right, bottom = x + width, y + height
    if not (x - margin <= px <= right + margin and y - margin <= py <= bottom + margin):
        return None
    horizontal = None
    vertical = None
    if min(abs(px - x), abs(px - right)) <= margin:
        horizontal = 'left' if abs(px - x) <= abs(px - right) else 'right'
    if min(abs(py - y), abs(py - bottom)) <= margin:
        vertical = 'top' if abs(py - y) <= abs(py - bottom) else 'bottom'
    if horizontal and vertical:
        return vertical + horizontal
    return horizontal or vertical


def resize_rect(rect: Coordinate, handle: ResizeHandle, point: tuple[int, int],
                bounds: tuple[int, int]) -> Coordinate:
    """Move selected edges, clamped to the image and at least one pixel wide."""
    x, y, width, height = rect
    left, top, right, bottom = x, y, x + width, y + height
    px, py = point
    max_width, max_height = bounds
    if 'left' in handle:
        left = max(0, min(px, right - 1))
    if 'right' in handle:
        right = min(max_width, max(px, left + 1))
    if 'top' in handle:
        top = max(0, min(py, bottom - 1))
    if 'bottom' in handle:
        bottom = min(max_height, max(py, top + 1))
    return left, top, right - left, bottom - top

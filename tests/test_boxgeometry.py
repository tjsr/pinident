import pytest

from controls.boxgeometry import display_size, hit_test_rect, resize_rect, rotate_rect, unrotate_rect


@pytest.mark.parametrize('angle', [0, 90, 180, 270])
def test_rotation_round_trip(angle):
    source_size = (400, 300)
    rect = (40, 60, 100, 80)
    shown = rotate_rect(rect, source_size, angle)

    assert unrotate_rect(shown, source_size, angle) == rect
    assert 0 <= shown[0] < display_size(source_size, angle)[0]
    assert 0 <= shown[1] < display_size(source_size, angle)[1]


@pytest.mark.parametrize(('point', 'expected'), [
    ((50, 70), 'topleft'), ((150, 70), 'topright'),
    ((50, 150), 'bottomleft'), ((150, 150), 'bottomright'),
    ((50, 110), 'left'), ((150, 110), 'right'),
    ((100, 70), 'top'), ((100, 150), 'bottom'),
    ((100, 110), None), ((20, 20), None),
])
def test_hit_test_all_edges_and_corners(point, expected):
    assert hit_test_rect((50, 70, 100, 80), point) == expected


def test_corner_resize_moves_two_edges_and_clamps_to_image():
    rect = (50, 70, 100, 80)
    assert resize_rect(rect, 'topleft', (70, 90), (400, 300)) == (70, 90, 80, 60)
    assert resize_rect(rect, 'bottomright', (999, 999), (400, 300)) == (50, 70, 350, 230)
    assert resize_rect(rect, 'left', (999, 100), (400, 300)) == (149, 70, 1, 80)

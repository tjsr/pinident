import json
from io import StringIO

import cv2
import numpy as np
import pytest
import wx

from boxdata import BoxData
from controls.boxgeometry import rotate_rect
from scrubberframe import save_boxes_to_stream
from videoscrubber import VideoScrubber


class Mouse:
    def __init__(self, x: int, y: int, left_down: bool = False):
        self.position = wx.Point(x, y)
        self.left_down = left_down

    def GetPosition(self):
        return self.position

    def LeftIsDown(self):
        return self.left_down

    def Skip(self):
        pass


def make_window():
    app = wx.App.Get() or wx.App(False)
    source = np.zeros((400, 800, 3), dtype=np.uint8)
    window = VideoScrubber(None, 'Resize test', '',
                           image_array=[cv2.cvtColor(source, cv2.COLOR_RGB2BGR)],
                           load_catalog=False)
    window.Show()
    wx.Yield()
    box = BoxData((100, 80, 200, 160), ['pin'], 'automatic')
    window._ScrubberFrame__frame_boxes[0] = [box]
    window.display_image()
    wx.Yield()
    return app, window, box, source


@pytest.mark.gui
def test_hovering_narrow_box_shows_full_label_and_clears_when_leaving():
    app, window, box, source = make_window()
    try:
        panel = window._ScrubberFrame__image_panel
        panel.SetSize((400, 300))
        panel.set_image(source, 0)
        box.coords = (100, 80, 80, 100)
        box.tags = ['Very Long Pin Name That Cannot Fit In This Box']
        box.pin_id = 'pin-long'
        box.match_confidence = 0.42
        full_label = panel.get_box_label_text(box)

        x, y, width, height = panel._box_panel_rect(box)
        panel.on_motion(Mouse(x + width // 2, y + height // 2))
        assert panel.GetToolTip().GetTip() == full_label

        panel.on_motion(Mouse(0, 0))
        assert panel.GetToolTip() is None

        panel.on_motion(Mouse(x + width // 2, y + height // 2))
        box.review_state = 'rejected'
        panel.on_motion(Mouse(x + width // 2, y + height // 2))
        assert panel.GetToolTip() is None

        box.review_state = 'unconfirmed'
        panel.on_motion(Mouse(x + width // 2, y + height // 2))
        panel.boxes = []
        assert panel.GetToolTip() is None
    finally:
        window.Destroy()
        wx.Yield()


@pytest.mark.gui
@pytest.mark.parametrize('panel_size', [(400, 300), (800, 500)])
def test_corner_resize_updates_shared_box_tag_panel_and_json(panel_size):
    app, window, box, source = make_window()
    try:
        panel = window._ScrubberFrame__image_panel
        panel.SetSize(panel_size)
        panel.set_image(source, 0)
        x, y, width, height = panel._box_panel_rect(box)

        cursor_cases = [
            (x, y, wx.CURSOR_SIZENWSE),
            (x + width, y, wx.CURSOR_SIZENESW),
            (x, y + height, wx.CURSOR_SIZENESW),
            (x + width, y + height, wx.CURSOR_SIZENWSE),
            (x, y + height // 2, wx.CURSOR_SIZEWE),
            (x + width // 2, y, wx.CURSOR_SIZENS),
        ]
        for cursor_x, cursor_y, expected in cursor_cases:
            panel.on_motion(Mouse(cursor_x, cursor_y))
            assert panel.GetCursor().GetHandle() == wx.Cursor(expected).GetHandle()

        panel.on_motion(Mouse(x, y))
        assert panel.GetCursor().GetHandle() == wx.Cursor(wx.CURSOR_SIZENWSE).GetHandle()
        panel.on_left_down(Mouse(x, y, True))
        panel.on_motion(Mouse(x + 20, y + 12, True))
        wx.Yield()

        assert box.coords[0] > 100 and box.coords[1] > 80
        assert box.coords[0] + box.coords[2] == 300
        assert box.coords[1] + box.coords[3] == 240
        assert len(panel.boxes) == 1
        assert window._ScrubberFrame__frame_boxes[0][0] is box
        tag_panel = window._ScrubberFrame__tag_panel.find_panel_for_box(box)
        heading = next(child for child in tag_panel.GetChildren() if isinstance(child, wx.StaticText))
        assert str(box.coords) in heading.GetLabel()

        output = StringIO()
        save_boxes_to_stream(output, window._ScrubberFrame__frame_boxes)
        assert json.loads(output.getvalue())['0'][0]['coords'] == list(box.coords)

        panel.on_left_up(Mouse(x + 20, y + 12))
        assert not panel.HasCapture()
    finally:
        window.Destroy()
        wx.Yield()


@pytest.mark.gui
def test_dragging_inside_box_moves_it_without_resizing():
    app, window, box, source = make_window()
    try:
        panel = window._ScrubberFrame__image_panel
        panel.SetSize((400, 300))
        panel.set_image(source, 0)
        x, y, width, height = panel._box_panel_rect(box)
        center_x, center_y = x + width // 2, y + height // 2

        panel.on_left_down(Mouse(center_x, center_y, True))
        panel.on_motion(Mouse(center_x + 15, center_y + 10, True))
        panel.on_left_up(Mouse(center_x + 15, center_y + 10))

        assert box.coords[0] > 100 and box.coords[1] > 80
        assert box.coords[2:] == (200, 160)
        assert len(panel.boxes) == 1
    finally:
        window.Destroy()
        wx.Yield()


@pytest.mark.gui
def test_drawing_a_new_box_still_uses_source_coordinates():
    app, window, box, source = make_window()
    try:
        panel = window._ScrubberFrame__image_panel
        panel.SetSize((400, 300))
        panel.set_image(source, 0)

        panel.on_left_down(Mouse(250, 100, True))
        panel.on_motion(Mouse(300, 140, True))
        panel.on_left_up(Mouse(300, 140))
        wx.Yield()

        boxes = window._ScrubberFrame__frame_boxes[0]
        assert len(boxes) == 2
        assert boxes[1].coords == (500, 100, 100, 80)
        assert boxes[1].source == 'user'
    finally:
        window.Destroy()
        wx.Yield()


@pytest.mark.gui
def test_rotated_edge_resize_changes_source_box():
    app, window, box, source = make_window()
    try:
        panel = window._ScrubberFrame__image_panel
        panel.SetSize((400, 300))
        rotated = cv2.rotate(source, cv2.ROTATE_90_CLOCKWISE)
        panel.set_image(rotated, 90)
        x, y, width, height = panel._box_panel_rect(box)
        initial_display = rotate_rect(box.coords, (800, 400), 90)

        panel.on_motion(Mouse(x, y + height // 2))
        assert panel.GetCursor().GetHandle() == wx.Cursor(wx.CURSOR_SIZEWE).GetHandle()
        panel.on_left_down(Mouse(x, y + height // 2, True))
        panel.on_motion(Mouse(x + 18, y + height // 2, True))
        panel.on_left_up(Mouse(x + 18, y + height // 2))
        wx.Yield()

        shown = rotate_rect(box.coords, (800, 400), 90)
        assert shown[0] > initial_display[0]
        assert shown[0] + shown[2] == initial_display[0] + initial_display[2]
        assert box.coords[0] >= 0 and box.coords[1] >= 0
        assert box.coords[0] + box.coords[2] <= 800
        assert box.coords[1] + box.coords[3] <= 400
    finally:
        window.Destroy()
        wx.Yield()

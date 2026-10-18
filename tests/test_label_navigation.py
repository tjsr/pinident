import numpy as np
import pytest
import wx

from boxdata import BoxData
from videoscrubber import VideoScrubber


class Click:
    def __init__(self, x: int, y: int):
        self.position = wx.Point(x, y)

    def GetPosition(self):
        return self.position

    def ControlDown(self):
        return False

    def Skip(self):
        pass


@pytest.mark.gui
def test_clicking_rectangle_focuses_and_reveals_its_label_editor(tmp_path):
    app = wx.App.Get() or wx.App(False)
    window = VideoScrubber(None, 'Label navigation', '',
                           image_array=[np.zeros((120, 240, 3), dtype=np.uint8)],
                           load_catalog=False,
                           feedback_path=tmp_path / 'feedback.sqlite3')
    try:
        window.SetSize((1100, 480))
        window.Show()
        boxes = [BoxData((10 + (index % 4) * 55, 10 + (index // 4) * 25,
                          20, 20), [f'Pin {index}'], 'automatic')
                 for index in range(16)]
        window._ScrubberFrame__frame_boxes[0] = boxes
        window.display_image(reconcile=False)
        wx.Yield()

        image_panel = window._ScrubberFrame__image_panel
        tag_panel = window._ScrubberFrame__tag_panel
        box = boxes[-1]
        editor = tag_panel.find_panel_for_box(box)
        assert editor.GetPosition().y >= tag_panel.GetClientSize().height

        x, y, width, height = image_panel._box_panel_rect(box)
        click = Click(x + width // 2, y + height // 2)
        image_panel.on_left_down(click)
        image_panel.on_left_up(click)
        wx.Yield()

        assert wx.Window.FindFocus() is editor
        assert editor.GetRect().y >= 0
        assert editor.GetRect().GetBottom() <= tag_panel.GetClientSize().height

        image_panel.SetFocus()
        wx.Yield()
        image_panel.on_left_down(click)
        image_panel.on_left_up(click)
        wx.Yield()
        assert wx.Window.FindFocus() is editor
    finally:
        window.Destroy()
        wx.Yield()

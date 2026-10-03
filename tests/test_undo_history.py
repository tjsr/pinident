from io import StringIO

import numpy as np
import pytest
import wx

from boxdata import BoxData
from events.BoxLabelEditEvent import BoxLabelEditedEvent
from scrubberframe import save_boxes_to_stream
from videoscrubber import VideoScrubber


class Mouse:
    def __init__(self, x: int, y: int, down: bool = False):
        self.position = wx.Point(x, y)
        self.down = down

    def GetPosition(self):
        return self.position

    def LeftIsDown(self):
        return self.down

    def Skip(self):
        pass


class UndoKey:
    def __init__(self, redo: bool = False):
        self.redo = redo

    def GetKeyCode(self):
        return ord('Z')

    def ControlDown(self):
        return True

    def ShiftDown(self):
        return self.redo


@pytest.mark.gui
def test_undo_draw_is_frame_local_and_preserves_later_identification():
    app = wx.App.Get() or wx.App(False)
    image = np.zeros((120, 200, 3), dtype=np.uint8)
    window = VideoScrubber(None, 'Undo frame test', '', image_array=[image] * 2,
                           load_catalog=False)
    try:
        window.Show()
        wx.Yield()
        first = BoxData((10, 15, 40, 40), ['Old'], 'user', track_id='first')
        second = BoxData((110, 15, 40, 40), ['Luigi'], 'user', track_id='second',
                         pin_id='pinpanion:8', review_state='confirmed')
        window._ScrubberFrame__frame_boxes.update({0: [first], 1: [second]})
        window.display_image()
        panel = window._ScrubberFrame__image_panel
        x, y = panel.get_image_offset()
        panel.on_left_down(Mouse(x + 75, y + 70, True))
        panel.on_left_up(Mouse(x + 95, y + 90))
        wx.Yield()
        assert len(window._ScrubberFrame__frame_boxes[0]) == 2

        first.tags = ['Mario']
        first.pin_id = 'pinpanion:7'
        assert window.goto_frame(1)
        window.on_key_down(UndoKey())
        wx.Yield()
        assert panel.boxes == [second]
        assert window._ScrubberFrame__frame_boxes[1] == [second]

        assert window.goto_frame(0)
        window.on_key_down(UndoKey())
        wx.Yield()
        assert panel.boxes == [first]
        assert window._ScrubberFrame__frame_boxes[0] == [first]
        assert first.tags == ['Mario'] and first.pin_id == 'pinpanion:7'
        assert first.coords == (10, 15, 40, 40)

        output = StringIO()
        save_boxes_to_stream(output, window._ScrubberFrame__frame_boxes)
        assert 'Mario' in output.getvalue() and 'Luigi' in output.getvalue()
    finally:
        window.Destroy()
        wx.Yield()


@pytest.mark.gui
@pytest.mark.parametrize('delete_first', [False, True])
def test_delete_undo_and_redo_restore_the_same_box_without_touching_neighbours(delete_first):
    app = wx.App.Get() or wx.App(False)
    image = np.zeros((120, 200, 3), dtype=np.uint8)
    window = VideoScrubber(None, 'Undo deletion test', '', image_array=[image],
                           load_catalog=False)
    try:
        window.Show()
        wx.Yield()
        known = BoxData((10, 15, 40, 40), ['Mario'], 'user', track_id='known',
                        pin_id='pinpanion:7', review_state='confirmed')
        deleted = BoxData((110, 15, 40, 40), ['Luigi'], 'user', track_id='deleted',
                          pin_id='pinpanion:8', review_state='confirmed')
        window._ScrubberFrame__frame_boxes[0] = [known, deleted]
        window.display_image()
        removed, kept = (known, deleted) if delete_first else (deleted, known)
        label = window._ScrubberFrame__tag_panel.find_panel_for_box(removed)
        label.delete_button.ProcessEvent(wx.CommandEvent(
            wx.EVT_BUTTON.typeId, label.delete_button.GetId()))
        assert window._ScrubberFrame__frame_boxes[0] == [kept]

        window.on_key_down(UndoKey())
        wx.Yield()
        assert window._ScrubberFrame__frame_boxes[0] == [known, deleted]
        assert window._ScrubberFrame__image_panel.boxes == [known, deleted]
        assert deleted.coords == (110, 15, 40, 40)
        assert deleted.pin_id == 'pinpanion:8' and deleted.tags == ['Luigi']

        window.on_key_down(UndoKey(redo=True))
        wx.Yield()
        assert window._ScrubberFrame__frame_boxes[0] == [kept]
        assert kept.tags == (['Luigi'] if delete_first else ['Mario'])
        assert kept.coords == ((110, 15, 40, 40) if delete_first else (10, 15, 40, 40))
    finally:
        window.Destroy()
        wx.Yield()


@pytest.mark.gui
def test_undo_rejection_restores_metadata_in_all_affected_frames():
    app = wx.App.Get() or wx.App(False)
    image = np.zeros((120, 200, 3), dtype=np.uint8)
    window = VideoScrubber(None, 'Undo rejection test', '', image_array=[image] * 2,
                           load_catalog=False)
    try:
        window.Show()
        wx.Yield()
        before = BoxData((10, 15, 40, 40), ['Maybe pin'], 'automatic', track_id='same')
        current = BoxData((10, 15, 40, 40), ['Maybe pin'], 'automatic', track_id='same')
        window._ScrubberFrame__frame_boxes.update({0: [before], 1: [current]})
        window.goto_frame(1)
        window.mark_not_pin(current)
        assert before.review_state == current.review_state == 'rejected'
        assert before.tags == current.tags == ['Maybe pin']

        window.on_key_down(UndoKey())
        assert before.review_state == current.review_state == 'unconfirmed'
        assert before.tags == current.tags == ['Maybe pin']
        assert window._ScrubberFrame__frame_boxes[1][0] is current

        window.on_key_down(UndoKey(redo=True))
        assert before.review_state == current.review_state == 'rejected'
    finally:
        window.Destroy()
        wx.Yield()


@pytest.mark.gui
def test_resize_and_label_edit_undo_change_only_the_edited_fields():
    app = wx.App.Get() or wx.App(False)
    image = np.zeros((120, 200, 3), dtype=np.uint8)
    window = VideoScrubber(None, 'Undo edits test', '', image_array=[image],
                           load_catalog=False)
    try:
        window.Show()
        wx.Yield()
        box = BoxData((10, 15, 40, 40), ['Mario'], 'automatic',
                      pin_id='pinpanion:7', track_id='known')
        window._ScrubberFrame__frame_boxes[0] = [box]
        window.display_image()
        panel = window._ScrubberFrame__image_panel
        x, y, width, height = panel._box_panel_rect(box)
        panel.on_left_down(Mouse(x + width, y + height, True))
        panel.on_left_up(Mouse(x + width + 10, y + height + 10))
        assert box.coords != (10, 15, 40, 40)
        resized = box.coords

        tag_panel = window._ScrubberFrame__tag_panel.find_panel_for_box(box)
        tag_panel._BoxTagPanelEdit__on_label_edited(
            BoxLabelEditedEvent(tag_panel, 0, 'Corrected Mario'))
        assert box.tags == ['Corrected Mario'] and box.pin_id is None

        window.on_key_down(UndoKey())
        assert box.coords == resized
        assert box.tags == ['Mario'] and box.pin_id == 'pinpanion:7'
        window.on_key_down(UndoKey())
        assert box.coords == (10, 15, 40, 40)
        assert window._ScrubberFrame__frame_boxes[0][0] is box

        window.on_key_down(UndoKey(redo=True))
        assert box.coords == resized
        assert box.tags == ['Mario'] and box.pin_id == 'pinpanion:7'
    finally:
        window.Destroy()
        wx.Yield()


@pytest.mark.gui
def test_rotation_undo_changes_view_without_replacing_boxes():
    app = wx.App.Get() or wx.App(False)
    image = np.zeros((120, 200, 3), dtype=np.uint8)
    window = VideoScrubber(None, 'Undo rotation test', '', image_array=[image],
                           load_catalog=False)
    try:
        window.Show()
        wx.Yield()
        box = BoxData((10, 15, 40, 40), ['Mario'], 'user',
                      pin_id='pinpanion:7', review_state='confirmed')
        window._ScrubberFrame__frame_boxes[0] = [box]
        window.display_image()
        window.on_rotate_cw(None)
        assert window._rotation_angle == 90

        window.on_key_down(UndoKey())
        assert window._rotation_angle == 0
        assert window._ScrubberFrame__image_panel.boxes == [box]
        assert box.coords == (10, 15, 40, 40) and box.tags == ['Mario']

        window.on_key_down(UndoKey(redo=True))
        assert window._rotation_angle == 90
        assert window._ScrubberFrame__frame_boxes[0][0] is box
    finally:
        window.Destroy()
        wx.Yield()

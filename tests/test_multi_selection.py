"""Image-panel multi-selection and one-step bulk review actions."""

from io import StringIO
from unittest.mock import patch

import cv2
import numpy as np
import pytest
import wx

from boxdata import BoxData
from scrubberframe import save_boxes_to_stream
from videoscrubber import VideoScrubber


class Mouse:
    def __init__(self, x: int, y: int, *, down: bool = False, ctrl: bool = False):
        self.position = wx.Point(x, y)
        self.down = down
        self.ctrl = ctrl

    def GetPosition(self):
        return self.position

    def LeftIsDown(self):
        return self.down

    def ControlDown(self):
        return self.ctrl

    def Skip(self):
        pass


class Key:
    def __init__(self, code: int, *, ctrl: bool = False):
        self.code = code
        self.ctrl = ctrl
        self.skipped = False

    def GetKeyCode(self):
        return self.code

    def ControlDown(self):
        return self.ctrl

    def ShiftDown(self):
        return False

    def AltDown(self):
        return False

    def MetaDown(self):
        return False

    def Skip(self):
        self.skipped = True


def make_window():
    app = wx.App.Get() or wx.App(False)
    image = np.random.default_rng(405).integers(0, 255, (120, 240, 3), dtype=np.uint8)
    window = VideoScrubber(None, 'Multi selection', '',
                           image_array=[cv2.cvtColor(image, cv2.COLOR_RGB2BGR)],
                           load_catalog=False)
    window.Show()
    wx.Yield()
    boxes = [BoxData((x, 20, 30, 30), [f'Pin {index}'], 'automatic')
             for index, x in enumerate((20, 90, 180), 1)]
    window._ScrubberFrame__frame_boxes[0] = boxes
    window.display_image(reconcile=False)
    return app, window, image, boxes


def center(panel, box):
    x, y, width, height = panel._box_panel_rect(box)
    return x + width // 2, y + height // 2


def activate(menu, label):
    item = next(item for item in menu.GetMenuItems()
                if item.GetItemLabelText() == label)
    assert item.IsEnabled()
    menu.ProcessEvent(wx.CommandEvent(wx.EVT_MENU.typeId, item.GetId()))
    menu.Destroy()


@pytest.mark.gui
def test_ctrl_click_toggles_boxes_and_plain_click_selects_one():
    app, window, _image, boxes = make_window()
    try:
        panel = window._ScrubberFrame__image_panel
        for box in boxes[:2]:
            x, y = center(panel, box)
            panel.on_left_down(Mouse(x, y, ctrl=True))
            panel.on_left_up(Mouse(x, y, ctrl=True))
        assert panel.selected_boxes == boxes[:2]
        assert all(panel._is_box_selected(box) for box in boxes[:2])
        assert panel.selected_box is boxes[1]
        x, y = center(panel, boxes[0])
        panel.on_left_down(Mouse(x, y, ctrl=True))
        panel.on_left_up(Mouse(x, y, ctrl=True))
        assert panel.selected_boxes == [boxes[1]]
        x, y = center(panel, boxes[2])
        panel.on_left_down(Mouse(x, y, down=True))
        panel.on_left_up(Mouse(x, y))
        assert panel.selected_boxes == [boxes[2]]
    finally:
        window.Destroy()
        wx.Yield()


@pytest.mark.gui
def test_right_click_preserves_group_and_uses_bulk_menu():
    app, window, _image, boxes = make_window()
    try:
        panel = window._ScrubberFrame__image_panel
        panel._set_selection(boxes[:2])
        labels = []
        with patch.object(panel, 'PopupMenu', side_effect=lambda menu: labels.extend(
                item.GetItemLabelText() for item in menu.GetMenuItems())):
            x, y = center(panel, boxes[0])
            panel.on_right_up(Mouse(x, y))
        assert panel.selected_boxes == boxes[:2]
        assert labels == ['Confirm selected as pins', 'Not a pin (automatic only)',
                          'Remove selected areas']
        menu = panel.make_selection_context_menu(boxes[:2])
        assert [item.GetItemLabel() for item in menu.GetMenuItems()][1:] == [
            '&Not a pin (automatic only)', 'Remove selected areas\tDel']
        menu.Destroy()

        labels.clear()
        with patch.object(panel, 'PopupMenu', side_effect=lambda menu: labels.extend(
                item.GetItemLabelText() for item in menu.GetMenuItems())):
            x, y = center(panel, boxes[2])
            panel.on_right_up(Mouse(x, y))
        assert panel.selected_boxes == [boxes[2]]
        assert 'Remove area' in labels and 'Remove selected areas' not in labels
    finally:
        window.Destroy()
        wx.Yield()


@pytest.mark.gui
@pytest.mark.parametrize('angle,panel_size', [(0, (200, 150)), (90, (200, 150))])
def test_marquee_overlapping_boxes_selects_without_drawing(angle, panel_size):
    app, window, image, boxes = make_window()
    try:
        panel = window._ScrubberFrame__image_panel
        panel.SetSize(panel_size)
        shown = (cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                 if angle == 90 else image)
        panel.set_image(shown, angle)
        rects = [panel._box_panel_rect(box) for box in boxes[:2]]
        left = min(rect[0] for rect in rects) - 10
        top = min(rect[1] for rect in rects) - 10
        right = max(rect[0] + rect[2] for rect in rects) + 10
        bottom = max(rect[1] + rect[3] for rect in rects) + 10
        assert panel._point_on_image(wx.Point(left, top))
        panel.on_left_down(Mouse(left, top, down=True))
        panel.on_motion(Mouse(right, bottom, down=True))
        panel.on_left_up(Mouse(right, bottom))
        assert panel.selected_boxes == boxes[:2]
        assert window._ScrubberFrame__frame_boxes[0] == boxes
        assert not window._undo_history.get(0)
    finally:
        window.Destroy()
        wx.Yield()


@pytest.mark.gui
@pytest.mark.parametrize('right,should_select', [(35, False), (36, True)])
def test_marquee_selects_only_when_it_covers_more_than_half_an_existing_box(
        right, should_select):
    app, window, image, boxes = make_window()
    try:
        original = list(boxes)
        panel = window._ScrubberFrame__image_panel
        panel.SetSize((240, 120))
        panel.set_image(image, 0)
        # The first box is 30x30 at (20, 20). This drag covers exactly 50%
        # of it at x=35, and 16/30 of it at x=36.
        panel.on_left_down(Mouse(10, 10, down=True))
        panel.on_motion(Mouse(right, 50, down=True))
        panel.on_left_up(Mouse(right, 50))

        current = window._ScrubberFrame__frame_boxes[0]
        if should_select:
            assert current == original
            assert panel.selected_boxes == [original[0]]
        else:
            assert current[:3] == original
            assert len(current) == 4
            assert current[-1].coords == (10, 10, 25, 40)
            assert current[-1].review_state == 'confirmed'
            assert panel.selected_boxes == []
    finally:
        window.Destroy()
        wx.Yield()


@pytest.mark.gui
def test_bulk_confirm_not_pin_and_remove_are_single_undo_actions():
    app, window, _image, boxes = make_window()
    original = list(boxes)
    try:
        panel = window._ScrubberFrame__image_panel
        panel._set_selection(boxes[:2])
        activate(panel.make_selection_context_menu(panel.selected_boxes),
                 'Confirm selected as pins')
        assert [box.review_state for box in boxes] == [
            'confirmed', 'confirmed', 'unconfirmed']
        assert len(window._undo_history[0]) == 1
        assert window._feedback_store.counts()['positive'] == 2
        window.undo_action()
        assert all(box.review_state == 'unconfirmed' for box in boxes)
        assert window._feedback_store.counts()['positive'] == 0

        panel._set_selection(boxes[:2])
        activate(panel.make_selection_context_menu(panel.selected_boxes),
                 'Not a pin (automatic only)')
        assert [box.review_state for box in boxes] == [
            'rejected', 'rejected', 'unconfirmed']
        assert panel.selected_boxes == []
        assert window._feedback_store.counts()['negative'] == 2
        window.undo_action()
        assert all(box.review_state == 'unconfirmed' for box in boxes)
        assert window._feedback_store.counts()['negative'] == 0

        panel._set_selection(boxes[:2])
        activate(panel.make_selection_context_menu(panel.selected_boxes),
                 'Remove selected areas')
        assert window._ScrubberFrame__frame_boxes[0] == [original[2]]
        assert window._feedback_store.counts() == {'positive': 0, 'negative': 0}
        window.undo_action()
        assert window._ScrubberFrame__frame_boxes[0] == original
        output = StringIO()
        save_boxes_to_stream(output, window._ScrubberFrame__frame_boxes)
        assert output.getvalue().count('"coords"') == 3
    finally:
        window.Destroy()
        wx.Yield()


@pytest.mark.gui
def test_bulk_not_pin_keeps_manual_boxes_untouched():
    app, window, _image, boxes = make_window()
    try:
        panel = window._ScrubberFrame__image_panel
        boxes[1].source = 'user'
        panel._set_selection(boxes[:2])
        activate(panel.make_selection_context_menu(panel.selected_boxes),
                 'Not a pin (automatic only)')
        assert boxes[0].review_state == 'rejected'
        assert boxes[1].review_state == 'unconfirmed'
        assert window._feedback_store.counts()['negative'] == 1
    finally:
        window.Destroy()
        wx.Yield()


@pytest.mark.gui
def test_delete_shortcut_removes_selection_without_negative_feedback():
    app, window, _image, boxes = make_window()
    original = list(boxes)
    try:
        panel = window._ScrubberFrame__image_panel
        panel._set_selection(boxes[:2])
        panel.SetFocus()
        wx.Yield()
        key = Key(wx.WXK_DELETE)
        window.on_key_down(key)
        assert not key.skipped
        assert window._ScrubberFrame__frame_boxes[0] == [original[2]]
        assert window._feedback_store.counts() == {'positive': 0, 'negative': 0}
        assert len(window._undo_history[0]) == 1
        window.undo_action()
        assert window._ScrubberFrame__frame_boxes[0] == original
    finally:
        window.Destroy()
        wx.Yield()


@pytest.mark.gui
def test_n_shortcut_marks_selected_automatic_areas_not_pin():
    app, window, _image, boxes = make_window()
    try:
        panel = window._ScrubberFrame__image_panel
        boxes[1].source = 'user'
        panel._set_selection(boxes)
        panel.SetFocus()
        wx.Yield()
        key = Key(ord('n'))
        window.on_key_down(key)
        assert not key.skipped
        assert [box.review_state for box in boxes] == [
            'rejected', 'unconfirmed', 'rejected']
        assert window._feedback_store.counts()['negative'] == 2
        assert len(window._undo_history[0]) == 1
        window.undo_action()
        assert all(box.review_state == 'unconfirmed' for box in boxes)
        panel._set_selection([boxes[1]])
        manual_only = Key(ord('N'))
        window.on_key_down(manual_only)
        assert manual_only.skipped and boxes[1].review_state == 'unconfirmed'
    finally:
        window.Destroy()
        wx.Yield()


@pytest.mark.gui
def test_shortcuts_leave_focused_pin_name_field_alone():
    app, window, _image, boxes = make_window()
    try:
        panel = window._ScrubberFrame__image_panel
        boxes[0].pin_id = 'pin-first'
        panel._set_selection([boxes[0]])
        editor = window._ScrubberFrame__tag_panel.find_panel_for_box(boxes[0])
        text = editor._BoxTagPanelEdit__box_tags[0]._BoxTagLabelRow__text_entry
        text.SetFocus()
        wx.Yield()
        assert wx.Window.FindFocus() is text
        for code in (wx.WXK_DELETE, ord('X'), ord('N')):
            key = Key(code)
            window.on_key_down(key)
            assert key.skipped
        assert window._ScrubberFrame__frame_boxes[0] == boxes
        assert window._feedback_store.counts() == {'positive': 0, 'negative': 0}

        cut = Key(ord('X'), ctrl=True)
        window.on_key_down(cut)
        assert cut.skipped
    finally:
        window.Destroy()
        wx.Yield()


@pytest.mark.gui
def test_x_shortcut_rejects_only_the_selected_pin_identity():
    app, window, _image, boxes = make_window()
    try:
        panel = window._ScrubberFrame__image_panel
        box = boxes[0]
        box.pin_id = 'pin-first'
        box.match_confidence = 0.72
        panel._set_selection([box])
        panel.SetFocus()
        wx.Yield()

        key = Key(ord('x'))
        window.on_key_down(key)
        assert not key.skipped
        assert box.review_state == 'unconfirmed'
        assert box.pin_id is None and box.tags == []
        assert [review.pin_id for review in box.identity_rejections] == ['pin-first']
        assert window._feedback_store.counts()['negative'] == 0
        assert boxes[1].review_state == 'unconfirmed'
        window.undo_action()
        assert box.pin_id == 'pin-first' and box.tags == ['Pin 1']
        assert box.identity_rejections == ()

        panel._set_selection([boxes[1], boxes[2]])
        multiple = Key(ord('X'))
        window.on_key_down(multiple)
        assert multiple.skipped
        assert all(item.review_state == 'unconfirmed' for item in boxes)
    finally:
        window.Destroy()
        wx.Yield()

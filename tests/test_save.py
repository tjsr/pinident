import json
from unittest.mock import patch

import numpy as np
import pytest
import wx

from boxdata import BoxData
from frame_presence import FramePresence
from videoscrubber import VideoScrubber


def choose_save(window):
    item_id = window.GetMenuBar().FindMenuItem('File', 'Save')
    assert item_id != wx.NOT_FOUND
    window.GetEventHandler().ProcessEvent(wx.CommandEvent(wx.EVT_MENU.typeId, item_id))


@pytest.mark.gui
def test_save_menu_writes_current_annotations_without_closing(tmp_path):
    app = wx.App.Get() or wx.App(False)
    window = VideoScrubber(None, 'Save annotations', '',
                           image_array=[np.zeros((80, 100, 3), dtype=np.uint8)],
                           load_catalog=False,
                           feedback_path=tmp_path / 'feedback.sqlite3')
    annotations = tmp_path / 'video.json'
    try:
        window.box_data_filename = str(annotations)
        window.Show()
        wx.Yield()
        box = BoxData((10, 15, 30, 30), ['Pin'], 'user',
                      review_state='confirmed', identity_confirmed=True)
        hidden = BoxData((50, 15, 20, 20), [], 'automatic',
                         review_state='rejected')
        window._ScrubberFrame__frame_boxes[0] = [box, hidden]
        window._frame_presence[0] = FramePresence(True, True)
        choose_save(window)
        saved = json.loads(annotations.read_text(encoding='utf-8'))
        assert [item['coords'] for item in saved['0']] == [
            [10, 15, 30, 30], [50, 15, 20, 20]]
        assert saved['0'][1]['review_state'] == 'rejected'
        assert saved['_frame_presence']['0'] == {
            'contains_pin': True, 'contains_set': True}
        assert window.IsShown()
        assert 'Saved 2 areas' in window.GetStatusBar().GetStatusText(1)

        box.tags = ['Updated Pin']
        choose_save(window)
        assert json.loads(annotations.read_text(encoding='utf-8'))['0'][0]['tags'] == [
            'Updated Pin']
    finally:
        window.Destroy()
        wx.Yield()


@pytest.mark.gui
def test_failed_save_keeps_previous_annotation_file(tmp_path):
    app = wx.App.Get() or wx.App(False)
    window = VideoScrubber(None, 'Failed save', '',
                           image_array=[np.zeros((80, 100, 3), dtype=np.uint8)],
                           load_catalog=False,
                           feedback_path=tmp_path / 'feedback.sqlite3')
    annotations = tmp_path / 'video.json'
    annotations.write_text('{"previous": true}', encoding='utf-8')
    try:
        window.box_data_filename = str(annotations)
        def fail_after_partial_write(stream, *_args):
            stream.write('{"broken":')
            raise OSError('disk full')

        with patch('scrubberframe.save_boxes_to_stream',
                   side_effect=fail_after_partial_write):
            choose_save(window)
        assert annotations.read_text(encoding='utf-8') == '{"previous": true}'
        assert list(tmp_path.glob('.pinident-*.tmp')) == []
        assert window.GetStatusBar().GetStatusText(1) == 'Save failed; see console log'
    finally:
        window.Destroy()
        wx.Yield()


@pytest.mark.gui
def test_save_commits_text_still_waiting_for_label_debounce(tmp_path):
    app = wx.App.Get() or wx.App(False)
    window = VideoScrubber(None, 'Save pending label', '',
                           image_array=[np.zeros((80, 100, 3), dtype=np.uint8)],
                           load_catalog=False,
                           feedback_path=tmp_path / 'feedback.sqlite3')
    annotations = tmp_path / 'video.json'
    try:
        window.box_data_filename = str(annotations)
        window.Show()
        wx.Yield()
        box = BoxData((10, 15, 30, 30), ['Old'], 'user')
        window._ScrubberFrame__frame_boxes[0] = [box]
        window.display_image(reconcile=False)
        editor = window._ScrubberFrame__tag_panel.find_panel_for_box(box)
        row = editor.get_or_create_label(0)
        entry = row._BoxTagLabelRow__text_entry
        entry.SetValue('Just typed')
        assert row.timer.IsRunning()
        assert box.tags == ['Old']

        choose_save(window)
        assert box.tags == ['Just typed']
        assert json.loads(annotations.read_text(encoding='utf-8'))['0'][0]['tags'] == [
            'Just typed']
        assert not row.timer.IsRunning()
    finally:
        window.Destroy()
        wx.Yield()


@pytest.mark.gui
def test_save_does_not_turn_unchanged_manual_name_into_catalog_identity(tmp_path):
    app = wx.App.Get() or wx.App(False)
    window = VideoScrubber(None, 'Save unchanged name', '',
                           image_array=[np.zeros((80, 100, 3), dtype=np.uint8)],
                           load_catalog=False,
                           feedback_path=tmp_path / 'feedback.sqlite3')
    annotations = tmp_path / 'video.json'
    try:
        window.box_data_filename = str(annotations)
        window.Show()
        wx.Yield()
        box = BoxData((10, 15, 30, 30), ['Mario'], 'user')
        window._ScrubberFrame__frame_boxes[0] = [box]
        tag_panel = window._ScrubberFrame__tag_panel
        tag_panel.set_catalog_options([('pinpanion:7', 'Mario')])
        window.display_image(reconcile=False)

        choose_save(window)
        assert box.pin_id is None and box.tags == ['Mario']
        saved = json.loads(annotations.read_text(encoding='utf-8'))['0'][0]
        assert 'pin_id' not in saved
    finally:
        window.Destroy()
        wx.Yield()

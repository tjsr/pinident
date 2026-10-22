import json
from pathlib import Path
from threading import Event
from time import monotonic, sleep
from unittest.mock import patch

import numpy as np
import pytest
import wx

from boxdata import BoxData
from detection_feedback import FeedbackStore
from shutdown_progress import format_duration, remaining_seconds
from videoscrubber import VideoScrubber


def test_remaining_time_uses_progress_and_keeps_revising_stalled_guess():
    assert remaining_seconds(10, 2, 0) == 8
    assert remaining_seconds(10, 30, 0) == 15
    assert remaining_seconds(10, 12, 50) == 12
    assert remaining_seconds(10, 12, 100) == 0
    assert format_duration(75) == '~1m 15s'


@pytest.mark.gui
def test_close_shows_steps_while_worker_finishes_then_writes_annotations(tmp_path):
    app = wx.App.Get() or wx.App(False)
    release = Event()
    window = VideoScrubber(None, 'Shutdown progress', '',
                           image_array=[np.zeros((80, 100, 3), dtype=np.uint8)],
                           load_catalog=False,
                           feedback_path=tmp_path / 'feedback.sqlite3')
    annotations = tmp_path / 'video.json'
    window.box_data_filename = str(annotations)
    window._ScrubberFrame__frame_boxes[0] = [BoxData((1, 2, 10, 12), ['Pin'], 'user')]
    window.Show()
    wx.Yield()
    window.display_image(reconcile=False)
    editor = window._ScrubberFrame__tag_panel.find_panel_for_box(
        window._ScrubberFrame__frame_boxes[0][0])
    editor.get_or_create_label(0)._BoxTagLabelRow__text_entry.SetValue('Edited pin')
    window._play_executor.submit(release.wait, 2)
    closed = False
    try:
        window.Close()
        wx.Yield()
        dialog = window._shutdown_dialog
        assert dialog is not None and dialog.IsShown()
        assert not annotations.exists()
        assert any('Finish background work' in row.GetLabel() and '%' in row.GetLabel()
                   for row in dialog._rows)
        assert 'Overall:' in dialog._overall.GetLabel()
        release.set()
        deadline = monotonic() + 5
        while window._shutdown_dialog is not None and monotonic() < deadline:
            # wx.Timer needs MainLoop; the test suite pumps wx without one.
            window._shutdown_tick()
            wx.Yield()
            sleep(0.02)
        assert window._shutdown_dialog is None, (
            window._shutdown_stage,
            window._shutdown_play_barrier.done() if window._shutdown_play_barrier else None,
            window._shutdown_cloud_barrier.done() if window._shutdown_cloud_barrier else None,
            window._shutdown_save_future.done() if window._shutdown_save_future else None)
        closed = True
        assert json.loads(annotations.read_text(encoding='utf-8'))['0'][0]['coords'] == [
            1, 2, 10, 12]
        assert json.loads(annotations.read_text(encoding='utf-8'))['0'][0]['tags'] == [
            'Edited pin']
    finally:
        release.set()
        if not closed:
            if window._shutdown_dialog is not None:
                window._shutdown_dialog.Destroy()
            window.Destroy()


@pytest.mark.gui
def test_failed_close_save_keeps_editor_open(tmp_path):
    app = wx.App.Get() or wx.App(False)
    window = VideoScrubber(None, 'Failed close save', '',
                           image_array=[np.zeros((40, 40, 3), dtype=np.uint8)],
                           load_catalog=False,
                           feedback_path=tmp_path / 'feedback.sqlite3')
    window.box_data_filename = str(tmp_path / 'video.json')
    window.Show()
    wx.Yield()
    try:
        with (patch('scrubberframe.save_boxes_to_file',
                    side_effect=OSError('disk full')),
              patch('scrubberframe.wx.MessageBox')):
            window.Close()
            deadline = monotonic() + 3
            while window._closing and monotonic() < deadline:
                window._shutdown_tick()
                wx.Yield()
                sleep(0.01)
        assert not window._closing
        assert window._shutdown_dialog is None
        assert window.IsShown()
    finally:
        window.Destroy()


@pytest.mark.gui
def test_close_flushes_pending_classifier_feedback_and_annotations(
        tmp_path: Path) -> None:
    app = wx.App.Get() or wx.App(False)
    feedback_path = tmp_path / 'feedback.sqlite3'
    annotations = tmp_path / 'video.json'
    window = VideoScrubber(None, 'Close reviewed area', '',
                           image_array=[np.zeros((60, 60, 3), dtype=np.uint8)],
                           load_catalog=False, feedback_path=feedback_path)
    window.box_data_filename = str(annotations)
    window.Show()
    wx.Yield()
    box = BoxData((5, 5, 30, 30), [], 'automatic')
    window._ScrubberFrame__frame_boxes[0] = [box]
    window.display_image(reconcile=False, catalog_matching=False)
    with window.classifier_review_action():
        window.confirm_box(box)
    assert window._feedback_store.counts()['positive'] == 0
    closed = False
    try:
        window.Close()
        deadline = monotonic() + 5
        while window._shutdown_dialog is not None and monotonic() < deadline:
            window._shutdown_tick()
            wx.Yield()
            sleep(0.01)
        assert window._shutdown_dialog is None
        closed = True
        saved = json.loads(annotations.read_text(encoding='utf-8'))
        assert saved['0'][0]['review_state'] == 'confirmed'
        store = FeedbackStore(feedback_path)
        try:
            assert store.counts()['positive'] == 1
        finally:
            store.close()
    finally:
        if not closed:
            if window._shutdown_dialog is not None:
                window._shutdown_dialog.Destroy()
            window.Destroy()

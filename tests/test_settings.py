from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import wx

from settings_dialog import SettingsDialog, storage_location
from videoscrubber import VideoScrubber


def test_storage_location_reports_disk_and_session_states(tmp_path):
    target = tmp_path / 'reviews.sqlite3'
    assert storage_location(target) == (str(target.resolve()), 'File not created yet')
    target.write_bytes(b'reviews')
    location, status = storage_location(target)
    assert location == str(target.resolve())
    assert status.startswith('Last modified: ')
    assert storage_location(':memory:') == (
        'In memory (this session only)', 'No file on disk')
    assert storage_location(None) == ('No annotation file selected', 'Not saved yet')


@pytest.mark.gui
def test_settings_menu_passes_active_data_paths_to_dialog(tmp_path):
    app = wx.App.Get() or wx.App(False)
    feedback_path = tmp_path / 'feedback.sqlite3'
    catalog_path = tmp_path / 'catalog.sqlite3'
    window = VideoScrubber(None, 'Settings test', '',
                           image_array=[np.zeros((20, 20, 3), dtype=np.uint8)],
                           feedback_path=feedback_path, catalog_path=str(catalog_path),
                           load_catalog=False)
    try:
        window.box_data_filename = str(tmp_path / 'video.json')
        item_id = window.GetMenuBar().FindMenuItem('File', 'Settings')
        assert item_id != wx.NOT_FOUND
        with patch('scrubberframe.SettingsDialog') as dialog:
            window.GetEventHandler().ProcessEvent(wx.CommandEvent(wx.EVT_MENU.typeId, item_id))
            dialog.assert_called_once()
            kwargs = dialog.call_args.kwargs
            assert Path(kwargs['feedback_path']) == feedback_path
            assert Path(kwargs['catalog_path']) == catalog_path
            assert Path(kwargs['annotations_path']) == tmp_path / 'video.json'
            assert Path(kwargs['model_path']) == tmp_path / 'feedback.svm.yml'
            assert kwargs['model_active'] is False
            assert kwargs['feedback_counts'] == {'positive': 0, 'negative': 0}
            dialog.return_value.ShowModal.assert_called_once()
            dialog.return_value.Destroy.assert_called_once()
    finally:
        window.Destroy()
        wx.Yield()


@pytest.mark.gui
def test_settings_dialog_shows_copyable_paths_and_feedback_status(tmp_path):
    app = wx.App.Get() or wx.App(False)
    parent = wx.Frame(None)
    feedback_path = tmp_path / 'feedback.sqlite3'
    feedback_path.touch()
    dialog = SettingsDialog(
        parent, feedback_path=feedback_path,
        catalog_path=tmp_path / 'catalog.sqlite3',
        annotations_path=tmp_path / 'video.json',
        feedback_counts={'positive': 2, 'negative': 1},
        wrong_identity_count=3, model_revision=4,
        model_path=tmp_path / 'feedback.svm.yml', model_active=True,
        model_current=True)
    try:
        assert dialog.GetTitle() == 'Settings'
        assert dialog.GetSize().height >= dialog.GetBestSize().height
        assert dialog.locations['feedback'].GetValue() == str(feedback_path.resolve())
        assert dialog.locations['catalog'].GetValue() == str(
            (tmp_path / 'catalog.sqlite3').resolve())
        assert dialog.locations['annotations'].GetValue() == str(
            (tmp_path / 'video.json').resolve())
        assert dialog.locations['model'].GetValue() == str(
            (tmp_path / 'feedback.svm.yml').resolve())
        assert dialog.locations['model_metadata'].GetValue() == str(
            (tmp_path / 'feedback.svm.yml.json').resolve())
        assert dialog.statuses['model'].GetLabel().startswith('File not created yet')
        assert dialog.locations['feedback'].GetWindowStyle() & wx.TE_READONLY
        assert dialog.statuses['feedback'].GetLabel().startswith('Last modified: ')
        assert dialog.statuses['catalog'].GetLabel().startswith('File not created yet')
        labels = [child.GetLabel() for child in dialog.GetChildren()
                  if isinstance(child, wx.StaticText)]
        assert any('Saved pin crops: 2; non-pin crops: 1; wrong-identity crops: 3'
                   in label for label in labels)
        assert any('OpenCV classifier: active, training data current.' in
                   ' '.join(label.split())
                   for label in labels)
    finally:
        dialog.Destroy()
        parent.Destroy()
        wx.Yield()

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import wx

from videoscrubber import VideoScrubber


@pytest.mark.gui
def test_file_and_training_menus_replace_buttons_and_load_annotations(
        tmp_path: Path) -> None:
    app = wx.App.Get() or wx.App(False)
    window = VideoScrubber(None, 'Menus', '',
                           image_array=[np.zeros((80, 100, 3), dtype=np.uint8)],
                           load_catalog=False,
                           feedback_path=tmp_path / 'feedback.sqlite3')
    try:
        menu_bar = window.GetMenuBar()
        assert [menu_bar.GetMenuLabel(index) for index in range(menu_bar.GetMenuCount())] == [
            'File', 'Training']
        assert [item.GetItemLabelText() for item in menu_bar.GetMenu(0).GetMenuItems()] == [
            'Save', 'Settings', 'Load Annotations...']
        assert [('<separator>' if item.IsSeparator() else item.GetItemLabelText())
                for item in menu_bar.GetMenu(1).GetMenuItems()] == [
            'Review saved areas...', '<separator>', 'Feedback model...',
            'Add to training data', 'Run training now']
        assert {child.GetLabel() for child in window._ScrubberFrame__button_panel.GetChildren()
                if isinstance(child, wx.Button)}.isdisjoint(
                    {'Save', 'Settings', 'Load Annotations...'})
        assert {child.GetLabel() for child in window._ScrubberFrame__tag_panel.GetChildren()
                if isinstance(child, wx.Button)}.isdisjoint(
                    {'Feedback model...', 'Add to training data', 'Run training now'})

        window._set_training_progress(25, 'Indexing', True)
        assert not window._training_add_item.IsEnabled()
        assert not window._training_run_item.IsEnabled()
        window._set_training_progress(100, 'Done', False)
        assert window._training_add_item.IsEnabled()
        assert window._training_run_item.IsEnabled()

        path = tmp_path / 'annotations.json'
        path.write_text(json.dumps({'0': [{'coords': [10, 15, 30, 30],
                                           'tags': ['Loaded'], 'source': 'user'}]}),
                        encoding='utf-8')
        with patch('scrubberframe.wx.FileDialog') as dialog:
            selected = dialog.return_value.__enter__.return_value
            selected.ShowModal.return_value = wx.ID_OK
            selected.GetPath.return_value = str(path)
            item_id = menu_bar.FindMenuItem('File', 'Load Annotations...')
            window.GetEventHandler().ProcessEvent(wx.CommandEvent(wx.EVT_MENU.typeId, item_id))
        assert window.box_data_filename == str(path)
        assert window._ScrubberFrame__frame_boxes[0][0].tags == ['Loaded']
        with patch('scrubberframe.ClassifierReviewDialog') as review_dialog:
            item_id = menu_bar.FindMenuItem('Training', 'Review saved areas...')
            window.GetEventHandler().ProcessEvent(
                wx.CommandEvent(wx.EVT_MENU.typeId, item_id))
            review_dialog.assert_called_once()
            assert review_dialog.call_args.args[1][0].box is (
                window._ScrubberFrame__frame_boxes[0][0])
            review_dialog.return_value.ShowModal.assert_called_once()
            review_dialog.return_value.Destroy.assert_called_once()
    finally:
        window.Destroy()
        wx.Yield()

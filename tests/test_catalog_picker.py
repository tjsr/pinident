import numpy as np
import pytest
import wx
from unittest.mock import patch

from boxdata import BoxData
from controls.BoxTagEditPanel import BoxTagPanelEdit
from videoscrubber import VideoScrubber


@pytest.mark.gui
def test_typing_east_waits_for_debounce_and_keeps_the_whole_query():
    app = wx.App.Get() or wx.App(False)
    frame = wx.Frame(None, size=(500, 200))
    box = BoxData((1, 1, 20, 20), [''], 'user')
    panel = BoxTagPanelEdit(frame, box, catalog_options=[
        ('pin:1', 'South East Pin'),
        ('pin:2', 'Easter Pin'),
        ('pin:3', 'West Pin'),
        ('pin:4', 'A Star'),
    ])
    layout = wx.BoxSizer(wx.VERTICAL)
    layout.Add(panel, 1, wx.EXPAND)
    frame.SetSizer(layout)
    try:
        frame.Show()
        frame.Layout()
        wx.Yield()
        row = panel.get_or_create_label(0)
        picker = row._BoxTagLabelRow__text_entry
        picker.SetFocus()
        wx.Yield()
        with patch.object(row, '_show_matches', wraps=row._show_matches) as search:
            for letter, expected in zip('East', ('E', 'Ea', 'Eas', 'East')):
                picker.WriteText(letter)
                wx.Yield()
                assert picker.GetValue() == expected
            assert search.call_count == 0
            assert row._search_timer.IsRunning()

            row._on_search_timer(None)
            assert search.call_count == 1
            assert row.visible_options == ['South East Pin', 'Easter Pin']
            assert picker.GetValue() == 'East'

            picker.WriteText('e')
            wx.Yield()
            assert picker.GetValue() == 'Easte'
            row._on_search_timer(None)
            assert row.visible_options == ['Easter Pin']
    finally:
        frame.Destroy()
        wx.Yield()


@pytest.mark.gui
def test_catalog_picker_filters_by_name_and_id_without_losing_selection():
    app = wx.App.Get() or wx.App(False)
    frame = wx.Frame(None)
    box = BoxData((1, 1, 20, 20), [''], 'user')
    panel = BoxTagPanelEdit(frame, box, catalog_options=[
        ('pinpanion:7', 'Mario'),
        ('pinpanion:8', 'Luigi'),
        ('pinpanion:9', 'Mario Anniversary'),
        ('pinpanion:10', 'Collector'),
        ('pinpanion:11', 'Collector'),
    ])
    try:
        frame.Show()
        wx.Yield()
        row = panel.get_or_create_label(0)
        picker = next(child for child in row.GetChildren()
                      if isinstance(child, wx.TextCtrl))

        picker.SetFocus()
        wx.Yield()
        picker.SetValue('ANNIVERSARY')
        row._on_search_timer(None)
        assert row.visible_options == ['Mario Anniversary']
        assert picker.GetValue() == 'ANNIVERSARY'
        row._results.SetSelection(0)
        choice_event = wx.CommandEvent(wx.EVT_LISTBOX.typeId, row._results.GetId())
        choice_event.SetEventObject(row._results)
        row._results.ProcessEvent(choice_event)
        wx.Yield()
        assert box.tags == ['Mario Anniversary']
        assert box.pin_id == 'pinpanion:9'
        row._open_catalog(wx.CommandEvent())
        assert row.visible_options == ['Mario Anniversary']

        picker.SetValue('PINPANION:8')
        row._on_search_timer(None)
        assert row.visible_options == ['Luigi']
        picker.SetValue('pinpanion:11')
        row._on_search_timer(None)
        assert row.visible_options == ['Collector [pinpanion:11]']
        picker.SetValue('no matching pin')
        row._on_search_timer(None)
        assert row.visible_options == []
        assert picker.GetValue() == 'no matching pin'
    finally:
        panel.get_or_create_label(0).timer.Stop()
        frame.Destroy()
        wx.Yield()


@pytest.mark.gui
def test_catalog_picker_uses_autopopulated_label_as_initial_filter():
    app = wx.App.Get() or wx.App(False)
    frame = wx.Frame(None, size=(500, 240))
    box = BoxData((1, 1, 20, 20), ['East'], 'automatic')
    panel = BoxTagPanelEdit(frame, box)
    frame.SetSizer(wx.BoxSizer(wx.VERTICAL))
    frame.GetSizer().Add(panel, 1, wx.EXPAND)
    try:
        frame.Show()
        frame.Layout()
        wx.Yield()
        panel.set_catalog_options([
            ('pin:1', 'South East Pin'),
            ('pin:2', 'Easter Pin'),
            ('pin:3', 'West Pin'),
        ])
        row = panel.get_or_create_label(0)
        assert row.tag == 'East'

        row._open_catalog(wx.CommandEvent())
        assert row.visible_options == ['South East Pin', 'Easter Pin']
        assert row.tag == 'East'
        assert box.tags == ['East'] and box.pin_id is None

        box.tags = ['West']
        panel.box = box
        assert row.tag == 'West'
        assert row.visible_options == ['West Pin']

        box.tags = ['']
        panel.box = box
        row._open_catalog(wx.CommandEvent())
        assert row.visible_options == ['South East Pin', 'Easter Pin', 'West Pin']
    finally:
        frame.Destroy()
        wx.Yield()


@pytest.mark.gui
def test_catalog_picker_selects_result_with_mouse_click():
    app = wx.App.Get() or wx.App(False)
    frame = wx.Frame(None, size=(500, 240))
    box = BoxData((1, 1, 20, 20), [''], 'user')
    panel = BoxTagPanelEdit(frame, box, catalog_options=[
        ('pinpanion:7', 'Mario'),
        ('pinpanion:9', 'Mario Anniversary'),
    ])
    frame.SetSizer(wx.BoxSizer(wx.VERTICAL))
    frame.GetSizer().Add(panel, 1, wx.EXPAND)
    try:
        frame.Show()
        frame.Layout()
        wx.Yield()
        row = panel.get_or_create_label(0)
        picker = row._BoxTagLabelRow__text_entry
        picker.SetFocus()
        picker.SetValue('Mario')
        wx.Yield()
        row._on_search_timer(None)
        wx.Yield()
        assert row._popup.IsShown()
        assert row.visible_options == ['Mario', 'Mario Anniversary']

        second_result_y = next(y for y in range(row._results.GetClientSize().height)
                               if row._results.HitTest(wx.Point(12, y)) == 1)
        click = wx.MouseEvent(wx.wxEVT_LEFT_DOWN)
        click.SetPosition(wx.Point(12, second_result_y))
        click.SetEventObject(row._results)
        row._results.ProcessEvent(click)
        wx.Yield()

        assert box.tags == ['Mario Anniversary']
        assert box.pin_id == 'pinpanion:9'
    finally:
        frame.Destroy()
        wx.Yield()


@pytest.mark.gui
def test_catalog_picker_fits_inside_tag_pane_at_original_window_width():
    app = wx.App.Get() or wx.App(False)
    image = np.zeros((120, 180, 3), dtype=np.uint8)
    window = VideoScrubber(None, 'Picker layout', '', image_array=[image],
                           load_catalog=False)
    long_name = ('A very long Pinny Arcade collector pin with multiple colours '
                 'and detailed artwork')
    try:
        window.SetSize((800, 600))
        window.Show()
        wx.Yield()
        tag_panel = window._ScrubberFrame__tag_panel
        tag_panel.set_catalog_options([('pinpanion:12345', long_name)])
        box = BoxData((20, 20, 40, 40), [long_name], 'user',
                      pin_id='pinpanion:12345')
        window._ScrubberFrame__frame_boxes[0] = [box]
        window.display_image()
        window.Layout()
        wx.Yield()
        editor = tag_panel.find_panel_for_box(box)
        row = editor.get_or_create_label(0)
        picker = next(child for child in row.GetChildren()
                      if isinstance(child, wx.TextCtrl))
        search_button = next(child for child in row.GetChildren()
                             if isinstance(child, wx.Button) and
                             child.GetLabel() == '\u25be')
        picker_rect = picker.GetScreenRect()
        pane_rect = tag_panel.GetScreenRect()
        assert picker_rect.width >= 270
        assert search_button.GetScreenRect().GetRight() < pane_rect.GetRight()
        assert picker.GetValue() == long_name
        row._open_catalog(wx.CommandEvent())
        assert row._popup.GetScreenRect().width > picker_rect.width
    finally:
        window.Destroy()
        wx.Yield()


@pytest.mark.gui
def test_large_catalog_stays_searchable_without_rebuilding_unchanged_rows():
    app = wx.App.Get() or wx.App(False)
    image = np.zeros((120, 180, 3), dtype=np.uint8)
    window = VideoScrubber(None, 'Large picker', '', image_array=[image],
                           load_catalog=False)
    try:
        window.Show()
        wx.Yield()
        tag_panel = window._ScrubberFrame__tag_panel
        options = [(f'pinpanion:{index}', f'Pin name {index}')
                   for index in range(2000)]
        tag_panel.set_catalog_options(options)
        box = BoxData((20, 20, 40, 40), ['Pin name 1999'], 'user',
                      pin_id='pinpanion:1999')
        tag_panel.boxes = [box]
        editor = tag_panel.find_panel_for_box(box)
        row = editor.get_or_create_label(0)
        picker = next(child for child in row.GetChildren()
                      if isinstance(child, wx.TextCtrl))
        assert picker.GetValue() == 'Pin name 1999'
        assert row._popup is None

        with patch.object(editor, 'Refresh', wraps=editor.Refresh) as refresh:
            tag_panel.boxes = [box]
            assert refresh.call_count == 0

        picker.SetValue('Pin name 1888')
        picker.SetFocus()
        row._on_search_timer(None)
        assert row.visible_options == ['Pin name 1888']
        row.select_catalog_option(0)
        wx.Yield()
        assert box.tags == ['Pin name 1888']
        assert box.pin_id == 'pinpanion:1888'
    finally:
        window.Destroy()
        wx.Yield()

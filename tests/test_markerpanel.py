import pytest
import wx

from controls.markerpanel import MarkerPanel, marker_labels


def _assert_labels_do_not_overlap(labels, width, gap=8):
    previous_right = -gap
    for _frame, _tick_x, left, text_width in labels:
        assert 0 <= left <= width - text_width
        assert left >= previous_right + gap
        previous_right = left + text_width


@pytest.mark.parametrize('width,frames', [
    (240, 10_000),
    (600, 10_000),
    (800, 100_000),
    (240, 12),
])
def test_marker_labels_fit_available_width(width, frames):
    labels = marker_labels(frames, width, lambda text: (8 * len(text), 14))

    assert labels[0][0] == 0
    assert labels[-1][0] == frames - 1
    _assert_labels_do_not_overlap(labels, width)


def test_marker_density_changes_with_width_and_single_frame_is_safe():
    measure = lambda text: (8 * len(text), 14)
    narrow = marker_labels(100_000, 240, measure)
    wide = marker_labels(100_000, 1000, measure)

    assert len(wide) > len(narrow)
    assert marker_labels(1, 100, measure) == [(0, 0, 0, 8)]
    assert marker_labels(100, 4, measure) == []


@pytest.mark.gui
def test_marker_panel_uses_actual_font_width_after_resize():
    app = wx.App.Get() or wx.App(False)
    window = wx.Frame(None, size=(800, 120))
    panel = MarkerPanel(window, 100_000)
    sizer = wx.BoxSizer(wx.VERTICAL)
    sizer.Add(panel, 1, wx.EXPAND)
    window.SetSizer(sizer)
    try:
        window.Show()
        for width in (800, 360):
            window.SetClientSize((width, 120))
            wx.Yield()
            dc = wx.ClientDC(panel)
            labels = marker_labels(panel.num_images, panel.GetClientSize().width,
                                   dc.GetTextExtent, panel.interval)
            _assert_labels_do_not_overlap(labels, panel.GetClientSize().width)
        assert app is not None
    finally:
        window.Destroy()
        wx.Yield()

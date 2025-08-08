import wx
import pytest
from boxdata import BoxData
from controls.BoxTagEditPanel import BoxTagPanelEdit
from events.events import EVT_BOX_LABEL_REMOVE

@pytest.fixture(scope="session")
def wx_app():
    app = wx.App(False)
    yield app
    app.Destroy()

@pytest.mark.gui
def test_remove_button_fires_event(wx_app):
    # Setup test frame and panel
    frame = wx.Frame(None)
    box = BoxData(coords=(1, 2, 3, 4), tags=["Tag1", "Tag2"], source='automatic')
    panel = BoxTagPanelEdit(frame, box)
    frame.Show()
    wx.Yield()  # Let wx finish layout

    # Get the first tag label row and its remove button
    label_row = panel.get_or_create_label(0)
    remove_button = getattr(label_row, "_BoxTagLabelRow__remove_button", None)
    assert remove_button is not None

    # Event capture
    events = []
    def on_remove(event):
        events.append(event)
        event.Skip()

    panel.Bind(EVT_BOX_LABEL_REMOVE, on_remove)

    # Simulate click
    remove_button.ProcessEvent(wx.CommandEvent(wx.EVT_BUTTON.typeId, remove_button.GetId()))
    wx.Yield()

    assert len(events) == 1
    assert events[0].label_index == 0
import wx
import pytest
from boxdata import BoxData
from controls.BoxTagEditPanel import BoxTagPanelEdit

@pytest.fixture(scope="session")
def wx_app():
    app = wx.App(False)
    yield app
    app.Destroy()

@pytest.mark.gui
def test_remove_button_removes_tag(wx_app):
    # Setup test frame and panel
    frame = wx.Frame(None)
    box = BoxData(coords=(1, 2, 3, 4), tags=["Tag1", "Tag2"], source='automatic')
    panel = BoxTagPanelEdit(frame, box)
    frame.Show()
    wx.Yield()  # Let wx finish layout

    # Get the first tag label row and its remove button
    label_row = panel.get_or_create_label(0)
    remove_button = next(child for child in label_row.GetChildren() if isinstance(child, wx.BitmapButton))

    # Simulate click
    click = wx.CommandEvent(wx.EVT_BUTTON.typeId, remove_button.GetId())
    click.SetEventObject(remove_button)
    remove_button.ProcessEvent(click)
    wx.Yield()

    assert box.tags == ["Tag2"]
    frame.Destroy()

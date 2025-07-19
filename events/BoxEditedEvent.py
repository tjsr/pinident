import wx

from boxdata import BoxData
from events.events import wxEVT_BOX_EDITED


class BoxEditedEvent(wx.CommandEvent):
    def __init__(self, source: wx.Panel, box: BoxData):
        super().__init__(wxEVT_BOX_EDITED, source.GetId())
        self.box = box

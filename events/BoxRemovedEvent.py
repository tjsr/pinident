import wx

from boxdata import BoxData
from events.events import wxEVT_BOX_REMOVED


class BoxRemovedEvent(wx.CommandEvent):
	def __init__(self, source: wx.Panel, box: BoxData):
		super().__init__(wxEVT_BOX_REMOVED, source.GetId())
		self.SetEventObject(source)
		self.box = box

	def Clone(self) -> "BoxRemovedEvent":
		# wxPython uses this to copy events internally
		return BoxRemovedEvent(self.GetEventObject(), self.box) # type: ignore[arg-type]

# class BoxRemovedEvent1(wx.CommandEvent):
#     """Custom event for box removal."""
#     def __init__(self, source: wx.Window, box: BoxData):
#         super().__init__(wxEVT_BOX_REMOVED, source.GetId())
#         self.box = box

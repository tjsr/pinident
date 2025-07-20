import wx

from boxdata import BoxData
from events.events import wxEVT_BOX_ADDED


class BoxAddedEvent(wx.CommandEvent):
	box: BoxData

	"""Custom event for box addition."""
	def __init__(self, source: wx.Window, box: BoxData):
		super().__init__(wxEVT_BOX_ADDED, source.GetId())
		self.SetEventObject(source)
		self.box = box

	def Clone(self) -> "BoxAddedEvent":
		# wxPython uses this to copy events internally
		return BoxAddedEvent(self.GetEventObject(), self.box) # type: ignore[arg-type]
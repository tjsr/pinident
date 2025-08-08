import wx

from boxdata import BoxData
from events.events import wxEVT_BOX_EDITED


class BoxEditedEvent(wx.PyCommandEvent):
	def __init__(self, source, box: BoxData):
		super().__init__(wxEVT_BOX_EDITED, source.GetId())
		self.SetEventObject(source)
		self.box = box

	def Clone(self) -> "BoxEditedEvent":
		# wxPython uses this to copy events internally
		return BoxEditedEvent(self.GetEventObject(), self.box) # type: ignore[arg-type]
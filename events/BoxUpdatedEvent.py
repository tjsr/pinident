import wx

from boxdata import BoxData
from events.events import wxEVT_BOX_UPDATED


class BoxUpdatedEvent(wx.CommandEvent):
	boxes: list[BoxData]

	def __init__(self, source: wx.Panel, boxes: list[BoxData]):
		super().__init__(wxEVT_BOX_UPDATED, source.GetId())
		self.SetEventObject(source)
		self.boxes = boxes

	def Clone(self) -> "BoxUpdatedEvent":
		# wxPython uses this to copy events internally
		return BoxUpdatedEvent(self.GetEventObject(), self.boxes) # type: ignore[arg-type]

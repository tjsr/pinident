import wx

from boxdata import BoxData
from events.events import wxEVT_BOX_SELECTED


class BoxSelectedEvent(wx.CommandEvent):
	__box: BoxData

	def __init__(self, source: wx.Panel, box: BoxData):
		super().__init__(wxEVT_BOX_SELECTED, source.GetId())
		self.SetEventObject(source)
		self.__box = box

	def Clone(self) -> "BoxSelectedEvent":
		# wxPython uses this to copy events internally
		return BoxSelectedEvent(self.GetEventObject(), self.__box) # type: ignore[arg-type]

	@property
	def box(self) -> BoxData:
		"""Get the box associated with this event."""
		return self.__box

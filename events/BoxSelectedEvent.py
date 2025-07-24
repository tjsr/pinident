import wx

from boxdata import BoxData
from events.events import wxEVT_BOX_SELECTED


class BoxSelectedEvent(wx.CommandEvent):
	__box: BoxData | None

	def __init__(self, source: wx.Panel, box: BoxData | None):
		super().__init__(wxEVT_BOX_SELECTED, source.GetId())
		self.SetEventObject(source)
		self.__box = box

	def Clone(self) -> "BoxSelectedEvent":
		# wxPython uses this to copy events internally
		return BoxSelectedEvent(self.GetEventObject(), self.__box) # type: ignore[arg-type]

	@property
	def box(self) -> BoxData | None:
		"""Get the box associated with this event."""
		return self.__box

class BoxDeselectedEvent(BoxSelectedEvent):
	"""Event for deselecting a box."""
	def __init__(self, source: wx.Panel):
		super().__init__(source, None)  # No box associated with deselection
		self.SetEventType(wxEVT_BOX_SELECTED)  # Use the same event type for consistency
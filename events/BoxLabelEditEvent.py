import wx

from boxdata import TagLabel
from events.events import wxEVT_BOX_LABEL_EDITED


class BoxLabelEditedEvent(wx.CommandEvent):
	__new_label: TagLabel
	__label_index: int

	"""Custom event for box label updates."""
	def __init__(self, source: wx.Panel, label_index: int, new_label: TagLabel | None = None):
		super().__init__(wxEVT_BOX_LABEL_EDITED, source.GetId())
		self.SetEventObject(source)
		self.__new_label = new_label
		self._label_index = label_index

	@property
	def new_label(self) -> TagLabel:
		return self.__new_label

	@property
	def label_index(self) -> int:
		return self._label_index

	def Clone(self) -> "BoxLabelEditedEvent":
		# wxPython uses this to copy events internally
		return BoxLabelEditedEvent(self.GetEventObject(), self.label_index, self.new_label) # type: ignore[arg-type]

class BoxLabelRemovedEvent(BoxLabelEditedEvent):
	__new_label: TagLabel | None

	"""Custom event for box label removal."""
	def __init__(self, source: wx.Panel, label_index: int):
		super().__init__(source, label_index)
		self.SetEventType(wxEVT_BOX_LABEL_EDITED)  # Use the same event type for removal
		self._label_index = label_index
		self.__new_label = None  # Empty label to indicate removal

class BoxLabelRemoveEvent(BoxLabelEditedEvent):
	__new_label: TagLabel | None

	"""Custom event for box label removal."""
	def __init__(self, source: wx.Panel, label_index: int):
		super().__init__(source, label_index)
		self.SetEventType(wxEVT_BOX_LABEL_EDITED)  # Use the same event type for removal
		self._label_index = label_index
		self.__new_label = None  # Empty label to indicate removal
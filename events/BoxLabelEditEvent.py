import wx

from boxdata import TagLabel
from events.events import wxEVT_BOX_LABEL_EDITED


class BoxLabelEditedEvent(wx.PyCommandEvent):
	__new_label: TagLabel
	__label_index: int

	"""Custom event for box label updates."""
	def __init__(self, source, label_index: int, new_label: TagLabel | None = None,
	             pin_id: str | None = None):
		super().__init__(wxEVT_BOX_LABEL_EDITED, source.GetId())
		self.SetEventObject(source)
		self.__new_label = new_label
		self._label_index = label_index
		self._pin_id = pin_id

	@property
	def new_label(self) -> TagLabel:
		return self.__new_label

	@property
	def label_index(self) -> int:
		return self._label_index

	@property
	def pin_id(self) -> str | None:
		return self._pin_id

	def Clone(self) -> "BoxLabelEditedEvent":
		# wxPython uses this to copy events internally
		return BoxLabelEditedEvent(self.GetEventObject(), self.label_index,
		                           self.new_label, self.pin_id) # type: ignore[arg-type]

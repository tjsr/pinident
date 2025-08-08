import wx

from events.events import wxEVT_BOX_LABEL_REMOVE

class BoxLabelRemoveEvent(wx.PyCommandEvent):
	__label_index: int

	def __init__(self, source, label_index: int):
		super().__init__(wxEVT_BOX_LABEL_REMOVE, source.GetId())
		self.SetEventObject(source)
		self.__label_index = label_index

	@property
	def label_index(self) -> int:
		return self.__label_index

import wx

from boxdata import TagLabel
from events.events import wxEVT_BOX_LABEL_EDITED


class BoxLabelEditedEvent(wx.CommandEvent):
    __new_label: TagLabel
    __label_index: int

    """Custom event for box label updates."""
    def __init__(self, source: wx.Panel, label_index: int, new_label: TagLabel):
        super().__init__(wxEVT_BOX_LABEL_EDITED, source.GetId())
        self.__new_label = new_label
        self._label_index = label_index

    @property
    def new_label(self) -> TagLabel:
        return self.__new_label

    @property
    def label_index(self) -> int:
        return self._label_index

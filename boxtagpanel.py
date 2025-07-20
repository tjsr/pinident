from typing import List

import wx

from boxdata import BoxData
from events.BoxEditedEvent import BoxEditedEvent
from events.BoxLabelEditEvent import BoxLabelEditedEvent
from events.events import EVT_BOX_LABEL_EDITED


class BoxTagLabelRow(wx.Panel):
    __combo_box: wx.ComboBox
    __text_entry: wx.TextCtrl
    __rem_button: wx.BitmapButton
    __is_only_row: bool = False
    __tag_index: int

    def __init__(
        self,
        parent: wx.Panel,
        box: BoxData,
        tag_index: int,
        choices: List[str]
    ):
        super().__init__(parent)
        label = box.GetTag(tag_index)
        # self.__combo_box = wx.ComboBox(self, value=label, choices=choices, style=wx.CB_DROPDOWN)
        self.__text_entry = wx.TextCtrl(self, value=label, style=wx.TE_PROCESS_ENTER)
        self.__rem_button = wx.BitmapButton(self, bitmap=wx.ArtProvider.GetBitmap(wx.ART_MINUS, wx.ART_BUTTON))

        sizer = wx.BoxSizer(wx.HORIZONTAL)
        sizer.Add(self.__combo_box, 1, wx.EXPAND | wx.ALL, 5)
        sizer.Add(self.__rem_button, 0, wx.ALL, 5)

        # self.__combo_box.Bind(wx.EVT_TEXT, self.label_updated)
        self.__text_entry.Bind(wx.EVT_TEXT, self.label_edited)

        self.SetSizer(sizer)

    @property
    def tag(self) -> str:
        """Get the label of the combo box."""
        return self.__text_entry.GetValue()

    def repaint(self) -> None:
        if self.__is_only_row:
            self.__rem_button.Hide()
        else:
            self.__rem_button.Show()

    def set_only_row(self, is_only_row: bool) -> None:
        """Set whether this is the only row in the set."""
        self.__is_only_row = is_only_row
        self.repaint()

    def set_tag(self, tag: str) -> None:
        """Set the label of the combo box."""
        self.__combo_box.SetValue(tag)

    def label_edited(self, event: wx.Event) -> None:
        """Handle label updates."""
        # This method can be overridden to handle label changes
        # For example, you might want to notify the parent panel
        # that the label has changed.
        value = self.__text_entry.GetLineText(0)
        updateEvent = BoxLabelEditedEvent(self, self.__tag_index, value)
        wx.PostEvent(self, updateEvent)


class BoxTagPanelEdit(wx.Panel):
    __box: BoxData
    __box_tags: List[BoxTagLabelRow]
    __heading_text: wx.StaticText
    __del_button: wx.BitmapButton
    __choices: List[str] = []

    def is_box(self, box: BoxData) -> bool:
        """Check if the given box matches the current box."""
        return self.__box == box or self.__box.coords == box.coords

    def __init__(
        self,
        parent: wx.Panel,
        box: BoxData
    ):
        super().__init__(parent)
        self.__box = box
        self.__heading_text = wx.StaticText(self, label=f"Box: {box.coords}")

    def repaint_box(self) -> None:
        tag_index: int = 0
        tag_count: int = len(self.__box.tags)

        self.__heading_text.SetLabelText(f"Box: {self.__box.coords}")

        while tag_index < tag_count:
            self.repaint_tags(tag_index)
            tag_index += 1

        # Remove any extra labels
        while len(self.__box_tags) > tag_index:
            self.__box_tags.pop()

    def repaint_tags(self, tag_index: int) -> None:
        tag = self.__box.tags[tag_index]
        if not isinstance(tag, str):
            raise ValueError(f"Invalid tag type: {type(tag)}. Expected str.")

        # box_label_count: int = len(self.__box_labels)
        current_label_row: BoxTagLabelRow = self.get_or_create_label(tag_index)
        if tag_index < len(self.__box_tags):
            self.__box_tags.append(current_label_row)
        else:
            self.__box_tags[tag_index] = current_label_row

        current_label_row.Bind(EVT_BOX_LABEL_EDITED, self.__on_label_edited)
        current_label_row.set_tag(tag)


    def get_or_create_label(self, index: int) -> BoxTagLabelRow:
        """Get or create a BoxTagLabelRow for the given index."""
        if index < len(self.__box_tags) and self.__box_tags[index] is not None:
            return self.__box_tags[index]

        return BoxTagLabelRow(self, self.__box, index, self.__choices)

    def __on_label_edited(self, event: BoxLabelEditedEvent) -> None:
        """Handle label updates."""
        self.__box.set_tag(event.label_index, event.new_label)
        boxUpdateEvent = BoxEditedEvent(self, self.__box)
        wx.PostEvent(self, boxUpdateEvent)

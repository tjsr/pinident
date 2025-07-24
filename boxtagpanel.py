from typing import List

import wx

from boxdata import BoxData
from events.BoxEditedEvent import BoxEditedEvent
from events.BoxLabelEditEvent import BoxLabelEditedEvent
from events.events import EVT_BOX_LABEL_EDITED


class BoxTagLabelRow(wx.Panel):
    # __combo_box: wx.ComboBox
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
        label = box.get_tag(tag_index)
        self.__tag_index = tag_index

        # self.__combo_box = wx.ComboBox(self, value=label, choices=choices, style=wx.CB_DROPDOWN)
        self.__text_entry = wx.TextCtrl(self, value=label, style=wx.TE_PROCESS_ENTER)
        self.__rem_button = wx.BitmapButton(self, bitmap=wx.ArtProvider.GetBitmap(wx.ART_MINUS, wx.ART_BUTTON))

        sizer = wx.BoxSizer(wx.HORIZONTAL)
        # sizer.Add(self.__combo_box, 1, wx.EXPAND | wx.ALL, 5)
        sizer.Add(self.__text_entry, 1, wx.EXPAND | wx.ALL, 5)
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

    def set_tag(self, tag: str, is_only_row: bool) -> None:
        """Set the label of the combo box."""
        # self.__combo_box.SetValue(tag)
        if tag is not None and tag != self.__text_entry.GetValue():
            self.__text_entry.SetValue(tag)

        if is_only_row != self.__is_only_row:
            self.set_only_row(is_only_row)

    def label_edited(self, event: wx.Event) -> None:
        """Handle label updates."""
        # This method can be overridden to handle label changes
        # For example, you might want to notify the parent panel
        # that the label has changed.
        value = self.__text_entry.GetLineText(0)
        if not value:
            value = ""
        updateEvent = BoxLabelEditedEvent(self, self.__tag_index, value)
        wx.PostEvent(self, updateEvent)


class BoxTagPanelEdit(wx.Panel):
    __box: BoxData
    __box_tags: List[BoxTagLabelRow]
    __heading_text: wx.StaticText
    __del_button: wx.BitmapButton
    __choices: List[str] = []
    __sizer: wx.BoxSizer
    __add_button: wx.Button

    def is_box(self, box: BoxData) -> bool:
        """Check if the given box matches the current box."""
        return self.__box == box or self.__box.coords == box.coords

    def __init__(
        self,
        parent: wx.Panel,
        box: BoxData
    ):
        super().__init__(parent)
        self.__is_selected: bool = False
        self.__box = box
        self.__heading_text = wx.StaticText(self, label=f"Box: {box.coords}")
        self.__box_tags = []
        self.__sizer = wx.BoxSizer(wx.VERTICAL)
        self.__sizer.Add(self.__heading_text, 0, wx.EXPAND | wx.ALL, 5)

        self.__add_button = wx.Button(self, label="Add")
        self.__add_button.Bind(wx.EVT_BUTTON, self.__on_add_tag)
        self.__sizer.Add(self.__add_button, 0, wx.ALIGN_LEFT | wx.ALL, 5)

        self.SetSizer(self.__sizer)
        self.repaint_box()

    def repaint_box(self) -> None:
        tag_index: int = 0
        tag_count: int = len(self.__box.tags)

        self.__heading_text.SetLabelText(f"Box: {self.__box.coords}")

        while tag_index < tag_count:
            print(f'BoxTagPanelEdit.repaint_box: Repainting tag {tag_index}/{tag_count} on {self}')
            self.repaint_tag(tag_index)
            tag_index += 1

        # Remove any extra labels
        while len(self.__box_tags) > tag_index:
            removed_tag = self.__box_tags.pop()
            removed_tag.Destroy()

        self.Layout()

    def repaint_tag(self, tag_index: int) -> None:
        tag = self.__box.tags[tag_index]
        if not isinstance(tag, str):
            raise ValueError(f"Invalid tag type: {type(tag)}. Expected str.")

        # box_label_count: int = len(self.__box_labels)
        current_label_row: BoxTagLabelRow = self.get_or_create_label(tag_index)
        if tag_index >= len(self.__box_tags):
            self.__box_tags.append(current_label_row)
        else:
            self.__box_tags[tag_index] = current_label_row

        is_only_row = len(self.__box_tags) == 1 and tag_index == 0

        current_label_row.Bind(EVT_BOX_LABEL_EDITED, self.__on_label_edited)
        current_label_row.set_tag(tag, is_only_row)

    def get_or_create_label(self, index: int) -> BoxTagLabelRow:
        """Get or create a BoxTagLabelRow for the given index."""
        if index < len(self.__box_tags) and self.__box_tags[index] is not None:
            return self.__box_tags[index]

        new_label = BoxTagLabelRow(self, self.__box, index, self.__choices)

        self.__sizer.Insert(index+1, new_label, 0, wx.EXPAND | wx.ALL, 2)
        return new_label

    def __on_label_edited(self, event: BoxLabelEditedEvent) -> None:
        """Handle label updates."""
        print(f'BoxTagPanelEdit.__on_label_edited: {event.label_index}={event.new_label}')
        self.__box.set_tag(event.label_index, event.new_label)
        boxUpdateEvent = BoxEditedEvent(self, self.__box)
        wx.PostEvent(self, boxUpdateEvent)

    def __on_add_tag(self, event: wx.CommandEvent) -> None:
        # Add a new empty tag
        self.__box.tags.append("")
        self.repaint_box()
        boxEditEvent = BoxEditedEvent(self, self.__box)
        wx.PostEvent(self, boxEditEvent)

    @property
    def box(self) -> BoxData:
        return self.__box

    @property
    def selected(self) -> bool:
        return self.__is_selected

    @selected.setter
    def selected(self, value: bool) -> None:
        """Set the selection state of the box."""
        self.__is_selected = value
        heading_font: wx.Font = wx.Font(10, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL)
        if value:
            self.SetBackgroundColour(wx.Colour(200, 255, 200))  # Highlight color
            heading_font = wx.Font(10, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD)
        else:
            self.SetBackgroundColour(wx.NullColour)  # Default color

        self.__heading_text.SetFont(heading_font)
        self.Refresh()  # Refresh to apply the new background color
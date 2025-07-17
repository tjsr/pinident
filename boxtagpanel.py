from typing import List, Callable

import wx

from boxdata import BoxData

tEVT_BOX_LABEL_UPDATED = wx.NewEventType()
EVT_BOX_LABEL_UPDATED = wx.PyEventBinder(tEVT_BOX_LABEL_UPDATED, 1)


class BoxLabelUpdatedEvent(wx.PyCommandEvent):
    """Custom event for box label updates."""
    def __init__(self, source: wx.Panel, new_label: str):
        super().__init__(wx.NewId(), source.GetId())
        self.new_label = new_label

    def GetNewLabel(self) -> str:
        return self.new_label


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
        choices: List[str],
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
        self.__text_entry.Bind(wx.EVT_TEXT, self.label_updated)

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

    def label_updated(self, event: wx.Event) -> None:
        """Handle label updates."""
        # This method can be overridden to handle label changes
        # For example, you might want to notify the parent panel
        # that the label has changed.
        value = self.__text_entry.GetLineText(0)
        updateEvent = BoxLabelUpdatedEvent(self, value)
        wx.PostEvent(self, updateEvent)


class BoxTagPanelEdit(wx.Panel):
    __box: BoxData
    __box_labels: List[BoxTagLabelRow]
    __on_box_updated: Callable[[], None]
    # __on_delete_box: Callable[[BoxData], None]
    # __on_update_box_data: Callable[[BoxData], None]
    __heading_text: wx.StaticText
    __del_button: wx.BitmapButton
    __choices: List[str] = []

    def __init__(
        self,
        parent: wx.Panel,
        box: BoxData
    ):
        super().__init__(parent)
        self.__box = box
        self.on_delete_box = lambda x: None
        self.on_update_box_data = lambda x: None
        self.__heading_text = wx.StaticText(parent, label=f"Box: {box.coords}")

    def repaint_boxes(self) -> None:
        tag_index: int = 0
        tag_count: int = len(self.__box.tags)

        self.__heading_text.SetLabelText(f"Box: {self.__box.coords}")

        while tag_index < tag_count:
            self.repaint_box(tag_index)
            tag_index += 1

        # Remove any extra labels
        while len(self.__box_labels) > tag_index:
            self.__box_labels.pop()

    def repaint_box(self, tag_index: int) -> None:
        tag = self.__box.tags[tag_index]
        if not isinstance(tag, str):
            raise ValueError(f"Invalid tag type: {type(tag)}. Expected str.")

        # box_label_count: int = len(self.__box_labels)
        current_label_row: BoxTagLabelRow = self.get_or_create_label(tag_index)
        if tag_index < len(self.__box_labels):
            self.__box_labels.append(current_label_row)
        else:
            self.__box_labels[tag_index] = current_label_row

        current_label_row.set_tag(tag)

    def get_or_create_label(self, index: int) -> BoxTagLabelRow:
        """Get or create a BoxTagLabelRow for the given index."""
        if index < len(self.__box_labels) and self.__box_labels[index] is not None:
            return self.__box_labels[index]

        return BoxTagLabelRow(self, "", self.__choices)

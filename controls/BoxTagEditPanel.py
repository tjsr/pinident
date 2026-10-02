from typing import List

import wx

from boxdata import BoxData
from controls.boxtaglaberow import BoxTagLabelRow
from events.BoxEditedEvent import BoxEditedEvent
from events.BoxLabelEditEvent import BoxLabelEditedEvent
from events import BoxLabelRemoveEvent
from events.events import EVT_BOX_LABEL_EDITED, EVT_BOX_EDITED, EVT_BOX_LABEL_REMOVE
from logutil import getLog


class BoxTagPanelEdit(wx.Panel):
    __box: BoxData
    __box_tags: List[BoxTagLabelRow]
    __heading_text: wx.StaticText
    __del_button: wx.BitmapButton
    __choices: List[str] = []
    __sizer: wx.BoxSizer
    __add_button: wx.Button

    def is_box(self, box: BoxData) -> bool:
        return self.__box.matches(box)

    def __init__(
        self,
        parent,
        box: BoxData
    ):
        super().__init__(parent)
        self.__is_selected: bool = False
        self.__heading_text = wx.StaticText(self, label=f"Box: {box.coords}")
        self.__heading_text.SetMinSize(wx.Size(240, -1))
        self.__sizer = wx.BoxSizer(wx.VERTICAL)
        self.__sizer.Add(self.__heading_text, 0, wx.EXPAND | wx.ALL, 5)

        self.__add_button = wx.Button(self, label="Add")
        self.__add_button.Bind(wx.EVT_BUTTON, self.__on_add_tag)
        self.Bind(EVT_BOX_EDITED, self.update_after_edited)
        self.__sizer.Add(self.__add_button, 0, wx.ALIGN_LEFT | wx.ALL, 5)

        self.SetSizer(self.__sizer)

        self.box = box


    def Refresh(self, eraseBackground=True, rect=None) -> None:
        log = getLog()
        tag_index: int = 0
        tag_count: int = len(self.__box.tags)

        self.__heading_text.SetLabelText(f"Box: {self.__box.coords}")

        # current_tags = []

        sizer_children = self.__sizer.GetChildren()
        while len(sizer_children) > 2:
            child = sizer_children[1].GetWindow()
            self.__sizer.Detach(1)
            # child.Destroy()

        while tag_index < tag_count:
            # log.debug(f'Repainting tag {tag_index+1}/{tag_count} on {self}')
            tag = self.repaint_tag(tag_index)
            # log.debug(f'Repainted tag {tag_index+1}/{tag_count} on {self} with label: {tag.tag}')
            tag_index += 1
            self.__sizer.Insert(self.__sizer.GetItemCount() - 1, tag, 0, wx.EXPAND | wx.ALL, 2)

        self.Layout()
        super().Refresh(eraseBackground, rect)


    def repaint_tag(self, tag_index: int) -> BoxTagLabelRow:
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

        current_label_row.set_tag(tag, is_only_row)
        # self.Bind(EVT_BOX_LABEL_EDITED, self.__on_label_edited, current_label_row)
        try:
            current_label_row.Refresh()
        except Exception as e:
            getLog().error(f"Failed to refresh label row: {current_label_row}. It may have been destroyed.", exc_info=e)
            return current_label_row
        return current_label_row

    def get_or_create_label(self, index: int) -> BoxTagLabelRow:
        """Get or create a BoxTagLabelRow for the given index."""
        if index < len(self.__box_tags) and self.__box_tags[index] is not None:
            return self.__box_tags[index]

        new_label = BoxTagLabelRow(self, self.__box, index)
        new_label.Bind(EVT_BOX_LABEL_REMOVE, self.__on_tag_remove)
        new_label.Bind(EVT_BOX_LABEL_EDITED, self.__on_label_edited)
        new_label.Bind(wx.EVT_PAINT, self.__on_label_repainted)

        return new_label

    def __on_label_edited(self, event: BoxLabelEditedEvent) -> None:
        """Handle label updates."""
        getLog().debug(f'{event.label_index}={event.new_label}')
        self.__box.set_tag(event.label_index, event.new_label)
        boxUpdateEvent = BoxEditedEvent(self, self.__box)
        wx.PostEvent(self, boxUpdateEvent)

    def __on_add_tag(self, event: wx.CommandEvent) -> None:
        # Add a new empty tag
        self.__box.tags.append("")
        boxEditEvent = BoxEditedEvent(event.GetEventObject(), self.__box)
        wx.PostEvent(self, boxEditEvent)

    def __on_label_repainted(self, event: wx.PaintEvent) -> None:
        # getLog().debug(f'Repainting label row: {self}->{event.GetEventObject()}')
        pass

    def __on_tag_remove(self, event: BoxLabelRemoveEvent) -> None:
        """Handle tag removal."""
        if event.label_index < 0 or event.label_index >= len(self.__box.tags):
            return

        self.__box.remove_tag(event.label_index)
        event_obj = event.GetEventObject()
        boxEditEvent = BoxEditedEvent(event_obj, self.__box)
        wx.PostEvent(self, boxEditEvent)

    def update_after_edited(self, event: BoxLabelEditedEvent):
        getLog().debug('Got box edited event, updating data.')
        self.box = event.box

    @property
    def box(self) -> BoxData:
        return self.__box

    @box.setter
    def box(self, value: BoxData) -> None:
        """Set the box data and update the UI."""
        if not isinstance(value, BoxData):
            raise ValueError("Expected a BoxData instance.")
        self.__box = value
        self.__box_tags = []
        self.Refresh()  # Refresh to apply the new box data

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

    def __str__(self) -> str:
        """Return a string representation of the box."""
        return f"BoxTagPanelEdit(box={self.__box})"

if __name__ == "__main__":
    import wx
    from boxdata import BoxData

    class TestFrame(wx.Frame):
        def __init__(self):
            super().__init__(None, title="BoxTagPanelEdit Test", size=wx.Size(300, 200))
            box = BoxData(coords=(10, 20, 30, 40), tags=["Tag1", "Tag2"], source='automatic')
            panel = BoxTagPanelEdit(self, box)
            # panel.Bind(EVT_BOX_EDITED, self.__on_box_edited)
            self.Show()

        def __on_box_edited(self, event: BoxEditedEvent):
            getLog().debug(f"Box edited: {event.box}")

    app = wx.App(False)
    frame = TestFrame()
    app.MainLoop()


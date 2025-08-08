from typing import List

import wx
from wx import EVT_BUTTON

from boxdata import BoxData
from events.BoxLabelEditEvent import BoxLabelEditedEvent
from events import BoxLabelRemoveEvent
from logutil import getLog


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
        tag_index: int
    ):
        super().__init__(parent)
        label = box.get_tag(tag_index)
        self.__tag_index = tag_index

        # self.__combo_box = wx.ComboBox(self, value=label, choices=choices, style=wx.CB_DROPDOWN)
        self.__text_entry = wx.TextCtrl(self, value=label, style=wx.TE_PROCESS_ENTER)
        self.Bind(wx.EVT_WINDOW_DESTROY, self.__on_text_destroy)
        self.__rem_button = wx.BitmapButton(self, bitmap=wx.ArtProvider.GetBitmap(wx.ART_MINUS, wx.ART_BUTTON))

        self.timer = wx.Timer(self)
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        # sizer.Add(self.__combo_box, 1, wx.EXPAND | wx.ALL, 5)
        sizer.Add(self.__text_entry, 1, wx.EXPAND | wx.ALL, 5)
        sizer.Add(self.__rem_button, 0, wx.ALL, 5)

        # self.__combo_box.Bind(wx.EVT_TEXT, self.label_updated)
        self.__text_entry.Bind(wx.EVT_TEXT, self.label_edited)
        self.Bind(wx.EVT_TIMER, self.on_label_edited_timer, self.timer)
        self.__rem_button.Bind(EVT_BUTTON, self.on_remove_button_clicked)

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
        if self.__text_entry is None:
            return

        # self.__combo_box.SetValue(tag)
        try:
            if self.__text_entry.IsBeingDeleted():
                getLog().debug("Text entry is being deleted, cannot set tag.")
                return

            if tag is not None and tag != self.__text_entry.GetValue():
                self.__text_entry.SetValue(tag)

            if is_only_row != self.__is_only_row:
                self.set_only_row(is_only_row)
        except wx._core.PyAssertionError as e:
            getLog().error(f"PyAssertionError setting tag: {e}")
            # Handle the case where the text entry is not initialized or has been destroyed
        except Exception as ex:
            getLog().error(f"Unexpected error setting tag: {ex}")
            # Handle any other unexpected errors

    def label_edited(self, event: wx.Event) -> None:
        """Handle label edits."""
        # Start the timer to delay the event handling
        self.timer.Start(500, oneShot=True)

    def on_label_edited_timer(self, event: wx.TimerEvent) -> None:
        self.fire_edited_event()
        event.Skip()

    def fire_edited_event(self):
        value = self.__text_entry.GetLineText(0)
        if not value:
            value = ""
        updateEvent = BoxLabelEditedEvent(self, self.__tag_index, value)
        wx.PostEvent(self, updateEvent)

    def on_text_blur(self, event):
        self.timer.Stop()
        self.fire_edited_event()
        event.Skip()

    def on_remove_button_clicked(self, event: wx.CommandEvent):
        """Handle the remove button click."""
        if self.__is_only_row:
            getLog().debug("Cannot remove the only row.")
            return

        remove_event = BoxLabelRemoveEvent(event.GetEventObject(), self.__tag_index)
        wx.PostEvent(self, remove_event)

        # # Remove the label of the current index from the box's tag list.
        # self.__box.remove_tag(self.__tag_index)
        #
        #
        # # Get the label text of the tag at the current index
        # label_text = self.__text_entry.GetValue()
        # if not label_text:
        #     return
        # # Remove any  tag matching that label text.
        #
        # # Remove this label row from the parent panel
        # parent_panel: BoxTagPanelEdit = self.GetParent()
        # parent_panel.__box.tags.pop(self.__tag_index)

    def __on_text_destroy(self, event: wx.WindowDestroyEvent) -> None:
        getLog().debug("Destroying text entry in BoxTagLabelRow")

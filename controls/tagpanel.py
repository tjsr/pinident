import wx
from wx.lib.scrolledpanel import ScrolledPanel
from logutil import getLog

from typing import List

from boxdata import BoxData
from controls import BoxTagPanelEdit, ImagePanel
from events.BoxEditedEvent import BoxEditedEvent
from events.BoxRemovedEvent import BoxRemovedEvent
from events.BoxSelectedEvent import BoxSelectedEvent
from events.BoxUpdatedEvent import BoxUpdatedEvent
from events.BoxAddedEvent import BoxAddedEvent
from events.events import EVT_BOX_UPDATED, EVT_BOX_ADDED, EVT_BOX_REMOVED, EVT_BOX_EDITED, wxEVT_BOX_ADDED

class TagPanel(ScrolledPanel, wx.PyEventBinder):
    __box_panels: List[BoxTagPanelEdit] = []

    vbox: wx.BoxSizer
    __box_sizer: wx.BoxSizer
    pin_cb: wx.CheckBox
    set_cb: wx.CheckBox
    card_cb: wx.CheckBox

    def __init__(
        self,
        parent: wx.Window
    ) -> None:
        super().__init__(parent)
        # self.__boxes = None

        self.vbox = wx.BoxSizer(wx.VERTICAL)
        self.__box_panels = []

        self.pin_cb = wx.CheckBox(self, label="Contains pin")
        self.set_cb = wx.CheckBox(self, label="Contains set")
        self.card_cb = wx.CheckBox(self, label="With backing card")
        self.__box_sizer = wx.BoxSizer(wx.VERTICAL)
        self.SetMinSize(wx.Size(180, -1))
        self.SetScrollRate(0, 20)
        self.Bind(wx.EVT_SIZE, self.on_size)

        self.vbox.Add(self.pin_cb, 0, wx.ALL, 5)
        self.vbox.Add(self.set_cb, 0, wx.ALL, 5)
        self.vbox.Add(self.card_cb, 0, wx.ALL, 5)
        self.vbox.Add(self.__box_sizer, 1, wx.ALL, 5)

        self.Bind(EVT_BOX_EDITED, self.__on_box_edited)

        self.SetSizer(self.vbox)
        self.SetupScrolling()
        self.Refresh()

    def find_panel_for_box(self, box: BoxData) -> BoxTagPanelEdit | None:
        idx: int = self.__get_existing_box_index(box)
        panel_count: int = len(self.__box_panels)
        if panel_count <= idx:
            return None
        if idx >= 0:
            return self.__box_panels[idx]

        error_message = f'No BoxTagPanelEdit found for box {box.coords} in list {self.__box_panels}.'
        getLog().error(error_message)
        raise ValueError(error_message)

    def resize_box_panels(self, min_width: int = 180) -> None:
        for idx, panel in enumerate(self.__box_panels):
            # Adjust the minimum size of each panel
            panel.SetMinSize(wx.Size(min_width, -1))
            # Optionally, update sizer item proportion/flags
            self.__box_sizer.SetItemMinSize(panel, min_width, -1)
        self.vbox.Layout()
        self.FitInside()

    def __create_panel_for_box(self, box: BoxData) -> BoxTagPanelEdit:
        box_tag_panel = BoxTagPanelEdit(self, box)
        box_tag_panel.Bind(EVT_BOX_EDITED, self.__on_box_edited)
        box_tag_panel.Bind(wx.EVT_PAINT, self.__on_tag_panel_painted)
        box_tag_panel.Bind(wx.EVT_LEFT_DOWN, self.__on_box_clicked)
        return box_tag_panel

    def Refresh(self, eraseBackground=True, rect=None) -> bool:
        log = getLog()
        # Save focus info before clearing panels
        focused: wx.Window | None = wx.Window.FindFocus()
        focused_value: str | None = None
        focused_pos: int | None = None
        if isinstance(focused, wx.TextCtrl):
            focused_value = focused.GetValue()
            focused_pos = focused.GetInsertionPoint()

        # self.__box_sizer.Clear(True)
        # self.__box_panels.clear()

        for idx, panel in enumerate(self.__box_panels):
            try:
                self.__box_sizer.Detach(panel)
                log.debug(f'Reusing panel for box[{idx+1}]={panel.box})')
            except ValueError:
                log.debug(f'Creating new panel for box[{idx+1}]={panel.box}')
                # # If no existing panel found, create a new one
                # self.__box_panels.append(panel)

            self.__box_sizer.Add(panel, 0, wx.ALIGN_LEFT | wx.ALL, 2)

        # for panel_index, panel in enumerate(self.__box_panels):
        #     box_index = self.__get_existing_box_index(panel.box)
        #     if box_index < 0:
        #         log.warning(f'Box {panel.box} not found in current boxes, removing panel {panel_index}.')
        #         # If the box is not found in the current boxes, remove the panel
        #         self.__box_panels.remove(panel)
        #         self.__box_sizer.Remove(panel_index)
        #         panel.Destroy()
        #         continue

            # if panel.box is not None:
            #     log.debug(f'Panel {panel_index} has box {panel.box}.')

        self.__box_sizer.Layout()
        self.Layout()
        super().Refresh(eraseBackground, rect)

        # Restore focus and cursor position
        if focused_value is not None:
            for panel in self.__box_panels:
                for ctrl in panel.GetChildren():
                    if isinstance(ctrl, wx.TextCtrl) and ctrl.GetValue() == focused_value:
                        ctrl.SetFocus()
                        if focused_pos is not None:
                            ctrl.SetInsertionPoint(focused_pos)
                        break

        return True

    def __on_tag_panel_painted(self, event: wx.PaintEvent) -> None:
        self.__box_sizer.Layout()
        self.vbox.Layout()

    @property
    def boxes(self) -> List[BoxData]:
        return [
            panel.box for panel in self.__box_panels if panel.box is not None
        ]

    def get_removed_boxes(self, boxes: List[BoxData]) -> List[BoxTagPanelEdit]:
        """Get the removed boxes."""
        if self.boxes is None:
            return []

        filtered_panels = [
            panel for panel in self.__box_panels
            if all(not panel.is_box(box) for box in boxes)
        ]

        return filtered_panels

    def __remove_panels(self, panels: List[BoxTagPanelEdit]) -> None:
        """Remove panels from the box sizer."""
        for panel in panels:
            idx = self.__get_existing_box_index(panel.box)
            if idx > 0:
                # getLog().debug(f'Removing panel {panel} from box sizer at index {idx}.')
                self.__box_sizer.Remove(idx)
                panel.Destroy()
                self.__box_panels.remove(panel)
            else:
                # getLog().warning(f'Panel {panel} not found in box panels, cannot remove.')
                pass

    def __get_existing_boxes(self, boxes: List[BoxData]) -> List[BoxData]:
        """Get the existing boxes."""
        if self.boxes is None:
            return []

        filtered_boxes = [
            panel.box for panel in self.__box_panels
            if any(panel.is_box(box) for box in boxes)
        ]

        return filtered_boxes

    def __get_new_boxes(self, boxes: List[BoxData]) -> List[BoxData]:
        """Get the new boxes that are not already in the box panels."""
        if self.boxes is None:
            return []

        new_boxes = [
            box for box in boxes
            if self.__get_existing_box_index(box) < 0
        ]

        return new_boxes

    def __update_panels(self, boxes: List[BoxData]) -> None:
        for box in boxes:
            index = self.__get_existing_box_index(box)
            if index >= 0:
                getLog().debug(f'Box {box.coords} already exists at index {index}, reusing panel.')
                panel = self.__box_panels[index]
                panel.box = box
            else:
                getLog().debug(f'No existing panel to update {box}.')


        """Update the panels with the given boxes."""
        if boxes is None or len(boxes) == 0:
            getLog().debug('No boxes provided to update panels.')
            return

        getLog().info(f'Updating panels with {len(boxes)} boxes.')

        self.Refresh()

    def __add_panel_for_box(self, box: BoxData) -> BoxTagPanelEdit:
        new_panel = self.__create_panel_for_box(box)
        self.__box_panels.append(new_panel)
        # insert_index = len(self.__box_sizer.GetChildren()) - 1  # Insert before the last item (the add button)
        getLog().debug(f'Inserting new panel for box {box}.')
        # self.__box_sizer.Insert(insert_index, new_panel, 0, wx.ALIGN_LEFT | wx.ALL, 2)
        self.__box_sizer.Add(new_panel, 0, wx.ALIGN_LEFT | wx.ALL, 2)
        return new_panel

    def __add_panels_for_boxes(self, boxes: List[BoxData]) -> None:
        """Create panels for the given boxes."""
        if boxes is None or len(boxes) == 0:
            getLog().debug('No boxes provided to create panels.')
            return

        getLog().info(f'Creating panels for {len(boxes)} boxes.')

        for box in boxes:
            if not isinstance(box, BoxData):
                getLog().error(f'Invalid box type {type(box)}. Expected BoxData.')
                continue

            self.__add_panel_for_box(box)

    @boxes.setter
    def boxes(self, boxes: List[BoxData]) -> None:
        removed = self.get_removed_boxes(boxes)
        self.__remove_panels(removed)

        existing = self.__get_existing_boxes(boxes)
        self.__update_panels(existing)

        new_boxes = self.__get_new_boxes(boxes)
        self.__add_panels_for_boxes(new_boxes)

        self.Refresh()

    def __get_existing_box_index(self, box: BoxData) -> int:
        """Get the index of an existing box in the list."""
        # if self.__boxes is None:
        #     getLog().error('No boxes available to search for existing box index.')
        #     return -1

        for idx, existing_panel in enumerate(self.__box_panels):
            if existing_panel.is_box(box):
                return idx
        return -1

    def update_box(self, box: BoxData) -> None:
        getLog().info(f'Updating box {box.coords} in TagPanel with {len(self.__box_panels)} boxes')
        """Update a specific box."""
        try:
            box_panel = self.find_panel_for_box(box)
            box_panel.Refresh()
        except ValueError:
            getLog().error(f"Box {box} not found among {len(self.__box_panels)} panels containing list {self.__box_panels}")
            # raise ValueError(f"Box {box} not found in current panels.")

    def __on_box_edited(self, event: BoxEditedEvent) -> None:
        """Handle box edited event."""
        getLog().info(f'Box {event.box} edited in {event.GetEventObject()}')
        self.update_box(event.box)

    def __on_box_clicked(self, event: wx.MouseEvent) -> None:
        """Handle box click event."""
        log = getLog()

        panel = event.GetEventObject()
        if isinstance(panel, BoxTagPanelEdit):
            box = panel.box
            getLog().debug(f'Clicked box: {box}')
            # Now you have the box associated with the clicked component
            evt = BoxSelectedEvent(panel, box)
            wx.PostEvent(self, evt)
            log.debug(f'Box clicked in TagPanel {self} with event {event}')

    def __on_boxes_updated(self, event: BoxUpdatedEvent) -> None:
        self.boxes = event.boxes

    def __on_box_added(self, event: BoxAddedEvent) -> None:
        log = getLog()
        """Handle box updated event."""
        log.debug(f'Added {self.__box_panels} and event {type(event)} {event.GetEventType()} {wxEVT_BOX_ADDED}')
        if isinstance(event, BoxAddedEvent):
            log.debug(f'Adding box {event.box.coords} to panel')
            self.__add_panel_for_box(event.box)
        else:
            log.warning(f'TagPanel.__on_box_added: event parameter is not a BoxAddedEvent, skipping')
        self.Refresh()

    def __remove_box_panel(self, box: BoxData) -> None:
        filtered_panels = [
            panel for panel in self.__box_panels
            if panel.is_box(box)
        ]

        self.__remove_panels(filtered_panels)

    def __on_box_removed(self, event: BoxRemovedEvent) -> None:
        """Handle box removed event."""
        try:
            self.__remove_box_panel(event.box)
            self.Refresh()
        except ValueError as e:
            getLog().error(f'Error removing box {event.box}: {e}')

    def bind_box_events(self, image_panel: ImagePanel) -> None:
        """Bind box events to the image panel."""
        image_panel.Bind(EVT_BOX_ADDED, self.__on_box_added)
        image_panel.Bind(EVT_BOX_REMOVED, self.__on_box_removed)
        image_panel.Bind(EVT_BOX_UPDATED, self.__on_boxes_updated)
        image_panel.Bind(EVT_BOX_EDITED, self.__on_box_edited)

    def on_box_selected(self, event: BoxSelectedEvent) -> None:
        """Handle box selection event."""
        getLog().debug(f'Box selected {event.box}')
        for panel in self.__box_panels:
            panel.selected = event.box is not None and panel.is_box(event.box)

    def on_size(self, event):
        self.resize_box_panels()
        self.SetScrollRate(0, 20)  # Re-apply scroll rate
        event.Skip()
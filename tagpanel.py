import wx
from wx.lib.scrolledpanel import ScrolledPanel
from logutil import getLog

from typing import List

from boxdata import BoxData
from boxtagpanel import BoxTagPanelEdit
from events.BoxEditedEvent import BoxEditedEvent
from events.BoxRemovedEvent import BoxRemovedEvent
from events.BoxSelectedEvent import BoxSelectedEvent
from events.BoxUpdatedEvent import BoxUpdatedEvent
from imagepanel import ImagePanel
from events.BoxAddedEvent import BoxAddedEvent
from events.events import EVT_BOX_UPDATED, EVT_BOX_ADDED, EVT_BOX_REMOVED, EVT_BOX_EDITED, wxEVT_BOX_ADDED

class TagPanel(ScrolledPanel, wx.PyEventBinder):
    __boxes: List[BoxData] | None
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
        self.__boxes = None

        self.vbox = wx.BoxSizer(wx.VERTICAL)
        # self.__box_sizers = []
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

        self.set_ui()
        self.SetSizer(self.vbox)
        self.SetupScrolling()

    def find_panel_for_box(self, box: BoxData) -> BoxTagPanelEdit | None:
        """Find the BoxTagPanelEdit for a given box."""
        for panel in self.__box_panels:
            if panel.is_box(box):
                return panel
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

    def set_ui(self) -> None:
        log = getLog()
        # Save focus info before clearing panels
        focused: wx.Window | None = wx.Window.FindFocus()
        focused_value: str | None = None
        focused_pos: int | None = None
        if isinstance(focused, wx.TextCtrl):
            focused_value = focused.GetValue()
            focused_pos = focused.GetInsertionPoint()

        # while len(self.__box_panels) > 0:
        #     panel = self.__box_panels.pop()
        #     panel.Destroy()

        self.__box_sizer.Clear(True)
        self.__box_panels.clear()

        if self.__boxes is None:
            log.info('No boxes to display')
            return

        for idx, box in enumerate(self.__boxes):
            try:
                box_tag_panel = self.find_panel_for_box(box)
                self.__box_sizer.Detach(box_tag_panel)
                log.debug(f'Reusing panel for box[{idx+1}]={box})')
            except ValueError:
                log.debug(f'Creating new panel for box[{idx+1}]={box}')
                # If no existing panel found, create a new one
                box_tag_panel = BoxTagPanelEdit(self, box)
                self.__box_panels.append(box_tag_panel)
                box_tag_panel.Bind(EVT_BOX_EDITED, self.__on_box_edited)
                box_tag_panel.Bind(wx.EVT_PAINT, self.__on_tag_panel_painted)

            self.__box_sizer.Add(box_tag_panel, 0, wx.ALIGN_LEFT | wx.ALL, 2)

        for panel_index, panel in enumerate(self.__box_panels):
            found = False
            if panel is not None:
                for idx, box in enumerate(self.__boxes):
                    if panel.is_box(box):
                        found = True
                        break
                if not found:
                    panel_box = panel.box
                    log.info(f'Removed panel {panel_index} for box {panel_box}.')
                    self.__box_panels.remove(panel)
                    self.__box_sizer.Remove(panel_index)
                    panel.Destroy()

                if panel.box is not None:
                    log.debug(f'Panel {panel_index} has box {panel.box}.')

            # print(f'TagPanel.set_ui: Panel {panel_index} has box {panel.box.coords} with id {hex(id(panel.box))}')
        # while len(self.__box_panels) > 0:
        #     panel: BoxTagPanelEdit = self.__box_panels[-1]
        #     panel = self.__box_panels.pop()
        #     panel.Destroy()

        self.__box_sizer.Layout()
        self.Layout()

        # Restore focus and cursor position
        if focused_value is not None:
            for panel in self.__box_panels:
                for ctrl in panel.GetChildren():
                    if isinstance(ctrl, wx.TextCtrl) and ctrl.GetValue() == focused_value:
                        ctrl.SetFocus()
                        if focused_pos is not None:
                            ctrl.SetInsertionPoint(focused_pos)
                        break

    def __on_tag_panel_painted(self, event: wx.PaintEvent) -> None:
        self.__box_sizer.Layout()
        self.vbox.Layout()

    @property
    def boxes(self) -> List[BoxData]:
        """Get the boxes."""
        return self.__boxes if self.__boxes is not None else []

    @boxes.setter
    def boxes(self, boxes: List[BoxData]) -> None:
        """Set the boxes and update the UI."""
        self.__boxes = boxes
        self.set_ui()

    def update_boxes(self, boxes: List[BoxData]) -> None:
        self.__boxes = boxes
        self.set_ui()

    def update_box(self, box: BoxData) -> None:
        getLog().info(f'Updating box {box.coords} in TagPanel with {len(self.__boxes)} boxes')
        """Update a specific box."""
        try:
            box_panel = self.find_panel_for_box(box)
            box_panel.Refresh()
        except ValueError:
            getLog().error(f"Box {box} not found among {len(self.__box_panels)} panels containing list {self.__boxes}")
            # raise ValueError(f"Box {box} not found in current panels.")

    def __on_box_edited(self, event: BoxEditedEvent) -> None:
        """Handle box edited event."""
        getLog().info(f'Box {event.box} edited in {event.GetEventObject()}')
        self.update_box(event.box)


    def __on_boxes_updated(self, event: BoxUpdatedEvent) -> None:
        self.boxes = event.boxes

    def __on_box_added(self, event: BoxAddedEvent) -> None:
        log = getLog()
        """Handle box updated event."""
        log.debug(f'Added self.__boxes {self.__boxes} and event {type(event)} {event.GetEventType()} {wxEVT_BOX_ADDED}')
        if isinstance(event, BoxAddedEvent):
            log.debug(f'Adding box {event.box.coords} to panel')
            self.__boxes.append(event.box)
        else:
            log.warning(f'TagPanel.__on_box_added: event parameter is not a BoxAddedEvent, skipping')
        self.set_ui()

    def __on_box_removed(self, event: BoxRemovedEvent) -> None:
        """Handle box removed event."""
        try:
            box_panel = self.find_panel_for_box(event.box)
            box_panel.Destroy()
            self.__box_panels.remove(box_panel)
            self.set_ui()
        except ValueError:
            pass

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
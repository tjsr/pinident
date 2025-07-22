import wx
from typing import List

from boxdata import BoxData
from boxtagpanel import BoxTagPanelEdit
from events.BoxEditedEvent import BoxEditedEvent
from events.BoxRemovedEvent import BoxRemovedEvent
from events.BoxUpdatedEvent import BoxUpdatedEvent
from imagepanel import ImagePanel
from events.BoxAddedEvent import BoxAddedEvent
from events.events import EVT_BOX_UPDATED, EVT_BOX_ADDED, EVT_BOX_REMOVED, EVT_BOX_EDITED, wxEVT_BOX_ADDED


class TagPanel(wx.Panel, wx.PyEventBinder):
    boxes: List[BoxData] | None
    __box_panels: List[BoxTagPanelEdit] = []

    vbox: wx.BoxSizer
    # __box_sizers: List[wx.BoxSizer]
    __box_sizer: wx.BoxSizer
    pin_cb: wx.CheckBox
    set_cb: wx.CheckBox
    card_cb: wx.CheckBox

    def __init__(
        self,
        parent: wx.Window
    ) -> None:
        super().__init__(parent)
        self.boxes = None

        self.vbox = wx.BoxSizer(wx.VERTICAL)
        # self.__box_sizers = []
        self.__box_panels = []

        self.pin_cb = wx.CheckBox(self, label="Contains pin")
        self.set_cb = wx.CheckBox(self, label="Contains set")
        self.card_cb = wx.CheckBox(self, label="With backing card")
        self.__box_sizer = wx.BoxSizer(wx.VERTICAL)

        self.vbox.Add(self.pin_cb, 0, wx.ALL, 5)
        self.vbox.Add(self.set_cb, 0, wx.ALL, 5)
        self.vbox.Add(self.card_cb, 0, wx.ALL, 5)
        self.vbox.Add(self.__box_sizer, 1, wx.EXPAND | wx.ALL, 5)

        self.set_ui()
        self.SetSizerAndFit(self.vbox)

    def find_panel_for_box(self, box: BoxData) -> BoxTagPanelEdit:
        """Find the BoxTagPanelEdit for a given box."""
        for panel in self.__box_panels:
            if panel.is_box(box):
                return panel
        raise ValueError("No BoxTagPanelEdit found for the given box.")

    def set_ui(self) -> None:
        # Remove all box/tag controls but keep checkboxes
        # for sizer in self.__box_sizers:
        #     self.vbox.Remove(sizer)
        # self.__box_sizers.clear()

        while len(self.__box_panels) > 0:
            panel = self.__box_panels.pop()
            panel.Destroy()

        self.__box_sizer.Clear(True)
        self.__box_panels.clear()

        if self.boxes is None:
            print('TagPanel.set_ui: No boxes to display')
            return

        for idx, box in enumerate(self.boxes):
            try:
                box_tag_panel = self.find_panel_for_box(box)
                self.__box_sizer.Detach(box_tag_panel)
                print(f'TagPanel.set_ui: Reusing panel for box {idx+1} with coords {box.coords}')
            except ValueError:
                print(f'TagPanel.set_ui: Creating new panel for box {idx+1} with coords {box.coords}')
                # If no existing panel found, create a new one
                box_tag_panel = BoxTagPanelEdit(self, box)
                self.__box_panels.append(box_tag_panel)
                box_tag_panel.Bind(EVT_BOX_EDITED, self.__on_box_edited)
            self.__box_sizer.Add(box_tag_panel, 0, wx.EXPAND | wx.ALL, 2)
            # self.__box_sizers.append(box_sizer)


            # coords: Tuple[int, int, int, int] = box['coords']
            # heading = wx.StaticText(self, label=f"Box {idx+1}: x={coords[0]}, y={coords[1]}, w={coords[2]}, h={coords[3]}")
            # del_btn = wx.BitmapButton(self, bitmap=wx.ArtProvider.GetBitmap(wx.ART_DELETE, wx.ART_BUTTON))
            # del_btn.Bind(wx.EVT_BUTTON, lambda evt, i=idx: self.on_delete_box(i))
            # heading_sizer = wx.BoxSizer(wx.HORIZONTAL)
            # heading_sizer.Add(heading, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
            # heading_sizer.Add(del_btn, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
            # box_sizer.Add(heading_sizer, 0, wx.EXPAND)
            #
            # tag_sizer = wx.BoxSizer(wx.HORIZONTAL)
            # combo_list: List[wx.ComboBox] = []
            # rem_btn_list: List[wx.BitmapButton] = []
            # for t_idx, tag in enumerate(box['tags']):
            #     combo = wx.ComboBox(self, value=tag, choices=[], style=wx.CB_DROPDOWN)
            #     combo.Bind(wx.EVT_TEXT, lambda evt, i=idx, ti=t_idx: self.on_add_tag(i, ti, evt.GetString()))
            #     combo_list.append(combo)
            #     tag_sizer.Add(combo, 0, wx.ALL, 2)
            #     rem_btn = wx.BitmapButton(self, bitmap=wx.ArtProvider.GetBitmap(wx.ART_MINUS, wx.ART_BUTTON))
            #     rem_btn.Bind(wx.EVT_BUTTON, lambda evt, i=idx, ti=t_idx: self.on_remove_tag(i, ti))
            #     rem_btn_list.append(rem_btn)
            #     tag_sizer.Add(rem_btn, 0, wx.ALL, 2)
            # add_btn = wx.BitmapButton(self, bitmap=wx.ArtProvider.GetBitmap(wx.ART_PLUS, wx.ART_BUTTON))
            # add_btn.Bind(wx.EVT_BUTTON, lambda evt, i=idx: self.on_add_tag(i, len(box['tags']), ""))
            # tag_sizer.Add(add_btn, 0, wx.ALL, 2)
            # box_sizer.Add(tag_sizer, 0, wx.EXPAND)
            #
            # self.heading_texts.append(heading)
            # self.del_buttons.append(del_btn)
            # self.tag_sizers.append(tag_sizer)
            # self.combo_boxes.append(combo_list)
            # self.rem_buttons.append(rem_btn_list)
            # self.add_buttons.append(add_btn)

        self.vbox.Layout()
        self.Layout()

    def update_boxes(self, boxes: List[BoxData]) -> None:
        self.boxes = boxes
        self.set_ui()

    def update_box(self, box: BoxData) -> None:
        print(f'TagPanel.update_box: Updating box {box.coords} in TagPanel with {len(self.boxes)} boxes')
        """Update a specific box."""
        try:
            box_panel = self.find_panel_for_box(box)
            box_panel.repaint_box()
        except ValueError:
            raise ValueError("Box not found in current panels.")

    def __on_box_edited(self, event: BoxEditedEvent) -> None:
        """Handle box edited event."""
        print('TagPanel.__on_box_edited', f'Box {event.box} edited in {event.GetEventObject()}')
        self.update_box(event.box)

    def __on_boxes_updated(self, event: BoxUpdatedEvent) -> None:
        """Handle box edited event."""
        self.update_boxes(event.boxes)

    def __on_box_added(self, event: BoxAddedEvent) -> None:
        """Handle box updated event."""
        print(f'TagPanel.__on_box_added has self.__boxes {hex(id(self.boxes))} and event {type(event)} {event.GetEventType()} {wxEVT_BOX_ADDED}')
        if isinstance(event, BoxAddedEvent):
            print(f'TagPanel.__on_box_added: Adding box {event.box.coords} to panel')
            self.boxes.append(event.box)
        else:
            print(f'TagPanel.__on_box_added: event parameter is not a BoxAddedEvent, skipping')
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

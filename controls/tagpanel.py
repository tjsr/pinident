import wx
from wx.lib.scrolledpanel import ScrolledPanel
from logutil import getLog
from time import perf_counter

from typing import Callable, List

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
        parent: wx.Window,
        on_confirm: Callable[[BoxData], None] | None = None,
        on_remove: Callable[[BoxData], None] | None = None,
        on_interact: Callable[[], None] | None = None,
        on_change_begin: Callable[[], None] | None = None,
        on_change_end: Callable[[], None] | None = None,
        on_presence_change: Callable[[str, bool], None] | None = None,
        on_feedback_info: Callable[[], None] | None = None,
        catalog_options: list[tuple[str, str]] | None = None,
        on_add_training_data: Callable[[], None] | None = None,
        on_confirm_identity: Callable[[BoxData], None] | None = None,
    ) -> None:
        super().__init__(parent)
        # self.__boxes = None

        self.vbox = wx.BoxSizer(wx.VERTICAL)
        self.__box_panels = []
        self._on_confirm = on_confirm
        self._on_confirm_identity = on_confirm_identity
        self._on_remove = on_remove
        self._on_interact = on_interact
        self._on_change_begin = on_change_begin
        self._on_change_end = on_change_end
        self._on_presence_change = on_presence_change
        self._on_feedback_info = on_feedback_info
        self._catalog_options = catalog_options or []
        self._on_add_training_data = on_add_training_data

        self.pin_cb = wx.CheckBox(self, label="Contains pin")
        self.set_cb = wx.CheckBox(self, label="Contains set")
        self.card_cb = wx.CheckBox(self, label="With backing card")
        self.feedback_info_btn = wx.Button(self, label="Feedback model...")
        self.training_btn = wx.Button(self, label="Add to training data")
        self.training_gauge = wx.Gauge(self, range=100)
        self.training_status = wx.StaticText(self, label="Training idle")
        self.training_btn.Bind(wx.EVT_BUTTON, lambda _event: self._on_add_training_data()
                               if self._on_add_training_data else None)
        if on_feedback_info:
            self.feedback_info_btn.Bind(wx.EVT_BUTTON, lambda _event: on_feedback_info())
        self.pin_cb.Bind(wx.EVT_CHECKBOX, self._on_pin_checked)
        self.set_cb.Bind(wx.EVT_CHECKBOX, self._on_set_checked)
        self.__box_sizer = wx.BoxSizer(wx.VERTICAL)
        self.SetMinSize(wx.Size(410, -1))
        self.SetScrollRate(0, 20)
        self.Bind(wx.EVT_SIZE, self.on_size)

        self.vbox.Add(self.pin_cb, 0, wx.ALL, 5)
        self.vbox.Add(self.set_cb, 0, wx.ALL, 5)
        self.vbox.Add(self.card_cb, 0, wx.ALL, 5)
        self.vbox.Add(self.feedback_info_btn, 0, wx.ALL, 5)
        self.vbox.Add(self.training_btn, 0, wx.ALL, 5)
        self.vbox.Add(self.training_gauge, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 5)
        self.vbox.Add(self.training_status, 0, wx.ALL, 5)
        self.vbox.Add(self.__box_sizer, 1, wx.EXPAND | wx.ALL, 5)

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

    def resize_box_panels(self, min_width: int = 380) -> None:
        for idx, panel in enumerate(self.__box_panels):
            # Adjust the minimum size of each panel
            panel.SetMinSize(wx.Size(min_width, -1))
            # Optionally, update sizer item proportion/flags
            self.__box_sizer.SetItemMinSize(panel, min_width, -1)
        self.vbox.Layout()
        self.FitInside()

    def __create_panel_for_box(self, box: BoxData) -> BoxTagPanelEdit:
        box_tag_panel = BoxTagPanelEdit(self, box, self._on_confirm, self._on_remove,
                                       self._on_interact, self._on_change_begin,
                                       self._on_change_end, self._catalog_options,
                                       self._on_confirm_identity)
        box_tag_panel.Bind(EVT_BOX_EDITED, self.__on_box_edited)
        box_tag_panel.Bind(wx.EVT_PAINT, self.__on_tag_panel_painted)
        box_tag_panel.Bind(wx.EVT_LEFT_DOWN, self.__on_box_clicked)
        return box_tag_panel

    def set_catalog_options(self, options: list[tuple[str, str]]) -> None:
        started = perf_counter()
        self._catalog_options = options
        froze = bool(self.__box_panels) and not self.IsFrozen()
        if froze:
            self.Freeze()
        try:
            for panel in self.__box_panels:
                panel.set_catalog_options(options)
            if self.__box_panels:
                self.Layout()
        finally:
            if froze:
                self.Thaw()
        getLog().info('catalog choices=%d rows=%d refresh=%.1fms',
                      len(options), len(self.__box_panels),
                      (perf_counter()-started)*1000)

    def set_training_progress(self, percent: int, status: str, running: bool) -> None:
        self.training_gauge.SetValue(max(0, min(100, percent)))
        self.training_status.SetLabel(status)
        self.training_btn.Enable(not running)

    def Refresh(self, eraseBackground=True, rect=None) -> bool:
        return super().Refresh(eraseBackground, rect)

    def __on_tag_panel_painted(self, event: wx.PaintEvent) -> None:
        event.Skip()

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
            if idx >= 0:
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
        # A panel may be reused for a visually identical box on the next frame.
        # Update it with that frame's box object, not the previous frame's one.
        return [box for box in boxes if self.__get_existing_box_index(box) >= 0]

    def __get_new_boxes(self, boxes: List[BoxData]) -> List[BoxData]:
        """Get the new boxes that are not already in the box panels."""
        if self.boxes is None:
            return []

        new_boxes = [
            box for box in boxes
            if self.__get_existing_box_index(box) < 0
        ]

        return new_boxes

    def __update_panels(self, boxes: List[BoxData]) -> int:
        changed = 0
        for box in boxes:
            index = self.__get_existing_box_index(box)
            if index >= 0:
                panel = self.__box_panels[index]
                changed += panel.needs_refresh(box)
                panel.box = box
        return changed

    def __add_panel_for_box(self, box: BoxData) -> BoxTagPanelEdit:
        new_panel = self.__create_panel_for_box(box)
        new_panel.SetMinSize(wx.Size(380, -1))
        self.__box_panels.append(new_panel)
        self.__box_sizer.Add(new_panel, 0, wx.EXPAND | wx.ALL, 2)
        return new_panel

    def __add_panels_for_boxes(self, boxes: List[BoxData]) -> None:
        """Create panels for the given boxes."""
        if boxes is None or len(boxes) == 0:
            return

        for box in boxes:
            if not isinstance(box, BoxData):
                getLog().error(f'Invalid box type {type(box)}. Expected BoxData.')
                continue

            self.__add_panel_for_box(box)

    @boxes.setter
    def boxes(self, boxes: List[BoxData]) -> None:
        started = perf_counter()
        boxes = [box for box in boxes if box.review_state != 'rejected']
        removed = self.get_removed_boxes(boxes)
        existing = self.__get_existing_boxes(boxes)
        new_boxes = self.__get_new_boxes(boxes)
        diff_ready = perf_counter()
        structural_change = bool(removed or new_boxes)
        froze = structural_change and not self.IsFrozen()
        if froze:
            self.Freeze()
        try:
            self.__remove_panels(removed)
            updated = self.__update_panels(existing)
            updated_ready = perf_counter()
            self.__add_panels_for_boxes(new_boxes)
            added_ready = perf_counter()
            if structural_change:
                self.Layout()
                self.FitInside()
        finally:
            if froze:
                self.Thaw()
        if structural_change or updated:
            self.Refresh()
        finished = perf_counter()
        if structural_change or updated or finished - started > 0.05:
            getLog().info(
                'tag panels visible=%d reused=%d updated=%d added=%d removed=%d '
                'catalog=%d diff=%.1fms update=%.1fms create=%.1fms '
                'layout=%.1fms total=%.1fms',
                len(boxes), len(existing), updated, len(new_boxes), len(removed),
                len(self._catalog_options), (diff_ready-started)*1000,
                (updated_ready-diff_ready)*1000, (added_ready-updated_ready)*1000,
                (finished-added_ready)*1000, (finished-started)*1000)

    def set_presence(self, contains_pin: bool, contains_set: bool) -> None:
        self.pin_cb.SetValue(contains_pin)
        self.set_cb.SetValue(contains_set)

    def _on_pin_checked(self, event: wx.CommandEvent) -> None:
        if self._on_presence_change:
            self._on_presence_change('pin', self.pin_cb.GetValue())
        event.Skip()

    def _on_set_checked(self, event: wx.CommandEvent) -> None:
        if self._on_presence_change:
            self._on_presence_change('set', self.set_cb.GetValue())
        event.Skip()

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
        if self._on_interact:
            self._on_interact()
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
        if isinstance(event, BoxAddedEvent):
            if self.__get_existing_box_index(event.box) < 0:
                self.boxes = [*self.boxes, event.box]
        else:
            getLog().warning('TagPanel received an invalid box-added event')

    def __remove_box_panel(self, box: BoxData) -> None:
        filtered_panels = [
            panel for panel in self.__box_panels
            if panel.is_box(box)
        ]

        self.__remove_panels(filtered_panels)

    def __on_box_removed(self, event: BoxRemovedEvent) -> None:
        """Handle box removed event."""
        try:
            self.boxes = [box for box in self.boxes if not box.matches(event.box)]
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
        event.Skip()

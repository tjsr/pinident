from typing import Callable, List

import wx

from boxdata import BoxData, has_specific_identity
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
        box: BoxData,
        on_confirm: Callable[[BoxData], None] | None = None,
        on_remove: Callable[[BoxData], None] | None = None,
        on_interact: Callable[[], None] | None = None,
        on_change_begin: Callable[[], None] | None = None,
        on_change_end: Callable[[], None] | None = None,
        catalog_options: list[tuple[str, str]] | None = None,
        on_confirm_identity: Callable[[BoxData], None] | None = None,
    ):
        super().__init__(parent)
        self.__is_selected: bool = False
        self._on_confirm = on_confirm
        self._on_confirm_identity = on_confirm_identity
        self._on_remove = on_remove
        self._on_interact = on_interact
        self._on_change_begin = on_change_begin
        self._on_change_end = on_change_end
        self._catalog_options = catalog_options or []
        self._last_render_signature = None
        self.__heading_text = wx.StaticText(self, label=f"Box: {box.coords}")
        self.__heading_text.SetMinSize(wx.Size(240, -1))
        self.__sizer = wx.BoxSizer(wx.VERTICAL)
        self.__sizer.Add(self.__heading_text, 0, wx.EXPAND | wx.ALL, 5)
        self.__tag_sizer = wx.BoxSizer(wx.VERTICAL)
        self.__sizer.Add(self.__tag_sizer, 0, wx.EXPAND)

        actions = wx.BoxSizer(wx.HORIZONTAL)
        self.confirm_button = wx.BitmapButton(
            self, bitmap=wx.ArtProvider.GetBitmap(wx.ART_TICK_MARK, wx.ART_BUTTON))
        self.confirm_button.SetToolTip('Confirm this frame contains this pin')
        self.confirm_button.Bind(wx.EVT_BUTTON, self._confirm)
        actions.Add(self.confirm_button, 0, wx.ALL, 3)
        self.confirm_identity_button = wx.Button(self, label='Confirm identity')
        self.confirm_identity_button.SetToolTip('Confirm this pin identity in this frame')
        self.confirm_identity_button.Bind(wx.EVT_BUTTON, self._confirm_identity)
        actions.Add(self.confirm_identity_button, 0, wx.ALL, 3)
        self.delete_button = wx.BitmapButton(
            self, bitmap=wx.ArtProvider.GetBitmap(wx.ART_DELETE, wx.ART_BUTTON))
        self.delete_button.SetToolTip('Remove area')
        self.delete_button.Bind(wx.EVT_BUTTON, self._remove)
        actions.Add(self.delete_button, 0, wx.ALL, 3)
        self.__sizer.Add(actions, 0, wx.ALIGN_LEFT)

        self.__add_button = wx.Button(self, label="Add")
        self.__add_button.Bind(wx.EVT_BUTTON, self.__on_add_tag)
        self.Bind(EVT_BOX_EDITED, self.update_after_edited)
        self.__sizer.Add(self.__add_button, 0, wx.ALIGN_LEFT | wx.ALL, 5)

        self.SetSizer(self.__sizer)

        self.box = box


    def Refresh(self, eraseBackground=True, rect=None) -> None:
        log = getLog()
        tag_index: int = 0
        tag_count: int = max(1, len(self.__box.tags or []))

        state = ('Identity confirmed in another frame' if self.__box.identity_confirmed and
                 self.__box.review_state == 'inherited' else
                 'Identity confirmed' if self.__box.identity_confirmed and
                 self.__box.review_state == 'confirmed' else
                 'Pin confirmed' if self.__box.review_state == 'confirmed' else
                 'Pin confirmed in another frame' if self.__box.review_state == 'inherited' else
                 self.__box.review_state.title())
        strength = (f' · Match {self.__box.match_confidence:.0%}'
                    if self.__box.pin_id and not self.__box.identity_confirmed
                    and self.__box.match_confidence is not None else '')
        self.__heading_text.SetLabelText(f"{state}{strength} — Box: {self.__box.coords}")

        # current_tags = []

        while self.__tag_sizer.GetItemCount():
            self.__tag_sizer.Detach(0)
        for stale_row in self.__box_tags[tag_count:]:
            stale_row.Destroy()
        del self.__box_tags[tag_count:]

        while tag_index < tag_count:
            # log.debug(f'Repainting tag {tag_index+1}/{tag_count} on {self}')
            tag = self.repaint_tag(tag_index)
            # log.debug(f'Repainted tag {tag_index+1}/{tag_count} on {self} with label: {tag.tag}')
            tag_index += 1
            self.__tag_sizer.Add(tag, 0, wx.EXPAND | wx.ALL, 2)

        self.confirm_button.Enable(self.__box.review_state != 'confirmed')
        inherited_identity = (self.__box.review_state == 'inherited' and
                              self.__box.identity_confirmed)
        self.confirm_identity_button.SetLabel(
            'Verify here (optional)' if inherited_identity else 'Confirm identity')
        self.confirm_identity_button.SetToolTip(
            'Identity is already verified through an adjacent frame. '
            'Use this only to add a direct review in this frame.'
            if inherited_identity else 'Confirm this pin identity in this frame')
        self.confirm_identity_button.Enable(
            has_specific_identity(self.__box) and
            not (self.__box.review_state == 'confirmed' and
                 self.__box.identity_confirmed and not self.__box.identity_inferred))

        if not self.GetParent().IsFrozen():
            self.Layout()
        self._last_render_signature = self._box_signature(self.__box)
        super().Refresh(eraseBackground, rect)

    @staticmethod
    def _box_signature(box: BoxData) -> tuple:
        return (box.coords, tuple(box.tags or []), box.pin_id, box.review_state,
                box.identity_confirmed, box.match_confidence, box.identity_inferred)

    def needs_refresh(self, box: BoxData) -> bool:
        return self._last_render_signature != self._box_signature(box)

    def _confirm(self, _event: wx.CommandEvent) -> None:
        if self._on_interact:
            self._on_interact()
        if self._on_confirm:
            self._on_confirm(self.__box)

    def _confirm_identity(self, _event: wx.CommandEvent) -> None:
        if self._on_interact:
            self._on_interact()
        if self._on_confirm_identity:
            self._on_confirm_identity(self.__box)

    def _remove(self, _event: wx.CommandEvent) -> None:
        if self._on_interact:
            self._on_interact()
        if self._on_remove:
            self._on_remove(self.__box)


    def repaint_tag(self, tag_index: int) -> BoxTagLabelRow:
        tag = (self.__box.tags[tag_index]
               if tag_index < len(self.__box.tags or []) else '')
        if not isinstance(tag, str):
            raise ValueError(f"Invalid tag type: {type(tag)}. Expected str.")

        # box_label_count: int = len(self.__box_labels)
        current_label_row: BoxTagLabelRow = self.get_or_create_label(tag_index)

        if tag_index >= len(self.__box_tags):
            self.__box_tags.append(current_label_row)
        else:
            self.__box_tags[tag_index] = current_label_row

        is_only_row = len(self.__box.tags) == 1

        current_label_row.set_tag(tag, is_only_row, self.__box.pin_id if tag_index == 0 else None)
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

        new_label = BoxTagLabelRow(self, self.__box, index, self._catalog_options)
        new_label.Bind(EVT_BOX_LABEL_REMOVE, self.__on_tag_remove)
        new_label.Bind(EVT_BOX_LABEL_EDITED, self.__on_label_edited)
        new_label.Bind(wx.EVT_PAINT, self.__on_label_repainted)

        return new_label

    def __on_label_edited(self, event: BoxLabelEditedEvent) -> None:
        """Handle label updates."""
        getLog().debug(f'{event.label_index}={event.new_label}')
        current = (self.__box.get_tag(event.label_index)
                   if event.label_index < len(self.__box.tags or []) else '')
        if current == event.new_label and self.__box.pin_id == event.pin_id:
            return
        if self._on_change_begin:
            self._on_change_begin()
        if event.label_index < len(self.__box.tags or []):
            self.__box.set_tag(event.label_index, event.new_label)
        else:
            self.__box.add_tag(event.new_label)
        self.__box.pin_id = event.pin_id
        self.__box.match_confidence = None
        self.__box.identity_confirmed = False
        self.__box.identity_inferred = False
        if self._on_change_end:
            self._on_change_end()
        boxUpdateEvent = BoxEditedEvent(self, self.__box)
        wx.PostEvent(self, boxUpdateEvent)

    def set_catalog_options(self, options: list[tuple[str, str]]) -> None:
        self._catalog_options = options
        for row in self.__box_tags:
            row.set_catalog_options(options)
        self.Refresh()

    def __on_add_tag(self, event: wx.CommandEvent) -> None:
        # Add a new empty tag
        if self._on_change_begin:
            self._on_change_begin()
        if not self.__box.tags:
            self.__box.tags = ['', '']
        else:
            self.__box.tags.append("")
        self.__box.pin_id = None
        self.__box.match_confidence = None
        self.__box.identity_confirmed = False
        self.__box.identity_inferred = False
        if self._on_change_end:
            self._on_change_end()
        boxEditEvent = BoxEditedEvent(event.GetEventObject(), self.__box)
        wx.PostEvent(self, boxEditEvent)

    def __on_label_repainted(self, event: wx.PaintEvent) -> None:
        # getLog().debug(f'Repainting label row: {self}->{event.GetEventObject()}')
        pass

    def __on_tag_remove(self, event: BoxLabelRemoveEvent) -> None:
        """Handle tag removal."""
        if event.label_index < 0 or event.label_index >= len(self.__box.tags):
            return

        if self._on_change_begin:
            self._on_change_begin()
        self.__box.remove_tag(event.label_index)
        self.__box.pin_id = None
        self.__box.match_confidence = None
        self.__box.identity_confirmed = False
        self.__box.identity_inferred = False
        if self._on_change_end:
            self._on_change_end()
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
        if not hasattr(self, '_BoxTagPanelEdit__box_tags'):
            self.__box_tags = []
        self.__box = value
        if self.needs_refresh(value):
            self.Refresh()

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


"""Isolated saved-rectangle review UI for the desktop editor."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import wx

from boxdata import rejects_identity
from classifier_review import ReviewArea, review_actions, review_kind
from pin_catalog import PinMatch

if TYPE_CHECKING:
    from scrubberframe import ScrubberFrame


_KIND_LABELS = {
    'candidate': 'Candidate area',
    'suggested_candidate': 'Candidate with a provisional pin suggestion',
    'confirmed_pin': 'Pin present; identity needs verification',
    'linked_pin': 'Pin linked from another frame; verify this view if needed',
    'confirmed_identity': 'Pin identity directly confirmed in this frame',
    'rejected': 'Previously reviewed as not a pin',
}

_ACTION_TOOLTIPS = {
    'confirm_identity': (
        'Confirm the selected catalog result as this rectangle\'s pin identity. '
        'An exact name or ID typed in the search field also selects that pin. '
        'With no selection or search, confirm the saved suggested identity. '
        'Correcting a suggestion records the old identity as a wrong-ID review.'),
    'confirm_pin': (
        'Confirm that this rectangle contains a pin in this frame, without '
        'confirming its suggested name or catalog ID. This records a positive '
        'pin crop for training; any identity suggestion remains provisional. '
        'The next area opens immediately; annotations and feedback save in the '
        'background after a short pause. File → Save or closing the editor '
        'flushes pending changes.'),
    'restore_pin': (
        'Correct a rectangle previously marked Not a pin. This replaces its '
        'negative review with a positive pin crop, clears the old pin identity '
        'guess, and separates it from the rejected track so identity matching '
        'can try again. The next area opens immediately; annotations and feedback '
        'save in the background after a short pause. File → Save or closing the '
        'editor flushes pending changes.'),
}


class ClassifierReviewDialog(wx.Dialog):
    """Browse saved crops while the owner applies each review in memory."""

    def __init__(self, owner: ScrubberFrame, areas: list[ReviewArea],
                 catalog_options: list[tuple[str, str]]) -> None:
        filename = Path(owner.box_data_filename).name if owner.box_data_filename else 'annotations'
        super().__init__(owner, title=f'Review saved areas — {filename}',
                         size=wx.Size(780, 850), style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER)
        self._owner = owner
        self._areas = areas
        self._catalog_options = catalog_options
        self._position = 0
        self._current_area: ReviewArea | None = None
        self._choices: list[PinMatch] = []
        self._ranked: list[PinMatch] = []
        self.Bind(wx.EVT_CHAR_HOOK, self._on_key)

        panel = wx.Panel(self)
        layout = wx.BoxSizer(wx.VERTICAL)
        self._position_label = wx.StaticText(panel)
        self._kind_label = wx.StaticText(panel)
        self._current_label = wx.StaticText(panel)
        self._best_label = wx.StaticText(panel)
        for label in (self._position_label, self._kind_label,
                      self._current_label, self._best_label):
            layout.Add(label, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.TOP, 8)

        self._preview = wx.StaticBitmap(panel, bitmap=wx.Bitmap(1, 1))
        self._preview.SetMinSize(wx.Size(520, 400))
        layout.Add(self._preview, 1, wx.ALIGN_CENTER | wx.ALL, 8)
        layout.Add(wx.StaticText(
            panel, label='Percentages are visual match strengths, not identity certainty.'),
            0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 8)

        action_row = wx.BoxSizer(wx.HORIZONTAL)
        self._action_buttons: dict[str, wx.Button] = {}
        for action, label in (
                ('confirm_identity', 'Confirm identity (Y)'),
                ('reject_identity', 'Reject identity (N)'),
                ('confirm_pin', 'Pin only (P)'),
                ('not_pin', 'Not a pin (X)'),
                ('restore_pin', 'Restore as pin (R)')):
            button = wx.Button(panel, label=label)
            button.Bind(wx.EVT_BUTTON, partial(self._act, action))
            if action in _ACTION_TOOLTIPS:
                button.SetToolTip(_ACTION_TOOLTIPS[action])
            self._action_buttons[action] = button
            action_row.Add(button, 0, wx.RIGHT, 5)
        layout.Add(action_row, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 8)

        layout.Add(wx.StaticText(panel, label='Choose a catalog pin for this crop:'),
                   0, wx.LEFT | wx.RIGHT, 8)
        self._search = wx.TextCtrl(panel)
        self._search.SetHint('Search by name or pin ID')
        self._search.Bind(wx.EVT_TEXT, self._on_search)
        layout.Add(self._search, 0, wx.EXPAND | wx.ALL, 8)
        self._results = wx.ListBox(panel)
        self._results.Bind(wx.EVT_LISTBOX_DCLICK, self._on_choose)
        layout.Add(self._results, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 8)
        choose = wx.Button(panel, label='Confirm selected catalog pin')
        choose.Bind(wx.EVT_BUTTON, self._on_choose)
        layout.Add(choose, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 8)
        self._action_buttons['choose_identity'] = choose

        navigation = wx.BoxSizer(wx.HORIZONTAL)
        self._previous = wx.Button(panel, label='Previous area')
        self._next = wx.Button(panel, label='Skip / Next area')
        close = wx.Button(panel, label='Close')
        self._previous.Bind(wx.EVT_BUTTON, self._on_previous)
        self._next.Bind(wx.EVT_BUTTON, self._on_next)
        close.Bind(wx.EVT_BUTTON, self._on_close)
        navigation.Add(self._previous, 0, wx.RIGHT, 8)
        navigation.Add(self._next, 0, wx.RIGHT, 8)
        navigation.AddStretchSpacer()
        navigation.Add(close)
        layout.Add(navigation, 0, wx.EXPAND | wx.ALL, 8)
        self._status = wx.StaticText(panel)
        layout.Add(self._status, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 8)
        panel.SetSizer(layout)
        outer = wx.BoxSizer(wx.VERTICAL)
        outer.Add(panel, 1, wx.EXPAND)
        self.SetSizer(outer)
        self._show_position()

    def _show_position(self) -> None:
        while self._position < len(self._areas):
            area = self._areas[self._position]
            if area.box.review_state == 'rejected' and any(
                    later.box.review_state != 'rejected'
                    for later in self._areas[self._position + 1:]):
                # A review can reject linked future observations after the
                # initial queue was built. Move those behind remaining work.
                self._areas.append(self._areas.pop(self._position))
                continue
            crop = self._owner.classifier_review_open(area)
            if crop is not None:
                self._current_area = area
                self._render(area, crop)
                return
            self._position += 1
        self._current_area = None
        self._position_label.SetLabel(f'Review complete: {len(self._areas)} saved areas')
        self._kind_label.SetLabel('')
        self._current_label.SetLabel('')
        self._best_label.SetLabel('')
        self._results.Clear()
        for button in self._action_buttons.values():
            button.Disable()
        self._next.Disable()
        self._previous.Enable(bool(self._areas))
        self._status.SetLabel('All saved areas have been shown. Close or go back.')

    def _render(self, area: ReviewArea, crop: np.ndarray) -> None:
        box = area.box
        self._position_label.SetLabel(
            f'Area {self._position + 1:,} of {len(self._areas):,}  |  '
            f'Frame {area.frame_index}  |  Rectangle {box.coords}')
        self._kind_label.SetLabel(_KIND_LABELS[review_kind(box)])
        name = ', '.join(box.tags or []) or 'Unidentified'
        pin_id = f' [{box.pin_id}]' if box.pin_id else ''
        score = (f' ({box.match_confidence:.0%} match strength)'
                 if box.match_confidence is not None else '')
        self._current_label.SetLabel(f'Saved annotation: {name}{pin_id}{score}')
        self._ranked = self._owner.classifier_review_rank(crop, box)
        best = self._ranked[0] if self._ranked else None
        self._best_label.SetLabel(
            f'Model best visual match: {best.name} [{best.pin_id}] '
            f'({best.confidence:.0%} match strength)'
            if best is not None and best.confidence is not None else
            'Model best visual match: no eligible catalog reference')
        image = wx.Image(crop.shape[1], crop.shape[0], crop.tobytes())
        scale = min(520 / crop.shape[1], 400 / crop.shape[0])
        width = max(1, round(crop.shape[1] * scale))
        height = max(1, round(crop.shape[0] * scale))
        self._preview.SetBitmap(wx.Bitmap(image.Scale(width, height,
                                                      wx.IMAGE_QUALITY_HIGH)))
        available = review_actions(box)
        for action, button in self._action_buttons.items():
            button.Enable(action in available)
        self._previous.Enable(self._position > 0)
        self._next.Enable()
        self._search.ChangeValue('')
        self._refresh_choices()
        self._status.SetLabel('Choose a review action to advance, or Skip. '
                              'Reviews save in the background after a short pause.')
        self.Layout()

    def _refresh_choices(self) -> None:
        area = self._current_area
        if area is None:
            return
        query = self._search.GetValue().strip().casefold()
        if query:
            self._choices = [
                PinMatch(pin_id, name, 0)
                for pin_id, name in self._catalog_options
                if (query in pin_id.casefold() or query in name.casefold())
                and not rejects_identity(area.box, pin_id, [name])][:75]
        else:
            self._choices = list(self._ranked)
        self._results.SetItems([
            f'{match.confidence:.0%}  {match.name} [{match.pin_id}]'
            if match.confidence is not None else
            f'{match.name} [{match.pin_id}]'
            for match in self._choices])

    def _on_search(self, _event: wx.CommandEvent) -> None:
        self._refresh_choices()

    def _on_choose(self, _event: wx.CommandEvent) -> None:
        match = self._selected_identity()
        if match is not None:
            self._act('choose_identity', match=match)

    def _selected_identity(self) -> PinMatch | None:
        selected = self._results.GetSelection()
        if 0 <= selected < len(self._choices):
            return self._choices[selected]
        query = self._search.GetValue().strip().casefold()
        if not query:
            return None
        exact = [match for match in self._choices
                 if query in (match.name.casefold(), match.pin_id.casefold())]
        return exact[0] if len(exact) == 1 else None

    def _act(self, action: str, _event: wx.CommandEvent | None = None,
             *, match: PinMatch | None = None) -> None:
        area = self._current_area
        if area is None or action not in review_actions(area.box):
            return
        box = area.box
        selected_identity = self._selected_identity() if action == 'confirm_identity' else None
        if (action == 'confirm_identity' and selected_identity is None and
                self._search.GetValue().strip()):
            self._status.SetLabel('Select a catalog result before confirming this search.')
            return
        with self._owner.classifier_review_action():
            if action == 'confirm_identity':
                if selected_identity is not None:
                    self._owner.replace_with_pin(box, selected_identity)
                else:
                    self._owner.confirm_pin_identity(box)
            elif action == 'reject_identity':
                if not self._owner.classifier_review_reject_identity(box):
                    return
            elif action == 'confirm_pin':
                self._owner.confirm_box(box)
            elif action == 'not_pin':
                self._owner.mark_not_pin(box)
            elif action == 'restore_pin':
                if not self._owner.classifier_review_restore_pin(box):
                    return
            elif action == 'choose_identity' and match is not None:
                self._owner.replace_with_pin(box, match)
            else:
                return
        self._owner.queue_review_save()
        self._position += 1
        self._show_position()

    def _on_previous(self, _event: wx.CommandEvent) -> None:
        if self._position > 0:
            self._position -= 1
            self._show_position()

    def _on_next(self, _event: wx.CommandEvent) -> None:
        self._position += 1
        self._show_position()

    def _on_key(self, event: wx.KeyEvent) -> None:
        if isinstance(wx.Window.FindFocus(), wx.TextCtrl):
            event.Skip()
            return
        key = event.GetKeyCode()
        if ord('a') <= key <= ord('z'):
            key -= ord('a') - ord('A')
        actions = {ord('Y'): 'confirm_identity', ord('N'): 'reject_identity',
                   ord('P'): 'confirm_pin', ord('X'): 'not_pin',
                   ord('R'): 'restore_pin'}
        action = actions.get(key)
        if action is not None:
            self._act(action)
        elif key == wx.WXK_RIGHT:
            self._on_next(wx.CommandEvent())
        elif key == wx.WXK_LEFT:
            self._on_previous(wx.CommandEvent())
        else:
            event.Skip()

    def _on_close(self, _event: wx.CommandEvent) -> None:
        self.EndModal(wx.ID_OK)

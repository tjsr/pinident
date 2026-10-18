"""Visible, approximate progress for desktop shutdown."""

from time import perf_counter

import wx


STEPS = (
    ('Stop playback and prefetch', 1.0),
    ('Finish background work', 10.0),
    ('Write annotation JSON', 2.0),
    ('Release readers and finish', 1.0),
)


def remaining_seconds(estimate: float, elapsed: float, percent: int) -> float:
    """Use measured throughput when available; keep an honest moving guess otherwise."""
    if percent >= 100:
        return 0.0
    if percent > 0:
        return max(0.0, elapsed * (100 - percent) / percent)
    return max(estimate - elapsed, elapsed / 2, 1.0)


def format_duration(seconds: float) -> str:
    rounded = max(0, round(seconds))
    if rounded < 60:
        return f'~{rounded}s'
    minutes, seconds = divmod(rounded, 60)
    return f'~{minutes}m {seconds:02d}s'


class ShutdownProgressDialog(wx.Dialog):
    def __init__(self, parent: wx.Window):
        super().__init__(parent, title='Closing Pinident',
                         style=wx.DEFAULT_DIALOG_STYLE)
        self._started = perf_counter()
        self._step_started = self._started
        self._step_index = 0
        self._percent = 0
        panel = wx.Panel(self)
        layout = wx.BoxSizer(wx.VERTICAL)
        heading = wx.StaticText(panel, label='Finishing before exit')
        layout.Add(heading, 0, wx.ALL, 12)
        self._rows = []
        for name, _estimate in STEPS:
            row = wx.StaticText(panel, label='')
            layout.Add(row, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 12)
            self._rows.append(row)
        self._gauge = wx.Gauge(panel, range=100)
        layout.Add(self._gauge, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 12)
        self._overall = wx.StaticText(panel, label='')
        layout.Add(self._overall, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 12)
        panel.SetSizer(layout)
        outer = wx.BoxSizer(wx.VERTICAL)
        outer.Add(panel, 1, wx.EXPAND)
        self.SetSizerAndFit(outer)
        self.SetMinSize((440, -1))
        self.Bind(wx.EVT_CLOSE, self._prevent_close)
        self.refresh()

    def _prevent_close(self, event: wx.CloseEvent) -> None:
        if event.CanVeto():
            event.Veto()

    def advance(self) -> None:
        self._step_index = min(self._step_index + 1, len(STEPS))
        self._step_started = perf_counter()
        self._percent = 0
        self.refresh()

    def set_percent(self, percent: int) -> None:
        self._percent = max(self._percent, min(100, max(0, percent)))
        self.refresh()

    def refresh(self) -> None:
        now = perf_counter()
        remaining = 0.0
        for index, (name, estimate) in enumerate(STEPS):
            if index < self._step_index:
                label = f'✓ {name} — 100% · 0s left'
            elif index == self._step_index:
                eta = remaining_seconds(estimate, now - self._step_started,
                                        self._percent)
                remaining += eta
                label = f'▶ {name} — {self._percent}% · {format_duration(eta)} left'
            else:
                remaining += estimate
                label = f'○ {name} — 0% · {format_duration(estimate)} estimated'
            self._rows[index].SetLabel(label)
        completed = min(self._step_index, len(STEPS))
        overall_percent = round(100 * (completed + (self._percent / 100 if
                               completed < len(STEPS) else 0)) / len(STEPS))
        self._gauge.SetValue(overall_percent)
        self._overall.SetLabel(
            f'Overall: {overall_percent}% · {format_duration(remaining)} remaining '
            f'· {format_duration(now - self._started)} elapsed (approximate)')
        self.Layout()

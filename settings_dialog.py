"""Read-only locations and status for the desktop editor's local data."""

from datetime import datetime
from pathlib import Path

import wx


def storage_location(path: str | Path | None) -> tuple[str, str]:
    """Describe the currently selected local file without creating it."""
    if path is None:
        return 'No annotation file selected', 'Not saved yet'
    if str(path) == ':memory:':
        return 'In memory (this session only)', 'No file on disk'
    file_path = Path(path).resolve()
    if not file_path.is_file():
        return str(file_path), 'File not created yet'
    modified = datetime.fromtimestamp(file_path.stat().st_mtime)
    return str(file_path), f'Last modified: {modified:%Y-%m-%d %H:%M:%S}'


class SettingsDialog(wx.Dialog):
    def __init__(self, parent: wx.Window, *, feedback_path: str | Path,
                 catalog_path: str | Path, annotations_path: str | Path | None,
                 feedback_counts: dict[str, int], wrong_identity_count: int,
                 model_revision: int, model_path: str | Path | None = None,
                 model_active: bool = False, model_current: bool = False,
                 catalog_training_summary: str = '',
                 cloud_origin: str | None = None, cloud_status: str = 'Not checked',
                 cloud_cache_path: str | Path | None = None):
        super().__init__(parent, title='Settings', size=(740, 660),
                         style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER)
        layout = wx.BoxSizer(wx.VERTICAL)
        layout.Add(wx.StaticText(self, label='Data files used by this window'),
                   0, wx.ALL, 12)

        self.locations: dict[str, wx.TextCtrl] = {}
        self.statuses: dict[str, wx.StaticText] = {}
        for key, title, path, description in (
            ('feedback', 'Reviewed training examples', feedback_path,
             'Pin/non-pin and wrong-identity image crops are stored in this SQLite database.'),
            ('catalog', 'Known pin catalog', catalog_path,
             'Reference images and Pinpanion metadata are stored in this SQLite database.'),
            ('model', 'OpenCV pin/non-pin classifier', model_path,
             'Trained SVM parameters loaded by OpenCV to classify detector proposals.'),
            ('model_metadata', 'Classifier training metadata',
             Path(str(model_path) + '.json') if model_path is not None else None,
             'Feature version, training counts, original stock-image name tags, and provenance.'),
            ('cloud_model', 'Cached shared classifier', cloud_cache_path,
             'Downloaded portable model used when it is newer than the local classifier.'),
            ('annotations', 'Video annotations', annotations_path,
             'Frame boxes and direct review decisions are saved in this JSON file.'),
        ):
            location, status = (storage_location(path) if path is not None or
                                key == 'annotations' else
                                ('No persistent feedback database', 'No classifier file'))
            layout.Add(wx.StaticText(self, label=title), 0, wx.LEFT | wx.RIGHT | wx.TOP, 12)
            field = wx.TextCtrl(self, value=location, style=wx.TE_READONLY)
            layout.Add(field, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.TOP, 12)
            self.locations[key] = field
            detail = wx.StaticText(self, label=f'{status}. {description}')
            detail.Wrap(690)
            layout.Add(detail, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.TOP, 12)
            self.statuses[key] = detail

        count_text = (
            f"Example matcher revision {model_revision} (in memory). "
            f"Saved pin crops: {feedback_counts['positive']}; "
            f"non-pin crops: {feedback_counts['negative']}; "
            f"wrong-identity crops: {wrong_identity_count}. "
            f"OpenCV classifier: {'active' if model_active else 'not loaded'}"
            + (', training data current.' if model_active and model_current else
               ', training inputs changed since training.' if model_active else '.')
            + ' The feedback ranker refreshes in memory after reviews.'
            + (f' {catalog_training_summary}' if catalog_training_summary else '')
        )
        summary = wx.StaticText(self, label=count_text)
        summary.Wrap(690)
        layout.Add(summary, 0, wx.EXPAND | wx.ALL, 12)
        if cloud_origin:
            cloud = wx.StaticText(self, label=(
                f'Shared model: {cloud_origin}/model.json\n'
                f'Cloud sync: {cloud_status}\n'
                'Desktop upload/download authorization: PINIDENT_SYNC_TOKEN environment variable.'))
            cloud.Wrap(690)
            layout.Add(cloud, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 12)
        layout.AddStretchSpacer()
        close = self.CreateButtonSizer(wx.OK)
        layout.Add(close, 0, wx.ALIGN_RIGHT | wx.ALL, 12)
        self.SetSizer(layout)
        self.SetMinSize((740, 600))
        self.Fit()

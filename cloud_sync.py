"""Cloudflare review and portable classifier synchronization.

HTTP is performed off the wx thread. Call apply_reviews only on the owner
thread of FeedbackStore's SQLite connection.
"""

from __future__ import annotations

import base64
import json
import os
import logging
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import cv2
import numpy as np

from detection_feedback import FeedbackExample, FeedbackStore, make_feedback_example
from trained_detection_model import PortableDetectionModel


DEFAULT_ORIGIN = 'https://id.pinpanion.com'
SOURCE = 'cloudflare:web'
LOG = logging.getLogger(__name__)


class CloudSyncError(RuntimeError):
    """A remote HTTP or network failure while synchronizing with Cloudflare."""


@dataclass(frozen=True)
class CloudConfig:
    origin: str
    token: str | None

    @classmethod
    def from_environment(cls) -> 'CloudConfig':
        origin = os.environ.get('PINIDENT_SYNC_URL', DEFAULT_ORIGIN).rstrip('/')
        if not (origin.startswith('https://') or origin.startswith('http://localhost:') or
                origin.startswith('http://127.0.0.1:')):
            raise ValueError('PINIDENT_SYNC_URL must use HTTPS (or local development)')
        return cls(origin, os.environ.get('PINIDENT_SYNC_TOKEN') or None)


def _json_request(config: CloudConfig, path: str, *, method: str = 'GET',
                  payload: dict | None = None, auth: bool = False) -> tuple[object, dict[str, str]]:
    if auth and not config.token:
        raise ValueError('Set PINIDENT_SYNC_TOKEN to synchronize shared training data')
    body = json.dumps(payload, separators=(',', ':'), allow_nan=False).encode() if payload else None
    headers = {'Accept': 'application/json',
               'User-Agent': 'Pinident/0.2 (+desktop sync)'}
    if body is not None:
        headers['Content-Type'] = 'application/json'
    if auth:
        headers['Authorization'] = f'Bearer {config.token}'
    request = Request(config.origin + path, data=body, headers=headers, method=method)
    try:
        with urlopen(request, timeout=15) as response:
            if response.headers.get_content_type() != 'application/json':
                raise ValueError('Cloudflare returned a non-JSON response')
            raw = response.read(6_000_001)
            if len(raw) > 6_000_000:
                raise ValueError('Cloudflare response is too large')
            return json.loads(raw), dict(response.headers)
    except HTTPError as exc:
        if exc.code == 404 and path == '/model.json':
            return None, {}
        raise CloudSyncError(f'Cloudflare sync failed (HTTP {exc.code})') from exc
    except URLError as exc:
        raise CloudSyncError(f'Cloudflare sync is unavailable: {exc.reason}') from exc


def fetch_model(config: CloudConfig) -> tuple[int, PortableDetectionModel] | None:
    payload, headers = _json_request(config, '/model.json')
    if payload is None:
        return None
    version = int(headers.get('X-Model-Version', '0'))
    if version < 0:
        raise ValueError('Cloudflare model has an invalid version')
    return version, PortableDetectionModel.from_payload(payload)


def fetch_review_page(config: CloudConfig, after: int) -> tuple[list[dict], int]:
    payload, _headers = _json_request(config,
        '/api/training-reviews?' + urlencode({'after': after}), auth=True)
    if not isinstance(payload, dict) or not isinstance(payload.get('reviews'), list):
        raise ValueError('Invalid shared review page')
    next_cursor = payload.get('next')
    if not isinstance(next_cursor, int) or next_cursor < after:
        raise ValueError('Invalid shared review cursor')
    if payload['reviews'] and next_cursor == after:
        raise ValueError('Shared review page did not advance')
    return payload['reviews'], next_cursor


def publish_model(config: CloudConfig, model_payload: dict, base_version: int) -> int:
    result, _headers = _json_request(config, '/api/model', method='PUT', auth=True,
        payload={'base_version': base_version, 'model': model_payload})
    if not isinstance(result, dict) or not isinstance(result.get('version'), int):
        raise ValueError('Invalid published model version')
    return result['version']


def cloud_cursor(store: FeedbackStore) -> int:
    store.db.execute('''CREATE TABLE IF NOT EXISTS cloud_sync_state
        (key TEXT PRIMARY KEY, value TEXT NOT NULL)''')
    row = store.db.execute(
        "SELECT value FROM cloud_sync_state WHERE key = 'review_cursor'").fetchone()
    return int(row[0]) if row else 0


def _example(review: dict) -> FeedbackExample:
    if review.get('kind') != 'training' or review.get('label') not in (
            'not_pin', 'pin_unknown', 'pin_identified'):
        raise ValueError('Unsupported shared training review')
    review_id = review.get('id')
    image = review.get('image_data_url')
    if not isinstance(review_id, str) or not isinstance(image, str) or \
            not image.startswith(('data:image/jpeg;base64,', 'data:image/png;base64,')):
        raise ValueError('Invalid shared training review image')
    encoded = image.split(',', 1)[1]
    if len(encoded) > 500_000:
        raise ValueError('Shared training image is too large')
    bgr = cv2.imdecode(np.frombuffer(base64.b64decode(encoded, validate=True),
                                     dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None or bgr.size == 0 or bgr.shape[0] > 1024 or bgr.shape[1] > 1024:
        raise ValueError('Invalid shared training image')
    success, png = cv2.imencode('.png', bgr)
    if not success:
        raise ValueError('Could not encode shared training image')
    height, width = bgr.shape[:2]
    label = 'negative' if review['label'] == 'not_pin' else 'positive'
    pin_id = review.get('pin_id') if label == 'positive' else None
    return FeedbackExample(
        f'cloudflare:{review_id}', label, SOURCE,
        int(float(review.get('time_seconds') or 0) * 1000), review_id,
        pin_id if isinstance(pin_id, str) else None,
        (0, 0, width, height), png.tobytes(),
        (str(review.get('source') or ''),))


def _annotation_examples(annotation: dict) -> list[FeedbackExample]:
    annotation_id = annotation.get('id')
    image = annotation.get('image_data_url')
    boxes = annotation.get('boxes')
    if not isinstance(annotation_id, str) or not annotation_id or \
            not isinstance(image, str) or \
            not image.startswith(('data:image/jpeg;base64,', 'data:image/png;base64,')) or \
            not isinstance(boxes, list) or len(boxes) > 100:
        raise ValueError('Invalid shared frame annotation')
    encoded = image.split(',', 1)[1]
    if len(encoded) > 900_000:
        raise ValueError('Shared frame image is too large')
    bgr = cv2.imdecode(np.frombuffer(base64.b64decode(encoded, validate=True),
                                     dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None or bgr.size == 0 or bgr.shape[0] > 1024 or bgr.shape[1] > 1024 or \
            bgr.shape[1] != annotation.get('width') or bgr.shape[0] != annotation.get('height'):
        raise ValueError('Invalid shared frame image')
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    frame_index = int(float(annotation.get('time_seconds') or 0) * 1000)
    examples = []
    for index, box in enumerate(boxes):
        if not isinstance(box, dict) or box.get('review_state') not in ('confirmed', 'inherited'):
            continue
        coords = box.get('coords')
        if not isinstance(coords, list) or len(coords) != 4 or \
                any(type(value) is not int for value in coords):
            raise ValueError('Invalid shared frame box')
        track_id = box.get('track_id')
        if not isinstance(track_id, str) or not track_id:
            raise ValueError('Invalid shared frame track')
        pin_id = box.get('pin_id')
        if pin_id is not None and not isinstance(pin_id, str):
            raise ValueError('Invalid shared frame pin ID')
        examples.append(make_feedback_example(
            f'cloudflare:annotation:{annotation_id}:{index}', 'positive', rgb,
            tuple(coords), SOURCE, frame_index, track_id, pin_id,
            (str(annotation.get('source') or ''),)))
    return examples


def apply_reviews(store: FeedbackStore, events: list[dict], cursor: int) -> int:
    """Upsert changed IDs, then advance cursor in the same SQLite transaction."""
    changed = 0
    parsed: list[tuple[list[FeedbackExample], str | None]] = []
    for event in events:
        if not isinstance(event, dict) or not isinstance(event.get('revision'), int) or \
                event['revision'] > cursor or not isinstance(event.get('review'), dict):
            raise ValueError('Invalid shared review event')
        try:
            review = event['review']
            if review.get('kind') == 'annotation':
                examples = _annotation_examples(review)
                parsed.append((examples, f"cloudflare:annotation:{review['id']}:"))
            else:
                parsed.append(([_example(review)], None))
        except (ValueError, TypeError, OverflowError) as exc:
            LOG.warning('Skipping invalid shared review at revision %s: %s',
                        event['revision'], exc)
    with store.db:
        for examples, prefix in parsed:
            if prefix is not None:
                active_ids = {example.example_id for example in examples}
                existing = store.db.execute(
                    'SELECT example_id FROM feedback_examples WHERE substr(example_id, 1, ?) = ?',
                    (len(prefix), prefix)).fetchall()
                for (example_id,) in existing:
                    if example_id not in active_ids:
                        store.db.execute('DELETE FROM feedback_examples WHERE example_id = ?',
                                         (example_id,))
                        changed += 1
            for example in examples:
                old = store.get(example.example_id)
                if old != example:
                    store._upsert(example)
                    changed += 1
        store.db.execute('''CREATE TABLE IF NOT EXISTS cloud_sync_state
            (key TEXT PRIMARY KEY, value TEXT NOT NULL)''')
        store.db.execute('''INSERT INTO cloud_sync_state (key, value)
            VALUES ('review_cursor', ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value''',
            (str(cursor),))
    return changed

import json
import sqlite3
from time import monotonic, sleep

import cv2
import numpy as np
import pytest
import wx

from boxdata import BoxData
from pin_catalog import PinCatalog, PinMatcher
from pinpanion_sync import (FEED_URL, IMAGE_BASE_URL, parse_pin_records,
                            fetch_public_bytes, start_catalog_sync, sync_pinpanion_catalog)
from videoscrubber import VideoScrubber


def png_bytes(seed: int) -> tuple[np.ndarray, bytes]:
    rng = np.random.default_rng(seed)
    image = rng.integers(10, 245, (180, 180, 3), dtype=np.uint8)
    cv2.circle(image, (90, 90), 70, (240, 30, 20), 4)
    ok, encoded = cv2.imencode('.png', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    assert ok
    return image, encoded.tobytes()


def feed_bytes() -> bytes:
    return json.dumps({'pins': [
        {'id': 5, 'name': 'Pinny Arcade Logo', 'image_name': 'logo.webp',
         'categoryIds': [38], 'year': 2013, 'setId': None},
        {'id': 8, 'name': 'Merch', 'image_name': 'merch.webp',
         'categoryIds': [16, 38], 'year': 2013, 'setId': None},
    ]}).encode()


def test_sync_imports_new_pin_records_once_and_retains_metadata(tmp_path):
    catalog = PinCatalog(tmp_path / 'catalog.sqlite3')
    _, first_image = png_bytes(1)
    _, second_image = png_bytes(2)
    responses = {FEED_URL: feed_bytes(), IMAGE_BASE_URL + 'logo.webp': first_image,
                 IMAGE_BASE_URL + 'merch.webp': second_image}
    calls = []

    def fetch(url, limit):
        calls.append(url)
        return responses[url]

    first = sync_pinpanion_catalog(catalog, fetch)
    second = sync_pinpanion_catalog(catalog, fetch)

    assert (first.total, first.imported, first.failed) == (2, 2, 0)
    assert (second.existing, second.imported) == (2, 0)
    assert calls.count(IMAGE_BASE_URL + 'logo.webp') == 1
    assert calls.count(IMAGE_BASE_URL + 'merch.webp') == 1
    assert catalog.pin_ids() == {'pinpanion:5', 'pinpanion:8'}
    assert len(catalog.references()) == 2
    with sqlite3.connect(catalog.path) as db:
        raw, image_url = db.execute(
            'SELECT raw_json, image_url FROM pinpanion_metadata WHERE pin_id = ?',
            ('pinpanion:5',)).fetchone()
        source, rights = db.execute(
            'SELECT source, rights FROM pin_references WHERE pin_id = ?',
            ('pinpanion:5',)).fetchone()
    assert json.loads(raw)['categoryIds'] == [38]
    assert image_url == source == IMAGE_BASE_URL + 'logo.webp'
    assert rights == 'source terms unverified'


def test_transparent_pin_artwork_masks_background_features(tmp_path):
    image, _ = png_bytes(3)
    bgra = cv2.cvtColor(image, cv2.COLOR_RGB2BGRA)
    bgra[:, :, 3] = 0
    bgra[20:160, 20:160, 3] = 255
    ok, encoded = cv2.imencode('.webp', bgra)
    assert ok
    catalog = PinCatalog(tmp_path / 'catalog.sqlite3')
    catalog.import_pinpanion_pin({'id': 9, 'name': 'Masked', 'image_name': 'masked.webp'},
                                  encoded.tobytes(), IMAGE_BASE_URL + 'masked.webp')

    reference = catalog.references()[0]

    assert reference.mask is not None
    assert reference.mask[0, 0] == 0
    assert reference.mask[90, 90] == 255
    assert PinMatcher([reference]).match(reference.image).pin_id == 'pinpanion:9'


def test_failed_image_is_retried_without_partial_pin(tmp_path):
    catalog = PinCatalog(tmp_path / 'catalog.sqlite3')
    _, good_image = png_bytes(1)
    failing = {IMAGE_BASE_URL + 'merch.webp'}

    def fetch(url, limit):
        if url == FEED_URL:
            return feed_bytes()
        if url in failing:
            return b'not an image'
        return good_image

    first = sync_pinpanion_catalog(catalog, fetch)
    assert (first.imported, first.failed) == (1, 1)
    assert catalog.pin_ids() == {'pinpanion:5'}
    failing.clear()
    second = sync_pinpanion_catalog(catalog, fetch)
    assert (second.existing, second.imported, second.failed) == (1, 1, 0)


def test_startup_worker_rebuilds_model_after_import(tmp_path):
    catalog = PinCatalog(tmp_path / 'catalog.sqlite3')
    image, image_bytes = png_bytes(1)
    _, other_bytes = png_bytes(2)
    updates = []

    def fetch(url, limit):
        if url == FEED_URL:
            return feed_bytes()
        return image_bytes if url.endswith('logo.webp') else other_bytes

    worker = start_catalog_sync(catalog, updates.append, fetch)
    worker.join(timeout=10)

    assert not worker.is_alive()
    assert len(updates) == 2
    assert updates[0].match(np.zeros((180, 180, 3), np.uint8)) is None
    assert len(catalog.references()) == 2
    assert updates[1].match(image).pin_id == 'pinpanion:5'


@pytest.mark.parametrize('bad_name', ['../escape.png', '/absolute.png', 'nested/pin.png',
                                      'pin.png?other=1'])
def test_feed_rejects_unsafe_image_names(bad_name):
    record = {'id': 5, 'name': 'Pin', 'image_name': bad_name}
    with pytest.raises(ValueError):
        parse_pin_records(json.dumps({'pins': [record]}).encode())


def test_fetch_rejects_non_pinpanion_host_without_network():
    with pytest.raises(ValueError):
        fetch_public_bytes('https://example.com/pins.json', 100)


@pytest.mark.gui
def test_newly_built_model_identifies_an_existing_gui_candidate(tmp_path):
    app = wx.App.Get() or wx.App(False)
    image, image_bytes = png_bytes(1)
    catalog = PinCatalog(tmp_path / 'catalog.sqlite3')
    catalog.import_pinpanion_pin({'id': 5, 'name': 'Logo', 'image_name': 'logo.webp'},
                                  image_bytes, IMAGE_BASE_URL + 'logo.webp')
    frame = np.zeros((280, 300, 3), np.uint8)
    frame[35:215, 40:220] = image
    window = VideoScrubber(None, 'Sync test', '',
                           image_array=[cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)],
                           load_catalog=False)
    try:
        window.Show()
        wx.Yield()
        candidate = BoxData((40, 35, 180, 180), [], 'automatic')
        window._ScrubberFrame__frame_boxes[0] = [candidate]
        window.display_image()
        assert candidate.pin_id is None

        window.set_pin_matcher(PinMatcher(catalog.references()))
        deadline = monotonic() + 5
        while candidate.pin_id is None and monotonic() < deadline:
            wx.Yield()
            sleep(0.005)

        assert (candidate.pin_id, candidate.tags) == ('pinpanion:5', ['Logo'])
        assert window._ScrubberFrame__image_panel.boxes[0] is candidate
        assert window._ScrubberFrame__image_panel.get_box_colour(candidate) == wx.RED
    finally:
        window.Destroy()
        wx.Yield()

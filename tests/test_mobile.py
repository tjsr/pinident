import json
import http.client
import threading
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from mobile_scan import public_scan_result, scan_frame
from mobile_server import MobileService, decode_scan_image, make_server
from pin_catalog import PinCatalog, PinMatch, PinMatcher
from pinpanion_sync import FEED_URL, MAX_FEED_BYTES


def image_bytes(seed=1):
    image = np.random.default_rng(seed).integers(0, 255, (100, 150, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode('.png', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    assert ok
    return image, encoded.tobytes()


def make_catalog(tmp_path):
    catalog = PinCatalog(tmp_path / 'pins.sqlite3')
    _, data = image_bytes()
    catalog.import_pinpanion_pin({'id': 7, 'name': 'Test Pin', 'image_name': 'test.png'},
                                  data, 'https://pinpanion.com/imgs/test.png')
    return catalog


def test_catalog_search_and_exact_pin_lookup(tmp_path):
    catalog = make_catalog(tmp_path)
    assert catalog.search_pins('test')[0]['pin_id'] == 'pinpanion:7'
    assert catalog.search_pins('%') == []
    assert catalog.get_pin('pinpanion:7')['name'] == 'Test Pin'
    assert catalog.get_pin('pinpanion:8') is None


def test_mobile_scan_tracks_known_pin_without_repeating_match():
    image, _ = image_bytes()
    calls = []

    class Matcher:
        def match(self, crop):
            calls.append(crop.shape)
            return PinMatch('pinpanion:7', 'Test Pin', 10)

    with patch('mobile_scan.detect_pin_boxes', return_value=[(10, 15, 40, 40)]):
        first = scan_frame(image, Matcher())
        second = scan_frame(image.copy(), Matcher(), first)
    assert len(calls) == 1
    assert second.boxes[0].track_id == first.boxes[0].track_id
    assert public_scan_result(second)['boxes'][0]['pin_id'] == 'pinpanion:7'


def test_mobile_service_tags_and_rejects_a_scan_session(tmp_path):
    catalog = make_catalog(tmp_path)
    service = MobileService(catalog, PinMatcher([]))
    _, data = image_bytes()
    with patch('mobile_scan.detect_pin_boxes', return_value=[(10, 15, 40, 40)]):
        first = service.scan(data, 'camera-session')
        track_id = first['boxes'][0]['track_id']
        tagged = service.tag('camera-session', track_id, 'pinpanion:7')
        assert tagged['boxes'][0]['review_state'] == 'confirmed'
        assert tagged['boxes'][0]['name'] == 'Test Pin'
        next_frame = service.scan(data, 'camera-session')
        assert next_frame['boxes'][0]['review_state'] == 'inherited'
        assert next_frame['boxes'][0]['track_id'] == track_id

        other = service.scan(data, 'other-session')
        rejected_id = other['boxes'][0]['track_id']
        assert service.reject('other-session', rejected_id)['boxes'] == []
        assert service.scan(data, 'other-session')['boxes'] == []

    with pytest.raises(ValueError, match='not in the catalog'):
        service.tag('camera-session', track_id, 'pinpanion:999')
    with pytest.raises(ValueError, match='expired'):
        service.reject('missing-session', track_id)


def test_mobile_decoder_rejects_invalid_media():
    with pytest.raises(ValueError, match='decoded'):
        decode_scan_image(b'not an image')
    with pytest.raises(ValueError, match='5 MB'):
        decode_scan_image(b'x' * 5_000_001)


def test_mobile_http_serves_app_and_scan_decisions(tmp_path):
    service = MobileService(make_catalog(tmp_path), PinMatcher([]))
    feed = json.dumps({'pins': [{'id': 7, 'name': 'Test Pin', 'image_name': 'test.png',
                                 'year': 2026, 'categoryIds': [38]}],
                       'categories': [{'id': 38, 'name': 'Penny Arcade'}],
                       'sets': [], 'events': [], 'groups': []}).encode()
    fetches = []

    def fetch_feed(url, limit):
        fetches.append((url, limit))
        return feed

    server = make_server('127.0.0.1', 0, service,
                         Path(__file__).resolve().parents[1] / 'mobile_web',
                         fetch_feed=fetch_feed)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    connection = http.client.HTTPConnection('127.0.0.1', server.server_port, timeout=3)
    _, data = image_bytes()
    try:
        connection.request('GET', '/')
        response = connection.getresponse()
        assert response.status == 200
        assert b'Pinident Mobile' in response.read()

        connection.request('GET', '/catalog.json')
        response = connection.getresponse()
        catalog = json.loads(response.read())
        assert response.status == 200
        assert catalog['pins'][0]['id'] == 7
        assert catalog['categories'][0]['name'] == 'Penny Arcade'
        assert fetches == [(FEED_URL, MAX_FEED_BYTES)]

        connection.request('GET', '/api/pins?q=Test')
        response = connection.getresponse()
        assert json.loads(response.read())['pins'][0]['pin_id'] == 'pinpanion:7'

        with patch('mobile_scan.detect_pin_boxes', return_value=[(10, 15, 40, 40)]):
            connection.request('POST', '/api/scan', body=data,
                               headers={'Content-Type': 'image/png', 'X-Scan-Session': 'test-session'})
            response = connection.getresponse()
            assert response.status == 200
            track_id = json.loads(response.read())['boxes'][0]['track_id']

            body = json.dumps({'session_id': 'test-session', 'track_id': track_id,
                               'pin_id': 'pinpanion:7'}).encode()
            connection.request('POST', '/api/tag', body=body,
                               headers={'Content-Type': 'application/json'})
            response = connection.getresponse()
            assert json.loads(response.read())['boxes'][0]['review_state'] == 'confirmed'

            body = json.dumps({'session_id': 'test-session', 'track_id': track_id}).encode()
            connection.request('POST', '/api/reject', body=body,
                               headers={'Content-Type': 'application/json'})
            response = connection.getresponse()
            assert json.loads(response.read())['boxes'] == []
    finally:
        connection.close()
        server.shutdown()
        server.server_close()
        worker.join(timeout=3)

"""Legacy local Python scanner API retained for experiments and compatibility.

The TypeScript mobile PWA scans on-device and does not call this API. Build
``mobile_web/dist`` before using this server as a local static preview.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import re
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Callable
from urllib.parse import parse_qs, unquote, urlsplit

import cv2
import numpy as np

from mobile_scan import ScanState, public_scan_result, scan_frame
from pin_catalog import DEFAULT_CATALOG, PinCatalog, PinMatcher
from pinpanion_sync import (FEED_URL, MAX_FEED_BYTES, fetch_public_bytes,
                            parse_pin_records, start_catalog_sync)


STATIC_ROOT = Path(__file__).with_name('mobile_web') / 'dist'
MAX_IMAGE_BYTES = 5_000_000
MAX_IMAGE_PIXELS = 12_000_000
SESSION_TTL_SECONDS = 10 * 60
MAX_SESSIONS = 32
SESSION_PATTERN = re.compile(r'[A-Za-z0-9_-]{1,64}\Z')


def decode_scan_image(data: bytes) -> np.ndarray:
    if not data or len(data) > MAX_IMAGE_BYTES:
        raise ValueError('Image must be between 1 byte and 5 MB')
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError('Image could not be decoded')
    height, width = image.shape[:2]
    if width * height > MAX_IMAGE_PIXELS:
        raise ValueError('Image exceeds 12 megapixels')
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


class MobileService:
    def __init__(self, catalog: PinCatalog, matcher: PinMatcher):
        self.catalog = catalog
        self.matcher = matcher
        self.sessions: dict[str, tuple[float, ScanState]] = {}

    def scan(self, data: bytes, session_id: str | None = None) -> dict:
        image = decode_scan_image(data)
        previous = None
        if session_id:
            if not SESSION_PATTERN.fullmatch(session_id):
                raise ValueError('Invalid scan session')
            now = time.monotonic()
            self.sessions = {key: value for key, value in self.sessions.items()
                             if now - value[0] < SESSION_TTL_SECONDS}
            previous = self.sessions.get(session_id, (0, None))[1]
        state = scan_frame(image, self.matcher, previous)
        if session_id:
            if session_id not in self.sessions and len(self.sessions) >= MAX_SESSIONS:
                oldest = min(self.sessions, key=lambda key: self.sessions[key][0])
                del self.sessions[oldest]
            self.sessions[session_id] = (time.monotonic(), state)
        return public_scan_result(state)

    def _session_box(self, session_id: str, track_id: str):
        if (not isinstance(session_id, str) or not isinstance(track_id, str)
                or not SESSION_PATTERN.fullmatch(session_id)
                or not SESSION_PATTERN.fullmatch(track_id)):
            raise ValueError('Invalid scan session or track')
        entry = self.sessions.get(session_id)
        if entry is None or time.monotonic() - entry[0] >= SESSION_TTL_SECONDS:
            raise ValueError('Scan session expired; scan again')
        box = next((box for box in entry[1].boxes if box.track_id == track_id), None)
        if box is None:
            raise ValueError('Pin area is no longer in this scan session')
        return entry[1], box

    def tag(self, session_id: str, track_id: str, pin_id: str,
            name: str | None = None) -> dict:
        state, box = self._session_box(session_id, track_id)
        if box.review_state == 'rejected':
            raise ValueError('Rejected area cannot be tagged; scan it again')
        if not isinstance(pin_id, str) or not 0 < len(pin_id) <= 100:
            raise ValueError('Invalid pin ID')
        if pin_id.startswith('local:'):
            if not name or not name.strip() or len(name) > 100:
                raise ValueError('Local pin needs a name under 100 characters')
            pin_name = name.strip()
        else:
            known = self.catalog.get_pin(pin_id)
            if known is None:
                raise ValueError('Pin ID is not in the catalog')
            pin_name = known['name']
        box.pin_id = pin_id
        box.tags = [pin_name]
        box.review_state = 'confirmed'
        return public_scan_result(state)

    def reject(self, session_id: str, track_id: str) -> dict:
        state, box = self._session_box(session_id, track_id)
        box.review_state = 'rejected'
        box.pin_id = None
        box.tags = []
        return public_scan_result(state)


def make_server(host: str, port: int, service: MobileService,
                static_root: Path = STATIC_ROOT,
                fetch_feed: Callable[[str, int], bytes] = fetch_public_bytes) -> HTTPServer:
    root = static_root.resolve()

    class Handler(BaseHTTPRequestHandler):
        def send_json(self, status: int, payload: dict | list) -> None:
            data = json.dumps(payload).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            parsed = urlsplit(self.path)
            if parsed.path == '/catalog.json':
                try:
                    feed = fetch_feed(FEED_URL, MAX_FEED_BYTES)
                    parse_pin_records(feed)
                    payload = json.loads(feed)
                except (OSError, ValueError, TypeError):
                    self.send_json(503, {'error': 'Pinpanion catalog temporarily unavailable'})
                    return
                self.send_json(200, payload)
                return
            if parsed.path == '/api/pins':
                query = parse_qs(parsed.query).get('q', [''])[0]
                if len(query) > 100:
                    self.send_json(400, {'error': 'Search term is too long'})
                    return
                self.send_json(200, {'pins': service.catalog.search_pins(query)})
                return
            if parsed.path == '/api/health':
                self.send_json(200, {'status': 'ok'})
                return
            relative = unquote(parsed.path.lstrip('/')) or 'index.html'
            file = (root / relative).resolve()
            if not file.is_relative_to(root) or not file.is_file():
                self.send_error(404)
                return
            content = file.read_bytes()
            mime = ('application/manifest+json' if file.suffix == '.webmanifest'
                    else mimetypes.guess_type(file.name)[0] or 'application/octet-stream')
            self.send_response(200)
            self.send_header('Content-Type', mime)
            self.send_header('Content-Length', str(len(content)))
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            self.wfile.write(content)

        def do_POST(self) -> None:
            path = urlsplit(self.path).path
            if path in ('/api/tag', '/api/reject'):
                if self.headers.get_content_type() != 'application/json':
                    self.send_json(415, {'error': 'Send JSON'})
                    return
                try:
                    length = int(self.headers.get('Content-Length', '0'))
                    if not 0 < length <= 2048:
                        raise ValueError('Action body is too large or empty')
                    payload = json.loads(self.rfile.read(length))
                    if not isinstance(payload, dict):
                        raise ValueError('Action body must be an object')
                    if path == '/api/tag':
                        result = service.tag(payload.get('session_id'), payload.get('track_id'),
                                             payload.get('pin_id'), payload.get('name'))
                    else:
                        result = service.reject(payload.get('session_id'), payload.get('track_id'))
                except (ValueError, TypeError) as error:
                    self.send_json(400, {'error': str(error)})
                    return
                self.send_json(200, result)
                return
            if path != '/api/scan':
                self.send_json(404, {'error': 'Not found'})
                return
            if self.headers.get_content_type() not in ('image/jpeg', 'image/png', 'image/webp'):
                self.send_json(415, {'error': 'Send a JPEG, PNG, or WebP image'})
                return
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= MAX_IMAGE_BYTES:
                    raise ValueError('Image must be between 1 byte and 5 MB')
                data = self.rfile.read(length)
                result = service.scan(data, self.headers.get('X-Scan-Session'))
            except ValueError as error:
                self.send_json(400, {'error': str(error)})
                return
            self.send_json(200, result)

    return HTTPServer((host, port), Handler)


def main() -> None:
    parser = argparse.ArgumentParser(description='Run the Pinident mobile scanner')
    parser.add_argument('--catalog', type=Path, default=DEFAULT_CATALOG)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    catalog = PinCatalog(args.catalog)
    service = MobileService(catalog, PinMatcher([]))
    start_catalog_sync(catalog, lambda matcher: setattr(service, 'matcher', matcher))
    server = make_server(args.host, args.port, service)
    print(f'Pinident mobile scanner: http://{args.host}:{server.server_port}/')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()

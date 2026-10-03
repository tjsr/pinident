"""Incrementally import Pinpanion's public pin feed into the local catalog."""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from typing import Callable
from urllib.error import HTTPError
from urllib.parse import quote, urlsplit
from urllib.request import Request, urlopen

from logutil import getLog
from pin_catalog import PinCatalog, PinMatcher


FEED_URL = "https://pinpanion.com/pins.json"
IMAGE_BASE_URL = "https://pinpanion.com/imgs/"
MAX_FEED_BYTES = 5_000_000
MAX_IMAGE_BYTES = 8_000_000


@dataclass(frozen=True)
class SyncResult:
    total: int
    existing: int
    imported: int
    failed: int


def fetch_public_bytes(url: str, limit: int) -> bytes:
    """Fetch a bounded response only from Pinpanion's public HTTPS host."""
    if urlsplit(url).scheme != "https" or urlsplit(url).hostname != "pinpanion.com":
        raise ValueError("Unexpected Pinpanion URL")
    request = Request(url, headers={"User-Agent": "Pinident/0.1 (+local catalog sync)"})
    with urlopen(request, timeout=15) as response:
        final_url = urlsplit(response.geturl())
        if final_url.scheme != "https" or final_url.hostname != "pinpanion.com":
            raise ValueError("Pinpanion request redirected to another host")
        content = response.read(limit + 1)
    if len(content) > limit:
        raise ValueError("Pinpanion response exceeds size limit")
    return content


def parse_pin_records(data: bytes) -> list[dict]:
    payload = json.loads(data)
    if not isinstance(payload, dict) or not isinstance(payload.get("pins"), list):
        raise ValueError("Pinpanion feed must contain a pins list")
    records = []
    seen = set()
    for record in payload["pins"]:
        if not isinstance(record, dict):
            raise ValueError("Pinpanion pin record is not an object")
        pin_id = record.get("id")
        name = record.get("name")
        image_name = record.get("image_name")
        if (isinstance(pin_id, bool) or not isinstance(pin_id, int) or pin_id < 0
                or not isinstance(name, str) or not name.strip()
                or not isinstance(image_name, str) or not image_name.strip()):
            raise ValueError(f"Invalid Pinpanion pin record: {pin_id!r}")
        if pin_id in seen:
            raise ValueError(f"Duplicate Pinpanion pin ID: {pin_id}")
        image_path = urlsplit(image_name).path
        if image_path != image_name or image_path in (".", "..") or "/" in image_path or "\\" in image_path:
            raise ValueError(f"Unsafe Pinpanion image name: {image_name!r}")
        seen.add(pin_id)
        records.append(record)
    return records


def sync_pinpanion_catalog(catalog: PinCatalog,
                           fetch: Callable[[str, int], bytes] = fetch_public_bytes) -> SyncResult:
    """Import unseen IDs; failed images remain eligible for the next startup."""
    records = parse_pin_records(fetch(FEED_URL, MAX_FEED_BYTES))
    existing_ids = catalog.pin_ids()
    existing = imported = failed = 0
    for record in records:
        pin_id = f"pinpanion:{record['id']}"
        if pin_id in existing_ids:
            existing += 1
            continue
        image_url = IMAGE_BASE_URL + quote(record["image_name"])
        try:
            image_bytes = fetch(image_url, MAX_IMAGE_BYTES)
            if catalog.import_pinpanion_pin(record, image_bytes, image_url):
                imported += 1
                existing_ids.add(pin_id)
                if imported % 100 == 0:
                    getLog().info("Imported %s new Pinpanion pins", imported)
            else:
                existing += 1
        except HTTPError as error:
            failed += 1
            getLog().warning("Could not import Pinpanion pin %s: HTTP %s", pin_id, error.code)
            if error.code in (429, 503):
                getLog().warning("Pinpanion is limiting requests; remaining pins will retry on next startup")
                break
        except (OSError, ValueError) as error:
            failed += 1
            getLog().warning("Could not import Pinpanion pin %s: %s", pin_id, error)
    return SyncResult(len(records), existing, imported, failed)


def start_catalog_sync(catalog: PinCatalog, on_model_ready: Callable[[PinMatcher], None],
                       fetch: Callable[[str, int], bytes] = fetch_public_bytes) -> threading.Thread:
    """Keep the GUI responsive while loading references, syncing, and indexing."""
    def run() -> None:
        try:
            on_model_ready(PinMatcher(catalog.references()))
            result = sync_pinpanion_catalog(catalog, fetch)
            getLog().info("Pinpanion sync: %s", result)
            if result.imported:
                on_model_ready(PinMatcher(catalog.references()))
        except Exception:
            getLog().exception("Pinpanion catalog sync failed; existing annotations remain available")

    worker = threading.Thread(target=run, name="pinpanion-catalog-sync", daemon=True)
    worker.start()
    return worker

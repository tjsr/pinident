import base64
import json
from email.message import Message

import cv2
import numpy as np

import cloud_sync
from cloud_sync import CloudConfig, apply_reviews, cloud_cursor, fetch_review_page
from detection_feedback import FeedbackStore
from trained_detection_model import PortableDetectionModel


def review(review_id: str, label: str, color: int) -> dict:
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    image[:, :, 0] = color
    image[:, :, 1] = np.arange(16, dtype=np.uint8)
    ok, jpg = cv2.imencode('.jpg', image)
    assert ok
    return {
        'kind': 'training', 'id': review_id, 'label': label, 'source': 'photo.jpg',
        'time_seconds': None, 'pin_id': None, 'name': None,
        'image_data_url': 'data:image/jpeg;base64,' +
        base64.b64encode(jpg.tobytes()).decode(),
    }


def test_cloud_reviews_upsert_by_stable_id_and_cursor_is_durable(tmp_path):
    path = tmp_path / 'feedback.sqlite3'
    store = FeedbackStore(path)
    first = review('review-1', 'pin_unknown', 70)
    second = review('review-2', 'not_pin', 180)
    assert apply_reviews(store, [
        {'revision': 1, 'review': first}, {'revision': 2, 'review': second}], 2) == 2
    assert cloud_cursor(store) == 2
    assert store.counts() == {'positive': 1, 'negative': 1}
    assert apply_reviews(store, [{'revision': 2, 'review': second}], 2) == 0
    updated = review('review-1', 'not_pin', 80)
    assert apply_reviews(store, [{'revision': 3, 'review': updated}], 3) == 1
    assert store.counts() == {'positive': 0, 'negative': 2}
    assert store.get('cloudflare:review-1').source_path == 'cloudflare:web'
    store.db.close()
    reopened = FeedbackStore(path)
    assert cloud_cursor(reopened) == 3
    assert reopened.counts() == {'positive': 0, 'negative': 2}


def test_cloud_frame_annotation_extracts_only_confirmed_pin_crops(tmp_path):
    store = FeedbackStore(tmp_path / 'feedback.sqlite3')
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    frame[4:28, 5:29] = (50, 120, 220)
    ok, jpg = cv2.imencode('.jpg', frame)
    assert ok
    annotation = {
        'kind': 'annotation', 'id': 'frame-12345', 'source': 'video.mp4',
        'time_seconds': 1.5, 'width': 64, 'height': 48,
        'image_data_url': 'data:image/jpeg;base64,' +
        base64.b64encode(jpg.tobytes()).decode(),
        'boxes': [
            {'coords': [5, 4, 24, 24], 'track_id': 'pin-a',
             'review_state': 'confirmed', 'pin_id': 'pinpanion:7', 'name': 'Pin'},
            {'coords': [35, 4, 20, 20], 'track_id': 'candidate-b',
             'review_state': 'unconfirmed', 'pin_id': None, 'name': None},
        ],
    }
    assert apply_reviews(store, [{'revision': 1, 'review': annotation}], 1) == 1
    assert cloud_cursor(store) == 1
    example = store.get('cloudflare:annotation:frame-12345:0')
    assert example is not None
    assert example.label == 'positive'
    assert example.pin_id == 'pinpanion:7'
    assert example.coords == (5, 4, 24, 24)
    assert cv2.imdecode(np.frombuffer(example.crop_png, dtype=np.uint8),
                        cv2.IMREAD_COLOR).shape[:2] == (24, 24)
    assert store.counts() == {'positive': 1, 'negative': 0}
    assert apply_reviews(store, [{'revision': 1, 'review': annotation}], 1) == 0
    changed = {**annotation, 'boxes': [
        {**annotation['boxes'][0], 'review_state': 'unconfirmed'},
        annotation['boxes'][1],
    ]}
    assert apply_reviews(store, [{'revision': 2, 'review': changed}], 2) == 1
    assert store.get('cloudflare:annotation:frame-12345:0') is None
    assert store.counts() == {'positive': 0, 'negative': 0}


def test_cloud_frame_views_of_one_track_remain_separate_training_examples(tmp_path):
    store = FeedbackStore(tmp_path / 'feedback.sqlite3')
    image = np.full((24, 24, 3), 90, dtype=np.uint8)
    ok, jpg = cv2.imencode('.jpg', image)
    assert ok
    frame = {
        'kind': 'annotation', 'source': 'video.mp4', 'time_seconds': 0,
        'width': 24, 'height': 24,
        'image_data_url': 'data:image/jpeg;base64,' +
        base64.b64encode(jpg.tobytes()).decode(),
        'boxes': [{'coords': [0, 0, 24, 24], 'track_id': 'same-track',
                   'review_state': 'inherited', 'pin_id': 'pinpanion:7'}],
    }
    events = [
        {'revision': 1, 'review': {**frame, 'id': 'frame-one'}},
        {'revision': 2, 'review': {**frame, 'id': 'frame-two', 'time_seconds': 0.1}},
    ]
    assert apply_reviews(store, events, 2) == 2
    assert store.get('cloudflare:annotation:frame-one:0') is not None
    assert store.get('cloudflare:annotation:frame-two:0') is not None
    assert store.counts() == {'positive': 2, 'negative': 0}


def test_portable_model_uses_published_linear_decision_plane():
    payload = {'format': 'pinident-linear-svm', 'version': 1, 'feature_version': 2,
               'training_digest': 'a' * 64,
               'sample_counts': {'positive': 1, 'negative': 1},
               'weights': [0.0] * 258, 'bias': -1.0}
    model = PortableDetectionModel.from_payload(payload)
    assert model.classify_feature(np.zeros(258, dtype=np.float32)) == 'negative'
    payload['bias'] = 1.0
    assert PortableDetectionModel.from_payload(payload).classify_feature(
        np.zeros(258, dtype=np.float32)) == 'positive'


def test_desktop_sync_identifies_its_http_client(monkeypatch):
    class Response:
        headers = Message()

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _limit):
            return json.dumps({'reviews': [], 'next': 0}).encode()

    Response.headers['Content-Type'] = 'application/json'

    def open_request(request, timeout):
        assert timeout == 15
        assert request.get_header('User-agent') == 'Pinident/0.2 (+desktop sync)'
        return Response()

    monkeypatch.setattr(cloud_sync, 'urlopen', open_request)
    assert fetch_review_page(CloudConfig('https://pinident.example', 'secret'), 0) == ([], 0)

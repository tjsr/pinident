import json

import cv2
import numpy as np

from detection_feedback import make_feedback_example
from mobile_model_features import FEATURE_LENGTH, portable_feature
from trained_detection_model import TrainedDetectionModel


def sample(kind: str, seed: int) -> np.ndarray:
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    if kind == 'positive':
        cv2.circle(image, (32, 32), 22, (210, 45, 60), -1)
        cv2.circle(image, (32, 32), 11, (250, 230, 80), 3)
    else:
        image[:] = (40, 165, 55)
        cv2.line(image, (4, 10), (60, 52), (20, 25, 240), 9)
    noise = np.random.default_rng(seed).integers(0, 10, image.shape, dtype=np.uint8)
    return cv2.add(image, noise)


def test_exported_linear_model_matches_desktop_classification(tmp_path):
    examples = [make_feedback_example(
        f'{label}-{index}', label, sample(label, index), (0, 0, 64, 64),
        'sample.png', 0, f'{label}-{index}', None)
        for label in ('positive', 'negative') for index in range(4)]
    model = TrainedDetectionModel.train(examples)
    path = tmp_path / 'model.json'
    model.export_web(path)
    payload = json.loads(path.read_text(encoding='utf-8'))
    assert payload['format'] == 'pinident-linear-svm'
    assert payload['feature_version'] == 2
    assert len(payload['weights']) == FEATURE_LENGTH
    for label in ('positive', 'negative'):
        for seed in (0, 1, 10, 11):
            image = sample(label, seed)
            feature = portable_feature(image)
            assert feature is not None
            score = float(np.dot(payload['weights'], feature) + payload['bias'])
            assert ('positive' if score >= 0 else 'negative') == model.classify(image)


def test_portable_feature_reference_values():
    image = np.fromfunction(
        lambda y, x, c: np.where(c == 0, (x * 7 + y * 3) % 256,
                       np.where(c == 1, (x * 2 + y * 11) % 256,
                                (x * 13 + y * 5) % 256)),
        (32, 32, 3), dtype=int).astype(np.uint8)
    feature = portable_feature(image)
    assert feature is not None
    assert len(feature) == FEATURE_LENGTH
    np.testing.assert_allclose(feature[:4],
                               [-0.15275587, -0.14011656, -0.12747726, -0.11483795], atol=1e-6)
    np.testing.assert_allclose(feature[-2:],
                               [0.0125, 0.01044118], atol=1e-6)

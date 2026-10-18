"""Persisted pin/non-pin classifier trained from reviewed crops.

This classifies proposed rectangles. Finding the rectangles remains the job of
the OpenCV proposal stage; a positive prediction is not a pin identity.
The first fit uses OpenCV's linear SVM; additive runs continue its decision
plane with online margin updates and rehearsal of retained examples.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from detection_feedback import FeedbackExample, _classifier_feature, _descriptor
from mobile_model_features import FEATURE_LENGTH, portable_feature
from pin_catalog import PinCatalog


FEATURE_VERSION = 2


def model_path_for_feedback(feedback_path: str | Path | None) -> Path | None:
    if feedback_path is None or str(feedback_path) == ':memory:':
        return None
    return Path(feedback_path).resolve().with_suffix('.svm.yml')


def metadata_path(model_path: str | Path) -> Path:
    return Path(str(model_path) + '.json')


def _feature(crop: np.ndarray) -> np.ndarray | None:
    descriptor = _descriptor(crop)
    if descriptor is None:
        return None
    return _classifier_feature(descriptor)


def examples_digest(examples: list[FeedbackExample]) -> str:
    digest = hashlib.sha256()
    for example in sorted(examples, key=lambda item: item.example_id):
        if example.label not in ('positive', 'negative'):
            continue
        for value in (example.example_id, example.label):
            digest.update(value.encode('utf-8'))
            digest.update(b'\0')
        digest.update(hashlib.sha256(example.crop_png).digest())
    return digest.hexdigest()


def _stock_image_feature(image_bytes: bytes) -> np.ndarray | None:
    """Make one RGB pin feature from a stock image, including transparent PNGs."""
    try:
        decoded = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8),
                               cv2.IMREAD_UNCHANGED)
    except cv2.error:
        return None
    if decoded is None:
        return None
    if decoded.ndim == 2:
        rgb = cv2.cvtColor(decoded, cv2.COLOR_GRAY2RGB)
    elif decoded.shape[2] == 4:
        alpha = decoded[:, :, 3]
        visible_y, visible_x = np.where(alpha > 8)
        if not len(visible_x):
            return None
        left, right = int(visible_x.min()), int(visible_x.max()) + 1
        top, bottom = int(visible_y.min()), int(visible_y.max()) + 1
        rgba = decoded[top:bottom, left:right]
        foreground = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_BGR2RGB)
        opacity = rgba[:, :, 3:4].astype(np.float32) / 255.0
        rgb = np.uint8(np.round(foreground * opacity + 224 * (1 - opacity)))
    elif decoded.shape[2] == 3:
        rgb = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
    else:
        return None
    height, width = rgb.shape[:2]
    if max(height, width) > 512:
        scale = 512 / max(height, width)
        rgb = cv2.resize(rgb, (max(1, round(width * scale)),
                               max(1, round(height * scale))),
                         interpolation=cv2.INTER_AREA)
    return portable_feature(rgb)


@dataclass(frozen=True)
class TrainedDetectionModel:
    svm: object
    sample_counts: dict[str, int]
    training_digest: str
    feature_version: int = FEATURE_VERSION
    review_digest: str = ''
    catalog_digest: str = ''
    catalog_total: int = 0
    catalog_usable: int = 0
    catalog_unverified: int = 0
    catalog_pin_ids: tuple[str, ...] = ()
    catalog_tags: tuple[tuple[str, ...], ...] = ()
    catalog_state: tuple[int, int, int] = (0, 0, 0)
    catalog_provenance: tuple[tuple[int, str, str, str, str], ...] = ()
    review_samples: tuple[tuple[str, str], ...] = ()
    stock_samples: tuple[tuple[str, str], ...] = ()
    linear_weights: np.ndarray | None = None
    linear_bias: float | None = None
    incremental_updates: int = 0

    @classmethod
    def train(cls, examples: list[FeedbackExample],
              catalog_path: str | Path | None = None,
              progress: Callable[[int, int], None] | None = None,
              previous: 'TrainedDetectionModel | None' = None
              ) -> 'TrainedDetectionModel':
        features: list[np.ndarray] = []
        labels: list[int] = []
        sample_keys: list[tuple[str, str]] = []
        review_samples: list[tuple[str, str]] = []
        stock_samples: list[tuple[str, str]] = []
        counts = {'positive': 0, 'negative': 0}
        for example in examples:
            if example.label not in counts:
                continue
            fingerprint = hashlib.sha256(example.label.encode('utf-8') + b'\0' +
                                         example.crop_png).hexdigest()
            review_samples.append((example.example_id, fingerprint))
            bgr = cv2.imdecode(np.frombuffer(example.crop_png, dtype=np.uint8),
                               cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            feature = portable_feature(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            if feature is None:
                continue
            features.append(feature)
            labels.append(1 if example.label == 'positive' else -1)
            sample_keys.append(('review', example.example_id))
            counts[example.label] += 1
        catalog_hash = hashlib.sha256()
        catalog_total = catalog_usable = catalog_unverified = 0
        catalog_ids: list[str] = []
        catalog_tags: list[tuple[str, ...]] = []
        catalog_provenance: list[tuple[int, str, str, str, str]] = []
        max_reference_id = total_image_bytes = 0
        if catalog_path is not None:
            expected_stock = PinCatalog(catalog_path).stock_training_state()[0]
            for stock in PinCatalog(catalog_path).stock_training_images():
                catalog_total += 1
                max_reference_id = max(max_reference_id, stock.reference_id)
                total_image_bytes += len(stock.image_bytes)
                for value in (stock.reference_id, stock.pin_id, stock.name,
                              stock.source, stock.rights, *stock.tags):
                    catalog_hash.update(str(value).encode('utf-8'))
                    catalog_hash.update(b'\0')
                catalog_hash.update(hashlib.sha256(stock.image_bytes).digest())
                stock_fingerprint = hashlib.sha256(
                    json.dumps((stock.pin_id, stock.name, stock.source,
                                stock.rights, stock.tags), sort_keys=True).encode('utf-8') +
                    b'\0' + stock.image_bytes).hexdigest()
                stock_samples.append((str(stock.reference_id), stock_fingerprint))
                feature = _stock_image_feature(stock.image_bytes)
                if feature is not None:
                    features.append(feature)
                    labels.append(1)
                    sample_keys.append(('stock', str(stock.reference_id)))
                    counts['positive'] += 1
                    catalog_usable += 1
                    catalog_ids.append(stock.pin_id)
                    catalog_tags.append(stock.tags)
                    catalog_provenance.append((stock.reference_id, stock.pin_id,
                                               stock.name, stock.source, stock.rights))
                    if 'unverified' in stock.rights.casefold():
                        catalog_unverified += 1
                if progress and (catalog_total % 50 == 0 or
                                 catalog_total == expected_stock):
                    progress(catalog_total, expected_stock)
        if not counts['positive'] or not counts['negative']:
            raise ValueError('Training needs at least one usable positive and one negative crop')
        review_digest = examples_digest(examples)
        catalog_digest = catalog_hash.hexdigest() if catalog_total else ''
        training_digest = (hashlib.sha256(bytes.fromhex(review_digest) +
                          bytes.fromhex(catalog_digest)).hexdigest()
                           if catalog_total else review_digest)
        review_manifest = tuple(sorted(review_samples))
        stock_manifest = tuple(sorted(stock_samples))
        previous_reviews = dict(previous.review_samples) if previous else {}
        previous_stock = dict(previous.stock_samples) if previous else {}
        current_reviews = dict(review_manifest)
        current_stock = dict(stock_manifest)
        additive = bool(previous and previous.feature_version == FEATURE_VERSION and
                        (previous_reviews or previous_stock) and
                        all(current_reviews.get(key) == value
                            for key, value in previous_reviews.items()) and
                        all(current_stock.get(key) == value
                            for key, value in previous_stock.items()))
        if additive and previous.training_digest == training_digest:
            return previous
        if additive:
            weights, bias = previous._decision_plane()
            # Continue the saved linear decision plane. Rehearsing retained
            # examples keeps earlier media represented in each update.
            new_keys = ({('review', key) for key in current_reviews.keys() -
                         previous_reviews.keys()} |
                        {('stock', key) for key in current_stock.keys() -
                         previous_stock.keys()})
            order = [index for index, key in enumerate(sample_keys) if key in new_keys]
            order.extend(index for index, key in enumerate(sample_keys)
                         if key not in new_keys)
            for _ in range(3):
                for index in order:
                    feature = features[index]
                    label = labels[index]
                    margin = label * (float(np.dot(weights, feature)) + bias)
                    if margin >= 1.0:
                        continue
                    step = min(0.25, (1.0 - margin) /
                               (float(np.dot(feature, feature)) + 1.0))
                    weights += step * label * feature
                    bias += step * label
            return cls(previous.svm, counts, training_digest, FEATURE_VERSION,
                       review_digest, catalog_digest, catalog_total,
                       catalog_usable, catalog_unverified, tuple(catalog_ids),
                       tuple(catalog_tags),
                       (catalog_total, max_reference_id, total_image_bytes),
                       tuple(catalog_provenance), review_manifest, stock_manifest,
                       weights, bias, previous.incremental_updates + 1)
        svm = cv2.ml.SVM_create()
        svm.setType(cv2.ml.SVM_C_SVC)
        svm.setKernel(cv2.ml.SVM_LINEAR)
        svm.setC(1.0)
        total = counts['positive'] + counts['negative']
        svm.setClassWeights(np.asarray([
            total / (2 * counts['negative']), total / (2 * counts['positive'])
        ], dtype=np.float32).reshape(2, 1))
        svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS,
                             1000, 1e-6))
        if not svm.train(np.stack(features), cv2.ml.ROW_SAMPLE,
                         np.asarray(labels, dtype=np.int32)):
            raise RuntimeError('OpenCV could not train the pin/non-pin classifier')
        return cls(svm, counts, training_digest, FEATURE_VERSION, review_digest,
                   catalog_digest, catalog_total, catalog_usable,
                   catalog_unverified, tuple(catalog_ids), tuple(catalog_tags),
                   (catalog_total, max_reference_id, total_image_bytes),
                   tuple(catalog_provenance), review_manifest, stock_manifest)

    def classify(self, crop: np.ndarray) -> str | None:
        feature = (_feature(crop) if self.feature_version == 1 else
                   portable_feature(crop))
        return self.classify_feature(feature) if feature is not None else None

    def classify_feature(self, feature: np.ndarray) -> str:
        if self.linear_weights is not None:
            return ('positive' if float(np.dot(self.linear_weights, feature)) +
                    float(self.linear_bias) >= 0 else 'negative')
        _unused, result = self.svm.predict(feature.reshape(1, -1))
        return 'positive' if int(result[0, 0]) == 1 else 'negative'

    def feature(self, crop: np.ndarray) -> np.ndarray | None:
        return (_feature(crop) if self.feature_version == 1 else
                portable_feature(crop))

    def _decision_plane(self) -> tuple[np.ndarray, float]:
        if self.feature_version != FEATURE_VERSION:
            raise ValueError('Retrain the desktop classifier before exporting it for mobile')
        if self.linear_weights is not None:
            return self.linear_weights.astype(np.float64).copy(), float(self.linear_bias)
        vectors = self.svm.getSupportVectors()
        rho, alphas, indices = self.svm.getDecisionFunction(0)
        weights = np.zeros(FEATURE_LENGTH, dtype=np.float64)
        for alpha, index in zip(alphas.ravel(), indices.ravel()):
            weights += float(alpha) * vectors[int(index)]
        bias = -float(rho)
        if vectors.shape[1] != FEATURE_LENGTH or not np.isfinite(weights).all():
            raise ValueError('Unexpected trained classifier dimensions')
        # OpenCV's raw decision sign is not its public class label sign.
        probe = vectors[0].astype(np.float32)
        label = 1 if self.classify_feature(probe) == 'positive' else -1
        raw = float(np.dot(weights, probe) + bias)
        if abs(raw) < 1e-8:
            raise ValueError('Could not orient the classifier decision plane')
        orientation = label * (1 if raw > 0 else -1)
        weights *= orientation
        bias *= orientation
        return weights, bias

    def web_payload(self) -> dict:
        """Export the current linear decision plane, oriented toward pins."""
        weights, bias = self._decision_plane()
        return {'format': 'pinident-linear-svm', 'version': 1,
                'feature_version': FEATURE_VERSION,
                'training_digest': self.training_digest,
                'sample_counts': self.sample_counts,
                'catalog_samples': self.catalog_usable,
                'catalog_total': self.catalog_total,
                'catalog_rights_unverified': self.catalog_unverified,
                'weights': weights.tolist(), 'bias': bias}

    def export_web(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = self.web_payload()
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8',
                                             dir=target.parent, prefix='.pinident-',
                                             suffix='.json', delete=False) as stream:
                temporary = Path(stream.name)
                json.dump(payload, stream, separators=(',', ':'), allow_nan=False)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
            temporary = None
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def save(self, model_path: str | Path) -> None:
        target = Path(model_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        manifest = metadata_path(target)
        model_temp: Path | None = None
        manifest_temp: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(dir=target.parent, prefix='.pinident-',
                                             suffix='.yml', delete=False) as stream:
                model_temp = Path(stream.name)
            self.svm.save(str(model_temp))
            model_hash = hashlib.sha256(model_temp.read_bytes()).hexdigest()
            payload = {'feature_version': FEATURE_VERSION,
                       'training_digest': self.training_digest,
                       'review_digest': self.review_digest or self.training_digest,
                       'catalog_digest': self.catalog_digest,
                       'catalog_total': self.catalog_total,
                       'catalog_usable': self.catalog_usable,
                       'catalog_unverified': self.catalog_unverified,
                       'catalog_pin_ids': self.catalog_pin_ids,
                       'catalog_tags': self.catalog_tags,
                       'catalog_state': self.catalog_state,
                       'catalog_provenance': self.catalog_provenance,
                       'review_samples': self.review_samples,
                       'stock_samples': self.stock_samples,
                       'linear_weights': (self.linear_weights.tolist()
                                          if self.linear_weights is not None else None),
                       'linear_bias': self.linear_bias,
                       'incremental_updates': self.incremental_updates,
                       'sample_counts': self.sample_counts,
                       'model_sha256': model_hash}
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8',
                                             dir=target.parent, prefix='.pinident-',
                                             suffix='.json', delete=False) as stream:
                manifest_temp = Path(stream.name)
                json.dump(payload, stream, sort_keys=True)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(model_temp, target)
            model_temp = None
            os.replace(manifest_temp, manifest)
            manifest_temp = None
        finally:
            for temporary in (model_temp, manifest_temp):
                if temporary is not None:
                    temporary.unlink(missing_ok=True)

    @classmethod
    def load(cls, model_path: str | Path) -> 'TrainedDetectionModel':
        target = Path(model_path)
        payload = json.loads(metadata_path(target).read_text(encoding='utf-8'))
        if payload['feature_version'] not in (1, FEATURE_VERSION):
            raise ValueError('The saved classifier uses an unsupported feature version')
        if hashlib.sha256(target.read_bytes()).hexdigest() != payload['model_sha256']:
            raise ValueError('The saved classifier and metadata do not match')
        svm = cv2.ml.SVM_load(str(target))
        if not svm.isTrained():
            raise ValueError('The saved OpenCV classifier is not trained')
        weights = payload.get('linear_weights')
        if weights is not None:
            weights = np.asarray(weights, dtype=np.float64)
            if (weights.shape != (FEATURE_LENGTH,) or
                    not np.isfinite(weights).all() or
                    not isinstance(payload.get('linear_bias'), (int, float)) or
                    not np.isfinite(payload['linear_bias'])):
                raise ValueError('Invalid saved incremental classifier')
        provenance = tuple(tuple(item) for item in
                           payload.get('catalog_provenance', ()))
        catalog_tags = payload.get('catalog_tags')
        if catalog_tags is None:
            catalog_tags = [(item[2],) for item in provenance]
        return cls(svm, payload['sample_counts'], payload['training_digest'],
                   payload['feature_version'],
                   payload.get('review_digest', payload['training_digest']),
                   payload.get('catalog_digest', ''),
                   payload.get('catalog_total', 0),
                   payload.get('catalog_usable', 0),
                   payload.get('catalog_unverified', 0),
                   tuple(payload.get('catalog_pin_ids', ())),
                   tuple(tuple(tags) for tags in catalog_tags),
                   tuple(payload.get('catalog_state', (0, 0, 0))),
                   provenance,
                   tuple(tuple(item) for item in payload.get('review_samples', ())),
                   tuple(tuple(item) for item in payload.get('stock_samples', ())),
                   weights, payload.get('linear_bias'),
                   payload.get('incremental_updates', 0))


@dataclass(frozen=True)
class PortableDetectionModel:
    """The same linear decision plane used by the browser, downloaded from Cloudflare."""

    weights: np.ndarray
    bias: float
    sample_counts: dict[str, int]
    training_digest: str
    feature_version: int = FEATURE_VERSION

    @classmethod
    def from_payload(cls, payload: object) -> 'PortableDetectionModel':
        if not isinstance(payload, dict) or payload.get('format') != 'pinident-linear-svm' or \
                payload.get('version') != 1 or payload.get('feature_version') != FEATURE_VERSION:
            raise ValueError('Unsupported Cloudflare model')
        counts = payload.get('sample_counts')
        if not isinstance(counts, dict) or any(
                not isinstance(counts.get(key), int) or counts[key] < 1
                for key in ('positive', 'negative')):
            raise ValueError('Invalid Cloudflare model sample counts')
        digest = payload.get('training_digest')
        if not isinstance(digest, str) or len(digest) != 64 or \
                any(character not in '0123456789abcdef' for character in digest):
            raise ValueError('Invalid Cloudflare model digest')
        weights = np.asarray(payload.get('weights'), dtype=np.float32)
        bias = payload.get('bias')
        if weights.shape != (FEATURE_LENGTH,) or not np.isfinite(weights).all() or \
                not isinstance(bias, (int, float)) or not np.isfinite(bias):
            raise ValueError('Invalid Cloudflare model weights')
        return cls(weights, float(bias), counts, digest)

    def feature(self, crop: np.ndarray) -> np.ndarray | None:
        return portable_feature(crop)

    def classify_feature(self, feature: np.ndarray) -> str:
        return 'positive' if float(np.dot(self.weights, feature) + self.bias) >= 0 else 'negative'

    def classify(self, crop: np.ndarray) -> str | None:
        feature = self.feature(crop)
        return self.classify_feature(feature) if feature is not None else None

    def web_payload(self) -> dict:
        return {'format': 'pinident-linear-svm', 'version': 1,
                'feature_version': FEATURE_VERSION, 'training_digest': self.training_digest,
                'sample_counts': self.sample_counts, 'weights': self.weights.tolist(),
                'bias': self.bias}

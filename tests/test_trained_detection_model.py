import json
from concurrent.futures import Future
from pathlib import Path
import sqlite3
from time import monotonic, sleep

import cv2
import numpy as np
import pytest
import wx

from boxdata import BoxData
from detection_feedback import FeedbackModel, FeedbackStore, make_feedback_example
from frame_processing import FrameResult
from pin_catalog import PinCatalog
from trained_detection_model import (PortableDetectionModel, TrainedDetectionModel,
                                     examples_digest, metadata_path,
                                     model_path_for_feedback)
from videoscrubber import VideoScrubber


def sample(kind: str, seed: int) -> np.ndarray:
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    if kind == 'positive':
        cv2.circle(image, (32, 32), 24, (220, 50, 50), -1)
        cv2.circle(image, (32, 32), 12, (250, 230, 70), 3)
    else:
        image[:] = (35, 170, 45)
        cv2.line(image, (4, 10), (60, 52), (20, 30, 240), 9)
    noise = np.random.default_rng(seed).integers(0, 12, image.shape, dtype=np.uint8)
    return cv2.add(image, noise)


def test_opencv_model_trains_from_reviewed_crops_and_loads_for_classification(tmp_path):
    store = FeedbackStore(tmp_path / 'feedback.sqlite3')
    try:
        for label in ('positive', 'negative'):
            for index in range(3):
                store.save(make_feedback_example(
                    f'{label}-{index}', label, sample(label, index),
                    (0, 0, 64, 64), f'{label}.png', 0, f'{label}-{index}', None))
        model = TrainedDetectionModel.train(store.examples())
        assert model.sample_counts == {'positive': 3, 'negative': 3}
        assert model.classify(sample('positive', 10)) == 'positive'
        assert model.classify(sample('negative', 10)) == 'negative'

        path = model_path_for_feedback(store.path)
        assert path == Path(tmp_path / 'feedback.svm.yml')
        model.save(path)
        restored = TrainedDetectionModel.load(path)
        assert restored.classify(sample('positive', 11)) == 'positive'
        assert restored.classify(sample('negative', 11)) == 'negative'
    finally:
        store.close()


def test_retraining_uses_reviews_from_previous_media(tmp_path):
    path = tmp_path / 'feedback.sqlite3'
    first_source = str(tmp_path / 'first.mp4')
    second_source = str(tmp_path / 'second.png')
    first_examples = [make_feedback_example(
        f'first-{label}', label, sample(label, 1), (0, 0, 64, 64),
        first_source, 0, f'first-{label}', None)
        for label in ('positive', 'negative')]
    store = FeedbackStore(path)
    try:
        store.apply_source_delta(first_source, first_examples, set())
        original = TrainedDetectionModel.train(store.examples())
        assert original.sample_counts == {'positive': 1, 'negative': 1}
        original.save(model_path_for_feedback(path))
    finally:
        store.close()

    # A later media session reopens the same database and updates only its source.
    store = FeedbackStore(path)
    try:
        second_examples = [make_feedback_example(
            f'second-{label}', label, sample(label, 2), (0, 0, 64, 64),
            second_source, 0, f'second-{label}', None)
            for label in ('positive', 'negative')]
        store.apply_source_delta(second_source, second_examples, set())
        previous = TrainedDetectionModel.load(model_path_for_feedback(path))
        retrained = TrainedDetectionModel.train(store.examples(), previous=previous)
        assert retrained.sample_counts == {'positive': 2, 'negative': 2}
        assert retrained.training_digest != original.training_digest
        assert retrained.incremental_updates == 1
        assert retrained.linear_weights is not None
        assert retrained.svm is previous.svm
        assert not np.allclose(retrained._decision_plane()[0],
                               previous._decision_plane()[0])
        assert retrained.classify(sample('positive', 1)) == 'positive'
        assert retrained.classify(sample('negative', 1)) == 'negative'
        assert {item.source_path for item in store.examples()} == {
            first_source, second_source}
        retrained.save(model_path_for_feedback(path))
        restored = TrainedDetectionModel.load(model_path_for_feedback(path))
        assert restored.incremental_updates == 1
        assert restored.classify(sample('positive', 2)) == 'positive'
        assert restored.classify(sample('negative', 2)) == 'negative'
        portable = PortableDetectionModel.from_payload(restored.web_payload())
        assert portable.classify(sample('positive', 2)) == 'positive'
        assert portable.classify(sample('negative', 2)) == 'negative'
        assert TrainedDetectionModel.train(store.examples(), previous=restored) is restored

        store.apply_source_delta(second_source, [], {'second-negative'})
        assert {item.example_id for item in store.examples()} == {
            'first-positive', 'first-negative', 'second-positive'}
        corrected = TrainedDetectionModel.train(store.examples(), previous=restored)
        assert corrected.sample_counts == {
            'positive': 2, 'negative': 1}
        assert corrected.incremental_updates == 0
    finally:
        store.close()


def test_training_requires_both_pin_and_non_pin_reviews(tmp_path):
    store = FeedbackStore(tmp_path / 'feedback.sqlite3')
    try:
        store.save(make_feedback_example('only-pin', 'positive', sample('positive', 0),
                                         (0, 0, 64, 64), 'pin.png', 0, 'pin', None))
        with pytest.raises(ValueError, match='positive and one negative'):
            TrainedDetectionModel.train(store.examples())
        assert not model_path_for_feedback(store.path).exists()
    finally:
        store.close()


def test_training_uses_every_stored_pinpanion_stock_image(tmp_path):
    catalog = PinCatalog(tmp_path / 'catalog.sqlite3')
    for pin_id in (11, 12):
        image = sample('positive', pin_id)
        ok, encoded = cv2.imencode('.png', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        assert ok
        assert catalog.import_pinpanion_pin(
            {'id': pin_id, 'name': f'Pin {pin_id}'}, encoded.tobytes(),
            f'https://pinpanion.com/imgs/{pin_id}.png')
    with sqlite3.connect(catalog.path) as db:
        db.execute("UPDATE pins SET name = 'Local Alias' WHERE pin_id = 'pinpanion:11'")
    reviewed = [make_feedback_example(
        f'{label}-review', label, sample(label, 99), (0, 0, 64, 64),
        'reviewed.png', 0, label, None) for label in ('positive', 'negative')]

    model = TrainedDetectionModel.train(reviewed, catalog.path)

    assert model.sample_counts == {'positive': 3, 'negative': 1}
    assert model.catalog_total == 2
    assert model.catalog_usable == 2
    assert model.catalog_pin_ids == ('pinpanion:11', 'pinpanion:12')
    assert model.catalog_tags == (('Pin 11',), ('Pin 12',))
    assert model.catalog_provenance[0][2] == 'Local Alias'
    assert model.catalog_unverified == 2
    assert model.catalog_state == catalog.stock_training_state()
    assert model.review_digest != model.training_digest
    assert model.web_payload()['catalog_samples'] == 2
    path = model_path_for_feedback(tmp_path / 'feedback.sqlite3')
    model.save(path)
    assert json.loads(metadata_path(path).read_text(encoding='utf-8'))['catalog_tags'] == [
        ['Pin 11'], ['Pin 12']]
    restored = TrainedDetectionModel.load(path)
    assert restored.catalog_pin_ids == model.catalog_pin_ids
    assert restored.catalog_tags == model.catalog_tags
    assert restored.catalog_state == model.catalog_state
    assert restored.catalog_usable == 2
    assert restored.catalog_provenance[0][3:] == (
        'https://pinpanion.com/imgs/11.png', 'source terms unverified')

    next_image = sample('positive', 13)
    ok, encoded = cv2.imencode('.png', cv2.cvtColor(next_image, cv2.COLOR_RGB2BGR))
    assert ok
    assert catalog.import_pinpanion_pin(
        {'id': 13, 'name': 'Pin 13'}, encoded.tobytes(),
        'https://pinpanion.com/imgs/13.png')
    continued = TrainedDetectionModel.train(reviewed, catalog.path,
                                            previous=restored)
    assert continued.incremental_updates == 1
    assert continued.sample_counts == {'positive': 4, 'negative': 1}
    assert continued.catalog_pin_ids == (
        'pinpanion:11', 'pinpanion:12', 'pinpanion:13')


def test_training_reports_unusable_stock_image_and_handles_transparency(tmp_path):
    catalog = PinCatalog(tmp_path / 'catalog.sqlite3')
    rgba = np.zeros((80, 80, 4), dtype=np.uint8)
    rgba[8:72, 8:72, :3] = cv2.cvtColor(
        sample('positive', 16), cv2.COLOR_RGB2BGR)
    rgba[8:72, 8:72, 3] = 255
    ok, encoded = cv2.imencode('.png', rgba)
    assert ok
    for pin_id in (16, 17):
        assert catalog.import_pinpanion_pin(
            {'id': pin_id, 'name': f'Pin {pin_id}'}, encoded.tobytes(),
            f'https://pinpanion.com/imgs/{pin_id}.png')
    with sqlite3.connect(catalog.path) as db:
        db.execute("UPDATE pin_references SET image = ? WHERE pin_id = ?",
                   (b'broken image', 'pinpanion:17'))
    negative = make_feedback_example(
        'negative-review', 'negative', sample('negative', 5),
        (0, 0, 64, 64), 'reviewed.png', 0, 'background', None)

    model = TrainedDetectionModel.train([negative], catalog.path)

    assert model.sample_counts == {'positive': 1, 'negative': 1}
    assert (model.catalog_total, model.catalog_usable) == (2, 1)
    assert model.catalog_pin_ids == ('pinpanion:16',)
    assert model.catalog_state == catalog.stock_training_state()


def test_trained_classifier_downranks_negative_without_erasing_proposals(tmp_path):
    training = [make_feedback_example(
        f'{label}-{index}', label, sample(label, index),
        (0, 0, 64, 64), f'{label}.png', 0, f'{label}-{index}', None)
        for label in ('positive', 'negative') for index in range(3)]
    classifier = TrainedDetectionModel.train(training)
    store = FeedbackStore()
    try:
        feedback = FeedbackModel(store)
        feedback.trained_model = classifier
        frame = np.concatenate((sample('positive', 20), sample('negative', 20)), axis=1)
        assert feedback.snapshot_for_processing().filter_and_rank(
            frame, [(64, 0, 64, 64), (0, 0, 64, 64)]) == [
                (0, 0, 64, 64), (64, 0, 64, 64)]
    finally:
        store.close()


@pytest.mark.gui
def test_saved_classifier_with_changed_reviews_is_not_used_on_startup(tmp_path,
                                                                       monkeypatch):
    monkeypatch.delenv('PINIDENT_SYNC_TOKEN', raising=False)
    path = tmp_path / 'reviews.sqlite3'
    store = FeedbackStore(path)
    try:
        for label in ('positive', 'negative'):
            store.save(make_feedback_example(
                label, label, sample(label, 1), (0, 0, 64, 64),
                'reviewed.png', 0, label, None))
        TrainedDetectionModel.train(store.examples()).save(model_path_for_feedback(path))
        store.save(make_feedback_example(
            'new-negative', 'negative', sample('negative', 2),
            (0, 0, 64, 64), 'reviewed.png', 1, 'new-negative', None))
    finally:
        store.close()

    app = wx.App.Get() or wx.App(False)
    window = VideoScrubber(
        None, 'Stale model', '', image_array=[sample('positive', 3)],
        load_catalog=False, feedback_path=path,
        catalog_path=str(tmp_path / 'empty-catalog.sqlite3'))
    try:
        assert window._feedback_model.trained_model is None
    finally:
        window.Destroy()
        wx.Yield()


@pytest.mark.gui
def test_run_training_now_indexes_annotations_saves_and_reloads_classifier(
        tmp_path, monkeypatch):
    monkeypatch.delenv('PINIDENT_SYNC_TOKEN', raising=False)
    monkeypatch.setattr('scrubberframe.fetch_model', lambda _config: None)
    app = wx.App.Get() or wx.App(False)
    rgb = np.concatenate((sample('positive', 30), sample('negative', 30)), axis=1)
    second_view = np.concatenate((sample('positive', 31), sample('negative', 31)), axis=1)
    feedback_path = tmp_path / 'reviews.sqlite3'
    window = VideoScrubber(None, 'Train now', '',
                           image_array=[cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                                        for frame in (rgb, second_view)],
                           load_catalog=False, feedback_path=feedback_path,
                           catalog_path=str(tmp_path / 'empty-catalog.sqlite3'))
    second = None
    third = None
    try:
        window.Show()
        wx.Yield()
        window.box_data_filename = str(tmp_path / 'annotations.json')
        window._ScrubberFrame__frame_boxes[0] = [
            BoxData((0, 0, 64, 64), [''], 'user', review_state='confirmed',
                    track_id='pin-track', feedback_id='positive-1',
                    feedback_label='positive'),
            BoxData((64, 0, 64, 64), [], 'automatic', review_state='rejected',
                    feedback_id='negative-1', feedback_label='negative'),
        ]
        window._ScrubberFrame__frame_boxes[1] = [
            BoxData((0, 0, 64, 64), [''], 'automatic', track_id='pin-track',
                    review_state='inherited')]
        item_id = window.GetMenuBar().FindMenuItem('Training', 'Run training now')
        assert item_id != wx.NOT_FOUND
        window.GetEventHandler().ProcessEvent(wx.CommandEvent(wx.EVT_MENU.typeId, item_id))
        path = model_path_for_feedback(feedback_path)
        deadline = monotonic() + 5
        while not path.is_file() and monotonic() < deadline:
            wx.Yield()
            sleep(0.005)
        assert path.is_file()
        assert Path(str(path) + '.json').is_file()
        assert window._feedback_store.counts() == {'positive': 2, 'negative': 1}
        assert window._feedback_model.trained_model is not None
        assert window._feedback_model.trained_model.sample_counts == {
            'positive': 2, 'negative': 1}
        assert window._feedback_model.trained_model.classify(sample('negative', 31)) == 'negative'

        image_path = tmp_path / 'photo.png'
        assert cv2.imwrite(str(image_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        second = VideoScrubber(None, 'Reload trained model for image', '',
                               video_path=str(image_path), load_catalog=False,
                               feedback_path=feedback_path,
                               catalog_path=str(tmp_path / 'empty-catalog.sqlite3'))
        assert second._feedback_model.trained_model is not None
        Path(second.box_data_filename).write_text('{"0": []}', encoding='utf-8')
        second.load_box_data()
        assert second._feedback_model.trained_model is not None
        assert second._feedback_model.trained_model.classify(sample('positive', 31)) == 'positive'
        assert second._feedback_model.snapshot_for_processing().filter_and_rank(
            rgb, [(0, 0, 64, 64), (64, 0, 64, 64)]) == [(0, 0, 64, 64)]

        second._ScrubberFrame__frame_boxes[0] = [
            BoxData((0, 0, 64, 64), [''], 'user', review_state='confirmed',
                    feedback_id='photo-positive', feedback_label='positive'),
            BoxData((64, 0, 64, 64), [], 'automatic', review_state='rejected',
                    feedback_id='photo-negative', feedback_label='negative'),
        ]
        second.run_training_now()
        deadline = monotonic() + 5
        while (second._model_training_future is not None or
               second._training_future is not None) and monotonic() < deadline:
            wx.Yield()
            sleep(0.005)
        wx.Yield()
        continued = second._feedback_model.trained_model
        assert continued is not None
        assert continued.incremental_updates == 1
        assert continued.sample_counts == {'positive': 3, 'negative': 2}
        assert {item.source_path for item in second._feedback_store.examples()} == {
            window._feedback_source, str(image_path.resolve())}

        queued_detection = Future()
        completed_training = Future()
        completed_training.set_result(continued)
        second._pending_frame_job = queued_detection
        second._model_training_future = completed_training
        second._finish_model_training(
            completed_training,
            examples_digest(second._feedback_store.examples()),
            continued.training_digest, continued.incremental_updates)
        assert queued_detection.cancelled()
        assert second._pending_frame_job is None
        assert second._feedback_model.trained_model is continued

        # An older cloud model may arrive later, but this current local fit
        # remains the classifier for the open desktop session.
        cloud = Future()
        cloud.set_result((1, PortableDetectionModel.from_payload(
            continued.web_payload()), [], 0, False))
        second._cloud_sync_future = cloud
        second._finish_cloud_sync(cloud)
        assert second._feedback_model.trained_model is continued

        third = VideoScrubber(
            None, 'Local model takes priority over cloud cache', '',
            image_array=[cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)],
            load_catalog=False, feedback_path=feedback_path,
            catalog_path=str(tmp_path / 'empty-catalog.sqlite3'))
        assert isinstance(third._feedback_model.trained_model,
                          TrainedDetectionModel)
        assert (third._feedback_model.trained_model.training_digest ==
                continued.training_digest)

        used_models = []

        def capture_detection(job, read_frame):
            used_models.append(job.feedback_model.trained_model)
            return FrameResult(job.index, read_frame(job.index), job.existing_boxes,
                               len(job.existing_boxes), 0)

        monkeypatch.setattr('scrubberframe.process_frame', capture_detection)
        second.on_detect_async(wx.CommandEvent())
        deadline = monotonic() + 5
        while second._pending_frame_job is not None and monotonic() < deadline:
            wx.Yield()
            sleep(0.005)
        assert used_models == [continued]
    finally:
        if third is not None:
            third.Destroy()
        if second is not None:
            second.Destroy()
        window.Destroy()
        wx.Yield()


@pytest.mark.gui
def test_run_training_now_includes_catalog_stock_images_without_reviewed_pin(tmp_path,
                                                                               monkeypatch):
    monkeypatch.setenv('PINIDENT_SYNC_TOKEN', 'test-desktop-token')
    monkeypatch.setattr('scrubberframe.fetch_model', lambda _config: None)
    monkeypatch.setattr('scrubberframe.fetch_review_page',
                        lambda _config, after: ([], after))
    published = []
    monkeypatch.setattr('scrubberframe.publish_model',
                        lambda *_args: published.append(True))
    catalog = PinCatalog(tmp_path / 'known.sqlite3')
    for pin_id in (21, 22):
        image = sample('positive', pin_id)
        ok, encoded = cv2.imencode('.png', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        assert ok
        assert catalog.import_pinpanion_pin(
            {'id': pin_id, 'name': f'Pin {pin_id}'}, encoded.tobytes(),
            f'https://pinpanion.com/imgs/{pin_id}.png')
    app = wx.App.Get() or wx.App(False)
    background = sample('negative', 44)
    feedback_path = tmp_path / 'reviews.sqlite3'
    window = VideoScrubber(
        None, 'Catalog training', '',
        image_array=[cv2.cvtColor(background, cv2.COLOR_RGB2BGR)],
        feedback_path=feedback_path, catalog_path=str(catalog.path),
        load_catalog=False)
    try:
        window.Show()
        wx.Yield()
        window.box_data_filename = str(tmp_path / 'annotations.json')
        window._ScrubberFrame__frame_boxes[0] = [BoxData(
            (0, 0, 64, 64), [], 'automatic', review_state='rejected',
            feedback_id='negative-1', feedback_label='negative')]
        window.run_training_now()
        path = model_path_for_feedback(feedback_path)
        deadline = monotonic() + 5
        while not path.is_file() and monotonic() < deadline:
            wx.Yield()
            sleep(0.005)
        assert path.is_file()
        model = window._feedback_model.trained_model
        assert model is not None
        assert model.sample_counts == {'positive': 2, 'negative': 1}
        assert model.catalog_pin_ids == ('pinpanion:21', 'pinpanion:22')
        assert model.catalog_tags == (('Pin 21',), ('Pin 22',))
        assert not published
        assert 'publication is held' in window.GetStatusBar().GetStatusText(1)
        assert '2/2 Pinpanion stock images' in ' '.join(
            window._ScrubberFrame__tag_panel.training_status.GetLabel().split())
        assert window._feedback_store.counts() == {'positive': 0, 'negative': 1}
    finally:
        window.Destroy()
        wx.Yield()


@pytest.mark.gui
def test_run_training_pulls_web_reviews_then_publishes_model(tmp_path, monkeypatch):
    monkeypatch.setenv('PINIDENT_SYNC_TOKEN', 'test-desktop-token')
    monkeypatch.setattr('scrubberframe.fetch_model', lambda _config: None)
    monkeypatch.setattr('scrubberframe.fetch_review_page',
                        lambda _config, after: ([], after))
    published = []

    def publish(config, payload, base_version):
        published.append((config.token, payload, base_version))
        return 1

    monkeypatch.setattr('scrubberframe.publish_model', publish)
    app = wx.App.Get() or wx.App(False)
    rgb = np.concatenate((sample('positive', 30), sample('negative', 30)), axis=1)
    window = VideoScrubber(None, 'Publish trained model', '',
                           image_array=[cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)],
                           load_catalog=False, feedback_path=tmp_path / 'reviews.sqlite3',
                           catalog_path=str(tmp_path / 'empty-catalog.sqlite3'))
    try:
        window.Show()
        wx.Yield()
        window.box_data_filename = str(tmp_path / 'annotations.json')
        window._ScrubberFrame__frame_boxes[0] = [
            BoxData((0, 0, 64, 64), [''], 'user', review_state='confirmed',
                    feedback_id='positive-1', feedback_label='positive'),
            BoxData((64, 0, 64, 64), [], 'automatic', review_state='rejected',
                    feedback_id='negative-1', feedback_label='negative'),
        ]
        window.run_training_now()
        deadline = monotonic() + 5
        while not published and monotonic() < deadline:
            wx.Yield()
            sleep(0.005)
        assert len(published) == 1
        token, payload, base = published[0]
        assert token == 'test-desktop-token'
        assert payload['format'] == 'pinident-linear-svm'
        assert payload['web_review_revision'] == 0
        assert base == 0
    finally:
        window.Destroy()
        wx.Yield()

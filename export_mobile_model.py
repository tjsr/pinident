"""Export a trained desktop pin/non-pin classifier for the mobile build.

Run after the desktop application's "Run training now" action. The resulting
mobile_web/model.json is intentionally ignored by Git and uploaded as a static
asset by Wrangler when the mobile app is deployed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from detection_feedback import DEFAULT_FEEDBACK_DB, FeedbackStore
from trained_detection_model import (TrainedDetectionModel, examples_digest,
                                     model_path_for_feedback)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path,
                        default=model_path_for_feedback(DEFAULT_FEEDBACK_DB),
                        help='Desktop .svm.yml file (default: local feedback model)')
    parser.add_argument('--output', type=Path,
                        default=Path(__file__).with_name('mobile_web') / 'model.json',
                        help='Mobile JSON output (default: mobile_web/model.json)')
    parser.add_argument('--feedback', type=Path, default=DEFAULT_FEEDBACK_DB,
                        help='Reviewed-crop database used to train the model')
    args = parser.parse_args()
    if not args.feedback.is_file():
        raise SystemExit(f'Reviewed-crop database not found: {args.feedback}')
    model = TrainedDetectionModel.load(args.model)
    store = FeedbackStore(args.feedback)
    try:
        current_digest = examples_digest(store.examples())
    finally:
        store.close()
    if model.training_digest != current_digest:
        raise SystemExit('Desktop reviews changed after training; run training again before export')
    model.export_web(args.output)
    print(f'Exported {model.sample_counts} reviewed crops to {args.output}')
    print(f'Training digest: {model.training_digest}')


if __name__ == '__main__':
    main()

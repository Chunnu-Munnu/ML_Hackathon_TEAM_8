"""Contract tests for the API, using a temporary two-node trip."""

import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock, patch

import joblib
import numpy as np
from fastapi.testclient import TestClient
from sklearn.dummy import DummyClassifier

from api import main
from pathsnap.features import FEATURES
from pathsnap.pipeline import predict_trip


class OneArcClassifier:
    def predict_proba(self, features):
        if list(features) != FEATURES:
            raise AssertionError("Features must use the unlabeled model schema")
        return np.tile([0.2, 0.8], (len(features), 1))


class ApiTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.trip_id = "00000001"
        trip_dir = self.root / "trips" / self.trip_id
        trip_dir.mkdir(parents=True)
        (trip_dir / f"{self.trip_id}.nodes").write_text("77 12\n77.001 12\n", encoding="ascii")
        (trip_dir / f"{self.trip_id}.arcs").write_text("0 1\n", encoding="ascii")
        self.track_path = trip_dir / f"{self.trip_id}.track"
        self.track_path.write_text("77.0001 12 1\n77.0002 12 2\n", encoding="ascii")
        self.route_path = trip_dir / f"{self.trip_id}.route"
        self.route_path.write_text("0\n", encoding="ascii")
        (self.root / "trips" / "incomplete").mkdir()
        self.artifacts_dir = self.root / "artifacts"
        self.artifacts_dir.mkdir()
        self.log_path = self.root / "api.log"

    def app(self):
        return main.create_app(self.root / "trips", self.artifacts_dir, self.log_path)

    def test_missing_artifacts_and_trip_validation(self):
        with TestClient(self.app()) as client:
            health = client.get("/health").json()
            self.assertEqual(health["status"], "degraded")
            self.assertIn("Missing model.joblib", health["issues"])
            self.assertIn("Missing hmm_config.json", health["issues"])
            self.assertEqual(client.get("/trips").json(), {"trips": [self.trip_id]})
            for trip_id, status in [(self.trip_id, 503), ("unknown", 404), ("../trips", 422)]:
                with self.subTest(trip_id=trip_id):
                    self.assertEqual(client.post("/predict", json={"trip_id": trip_id}).status_code, status)
        self.assertIn("POST '/predict' status=503", self.log_path.read_text(encoding="utf-8"))

    def test_invalid_hmm_config_is_reported(self):
        (self.artifacts_dir / "hmm_config.json").write_text('{"beta": 0}', encoding="utf-8")
        with TestClient(self.app()) as client:
            self.assertIn("Invalid hmm_config.json", client.get("/health").json()["issues"])

    def test_wrong_classifier_labels_are_rejected_at_startup(self):
        model = DummyClassifier().fit([[0], [1]], [0, 2])
        joblib.dump(model, self.artifacts_dir / "model.joblib")
        with TestClient(self.app()) as client:
            self.assertIn("Invalid model.joblib", client.get("/health").json()["issues"])

    def _fixture_prediction(self):
        calls = []
        scorer = ModuleType("test_scorer")
        scorer.continuity = Mock(return_value=1.0)
        scorer.point_accuracy = Mock(return_value=1.0)
        scorer.route_prf = Mock(return_value={"f1": 1.0})
        runtime = main.Runtime(OneArcClassifier(), {}, scorer)

        def reconstruct(trip, model, hmm_cfg):
            self.assertIsNone(trip.route)
            calls.append("inference")
            # The live smoke test exercises the actual decoder. This fixture tests API boundaries.
            return predict_trip(trip, model, hmm_cfg, use_hmm=False)

        read_reference = main.load_route

        def score_input(trip_dir):
            self.assertEqual(calls, ["inference"])
            calls.append("reference")
            return read_reference(trip_dir)

        with (
            patch.object(main, "_load_runtime", return_value=runtime),
            patch.object(main, "predict_trip", side_effect=reconstruct),
            patch.object(main, "load_route", side_effect=score_input),
            TestClient(self.app()) as client,
        ):
            response = client.post("/predict", json={"trip_id": self.trip_id})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(calls, ["inference", "reference"])
        return response.json(), scorer

    def test_prediction_scoring_is_separate_and_payload_contains_no_labels(self):
        prediction, scorer = self._fixture_prediction()
        self.assertEqual(set(prediction), {
            "trip_id", "n_points", "gps", "pred_route", "point_arcs", "metrics", "timings",
        })
        self.assertEqual(prediction["point_arcs"], [0, 0])
        self.assertEqual(prediction["metrics"]["top1_accuracy"], 1.0)
        self.assertEqual(len(prediction["pred_route"]), 1)
        self.assertIn("evaluation_s", prediction["timings"])
        self.assertIn("inference_s", prediction["timings"])
        scorer.point_accuracy.assert_called_once()
        self.assertIn(f"PREDICT trip={self.trip_id}", self.log_path.read_text(encoding="utf-8"))

    def test_missing_reference_leaves_prediction_and_continuity_available(self):
        self.route_path.unlink()
        prediction, scorer = self._fixture_prediction()
        self.assertIsNone(prediction["metrics"]["top1_accuracy"])
        self.assertIsNone(prediction["metrics"]["route_f1"])
        self.assertEqual(prediction["metrics"]["continuity"], 1.0)
        self.assertEqual(prediction["point_arcs"], [0, 0])
        scorer.point_accuracy.assert_not_called()

    def test_gps_thinning_preserves_endpoints_and_prediction_count(self):
        n_points = 3002
        lons = np.linspace(77.0001, 77.0008, n_points)
        rows = (f"{lon:.9f} 12 {idx}" for idx, lon in enumerate(lons))
        self.track_path.write_text("\n".join(rows) + "\n", encoding="ascii")
        prediction, _ = self._fixture_prediction()
        self.assertEqual(len(prediction["gps"]), 3000)
        self.assertEqual(prediction["n_points"], n_points)
        self.assertEqual(len(prediction["point_arcs"]), n_points)
        self.assertAlmostEqual(prediction["gps"][0][1], lons[0])
        self.assertAlmostEqual(prediction["gps"][-1][1], lons[-1])


if __name__ == "__main__":
    unittest.main()

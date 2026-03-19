"""Unit tests for TransNetV2 model wrapper."""
import sys
import os
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


class TestTransNetV2:
    """Tests for the TransNetV2 model wrapper class."""

    def _make_model(self):
        """Create a TransNetV2 with a mocked ONNX session."""
        mock_session = MagicMock()
        mock_session.get_inputs.return_value = [MagicMock(name="input")]
        mock_session.run.return_value = [np.array([[0.1], [0.9], [0.05]])]

        with patch("app.models.transnetv2.ort.InferenceSession", return_value=mock_session):
            from app.models.transnetv2 import TransNetV2
            model = TransNetV2("fake_path.onnx")
            model.session = mock_session
            return model, mock_session

    def test_predict_normalizes_uint8(self):
        """predict() converts uint8 frames to float32 [0,1]."""
        model, mock_session = self._make_model()
        mock_session.run.return_value = [np.array([[0.5], [0.5], [0.5]])]

        frames = np.full((3, 27, 48, 3), 255, dtype=np.uint8)
        result = model.predict(frames)

        # Check that run was called with float32 input
        call_args = mock_session.run.call_args
        # call_args[0][1] is the feed_dict, keyed by model.input_name
        feed_dict = call_args[0][1]
        input_arr = next(iter(feed_dict.values()))
        assert input_arr.dtype == np.float32
        np.testing.assert_allclose(input_arr.max(), 1.0, atol=1e-5)

    def test_predict_float32_not_renormalized(self):
        """predict() does not re-normalize if input is already float32."""
        model, mock_session = self._make_model()
        mock_session.run.return_value = [np.array([[0.5], [0.5], [0.5]])]

        frames = np.full((3, 27, 48, 3), 0.5, dtype=np.float32)
        model.predict(frames)

        call_args = mock_session.run.call_args
        feed_dict = call_args[0][1]
        input_arr = next(iter(feed_dict.values()))
        # float32 input should pass through as-is (0.5, not 0.5/255)
        np.testing.assert_allclose(input_arr.max(), 0.5, atol=1e-5)

    def test_predict_returns_1d_array(self):
        """predict() always returns a 1D probability array."""
        model, mock_session = self._make_model()

        # Test with 2D output (N, 1)
        mock_session.run.return_value = [np.array([[0.1], [0.9], [0.3]])]
        frames = np.zeros((3, 27, 48, 3), dtype=np.uint8)
        result = model.predict(frames)
        assert result.ndim == 1
        assert len(result) == 3

    def test_predict_flat_output(self):
        """predict() handles already-flat model output."""
        model, mock_session = self._make_model()

        mock_session.run.return_value = [np.array([0.1, 0.9, 0.3])]
        frames = np.zeros((3, 27, 48, 3), dtype=np.uint8)
        result = model.predict(frames)
        assert result.ndim == 1
        assert len(result) == 3

    def test_predict_values_in_range(self):
        """Probabilities should be in [0, 1]."""
        model, mock_session = self._make_model()
        mock_session.run.return_value = [np.array([[0.0], [0.5], [1.0]])]

        frames = np.zeros((3, 27, 48, 3), dtype=np.uint8)
        result = model.predict(frames)

        assert all(0.0 <= p <= 1.0 for p in result)

    def test_input_name_from_session(self):
        """Model uses the input name from the ONNX session."""
        mock_session = MagicMock()
        mock_input = MagicMock()
        mock_input.name = "my_custom_input"
        mock_session.get_inputs.return_value = [mock_input]
        mock_session.run.return_value = [np.array([[0.5]])]

        with patch("app.models.transnetv2.ort.InferenceSession", return_value=mock_session):
            from app.models.transnetv2 import TransNetV2
            model = TransNetV2("fake.onnx")

        frames = np.zeros((1, 27, 48, 3), dtype=np.uint8)
        model.predict(frames)

        call_args = mock_session.run.call_args
        feed_dict = call_args[0][1]
        assert "my_custom_input" in feed_dict

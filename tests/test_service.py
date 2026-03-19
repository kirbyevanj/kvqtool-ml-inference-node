"""Unit tests for ml-inference-node service module."""
import sys
import os
import numpy as np
import pytest
from unittest.mock import MagicMock, patch, PropertyMock

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


class TestDecodeFrameBatch:
    """Tests for the decode_frame_batch function."""

    def test_single_frame(self):
        """Decode a single raw frame into correct shape."""
        from app.service import decode_frame_batch

        width, height = 48, 27
        raw = np.zeros((height * width * 3,), dtype=np.uint8).tobytes()
        result = decode_frame_batch([raw], width, height)

        assert result.shape == (1, height, width, 3)
        assert result.dtype == np.uint8

    def test_multiple_frames(self):
        """Decode multiple frames at once."""
        from app.service import decode_frame_batch

        width, height = 48, 27
        n = 5
        raw = [np.zeros((height * width * 3,), dtype=np.uint8).tobytes() for _ in range(n)]
        result = decode_frame_batch(raw, width, height)

        assert result.shape == (n, height, width, 3)

    def test_frame_values_preserved(self):
        """Ensure pixel values are preserved after decoding."""
        from app.service import decode_frame_batch

        width, height = 4, 3
        arr = np.arange(height * width * 3, dtype=np.uint8)
        raw = arr.tobytes()
        result = decode_frame_batch([raw], width, height)

        assert result.shape == (1, height, width, 3)
        expected = arr.reshape(height, width, 3)
        np.testing.assert_array_equal(result[0], expected)

    def test_different_resolution(self):
        """Decode frames at a non-standard resolution."""
        from app.service import decode_frame_batch

        width, height = 16, 9
        raw = np.zeros((height * width * 3,), dtype=np.uint8).tobytes()
        result = decode_frame_batch([raw], width, height)

        assert result.shape == (1, height, width, 3)

    def test_empty_batch(self):
        """Empty batch raises ValueError (numpy can't stack zero arrays)."""
        from app.service import decode_frame_batch
        import numpy as np

        with pytest.raises((ValueError, Exception)):
            decode_frame_batch([], 48, 27)

    def test_rgb_channel_order(self):
        """Channels are preserved in RGB order."""
        from app.service import decode_frame_batch

        width, height = 1, 1
        # R=10, G=20, B=30
        raw = bytes([10, 20, 30])
        result = decode_frame_batch([raw], width, height)

        assert result[0, 0, 0, 0] == 10  # R
        assert result[0, 0, 0, 1] == 20  # G
        assert result[0, 0, 0, 2] == 30  # B


class TestInferenceServicer:
    """Tests for InferenceServicer gRPC service."""

    def _make_servicer_no_model(self):
        """Create a servicer with no model loaded."""
        with patch("app.service.os.path.exists", return_value=False):
            from app.service import InferenceServicer
            return InferenceServicer()

    def _make_servicer_with_mock_model(self):
        """Create a servicer with a mocked TransNetV2 model."""
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.1, 0.9, 0.2])

        with patch("app.service.os.path.exists", return_value=True), \
             patch("app.service.TransNetV2", return_value=mock_model):
            from app.service import InferenceServicer
            servicer = InferenceServicer()
            servicer.transnet = mock_model
            return servicer, mock_model

    def test_detect_scenes_no_model_aborts(self):
        """DetectScenes returns UNAVAILABLE when model not loaded."""
        import importlib
        import app.service
        importlib.reload(app.service)
        from app.service import InferenceServicer

        with patch("app.service.os.path.exists", return_value=False):
            servicer = InferenceServicer()

        import grpc
        context = MagicMock()

        import proto.inference_pb2 as pb2
        request = pb2.SceneDetectionRequest(frames=[], width=48, height=27)
        servicer.DetectScenes(request, context)
        context.abort.assert_called_once()
        args = context.abort.call_args[0]
        assert args[0] == grpc.StatusCode.UNAVAILABLE

    def test_detect_scenes_with_model(self):
        """DetectScenes returns probabilities with loaded model."""
        import importlib
        import app.service
        importlib.reload(app.service)

        mock_model = MagicMock()
        mock_predictions = np.array([0.1, 0.9, 0.2])
        mock_model.predict.return_value = mock_predictions

        with patch("app.service.os.path.exists", return_value=True), \
             patch("app.service.TransNetV2", return_value=mock_model):
            from app.service import InferenceServicer
            servicer = InferenceServicer()

        import proto.inference_pb2 as pb2
        context = MagicMock()

        width, height = 48, 27
        raw_frame = np.zeros((height * width * 3,), dtype=np.uint8).tobytes()
        request = pb2.SceneDetectionRequest(
            frames=[raw_frame, raw_frame, raw_frame],
            width=width,
            height=height,
        )

        response = servicer.DetectScenes(request, context)
        context.abort.assert_not_called()
        assert len(response.probabilities) == 3

    def test_compute_salience_unimplemented(self):
        """ComputeSalience always returns UNIMPLEMENTED."""
        import importlib
        import app.service
        importlib.reload(app.service)

        with patch("app.service.os.path.exists", return_value=False):
            from app.service import InferenceServicer
            servicer = InferenceServicer()

        import grpc
        import proto.inference_pb2 as pb2
        context = MagicMock()
        request = pb2.SalienceRequest()

        servicer.ComputeSalience(request, context)
        context.abort.assert_called_once()
        args = context.abort.call_args[0]
        assert args[0] == grpc.StatusCode.UNIMPLEMENTED

    def test_compute_salience_unimplemented_even_with_model(self):
        """ComputeSalience is UNIMPLEMENTED regardless of model state."""
        import importlib
        import app.service
        importlib.reload(app.service)

        mock_model = MagicMock()
        with patch("app.service.os.path.exists", return_value=True), \
             patch("app.service.TransNetV2", return_value=mock_model):
            from app.service import InferenceServicer
            servicer = InferenceServicer()

        import grpc
        import proto.inference_pb2 as pb2
        context = MagicMock()
        request = pb2.SalienceRequest()

        servicer.ComputeSalience(request, context)
        context.abort.assert_called_once()
        args = context.abort.call_args[0]
        assert args[0] == grpc.StatusCode.UNIMPLEMENTED

    def test_model_not_loaded_warning_logged(self):
        """When model path doesn't exist, a warning is logged."""
        import importlib
        import app.service
        importlib.reload(app.service)

        with patch("app.service.os.path.exists", return_value=False), \
             patch("app.service.logger") as mock_logger:
            from app.service import InferenceServicer
            servicer = InferenceServicer()
            assert servicer.transnet is None
            mock_logger.warning.assert_called_once()

    def test_model_loaded_when_path_exists(self):
        """When model path exists, TransNetV2 is instantiated."""
        import importlib
        import app.service
        importlib.reload(app.service)

        mock_model = MagicMock()
        with patch("app.service.os.path.exists", return_value=True), \
             patch("app.service.TransNetV2", return_value=mock_model) as MockT:
            from app.service import InferenceServicer
            servicer = InferenceServicer()
            MockT.assert_called_once()
            assert servicer.transnet is mock_model

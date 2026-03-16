import logging
import numpy as np
import onnxruntime as ort

logger = logging.getLogger(__name__)


class TransNetV2:
    INPUT_WIDTH = 48
    INPUT_HEIGHT = 27

    def __init__(self, model_path: str):
        self.session = ort.InferenceSession(
            model_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        logger.info("TransNetV2 loaded from %s", model_path)

    def predict(self, frames: np.ndarray) -> np.ndarray:
        """Run inference on a batch of frames.

        Args:
            frames: uint8 array of shape (N, 27, 48, 3) in RGB.

        Returns:
            Array of shape (N,) with scene boundary probabilities.
        """
        if frames.dtype != np.float32:
            frames = frames.astype(np.float32) / 255.0

        outputs = self.session.run(None, {self.input_name: frames})
        predictions = outputs[0]

        if predictions.ndim == 2:
            return predictions[:, 0]
        return predictions.flatten()

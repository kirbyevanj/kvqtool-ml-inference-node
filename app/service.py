import logging
import os
import sys

import grpc
import numpy as np
from concurrent import futures

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.models.transnetv2 import TransNetV2

import proto.inference_pb2 as pb2
import proto.inference_pb2_grpc as pb2_grpc

logger = logging.getLogger(__name__)

MODEL_PATH = os.environ.get("TRANSNETV2_MODEL_PATH", "/models/transnetv2.onnx")
GRPC_PORT = int(os.environ.get("GRPC_PORT", "50051"))
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "4"))


class InferenceServicer(pb2_grpc.InferenceServiceServicer):
    def __init__(self):
        self.transnet = None
        if os.path.exists(MODEL_PATH):
            self.transnet = TransNetV2(MODEL_PATH)
        else:
            logger.warning("TransNetV2 model not found at %s", MODEL_PATH)

    def DetectScenes(self, request, context):
        if self.transnet is None:
            context.abort(grpc.StatusCode.UNAVAILABLE, "TransNetV2 model not loaded")
            return pb2.SceneDetectionResponse()

        frames = decode_frame_batch(
            request.frames, request.width, request.height
        )
        probabilities = self.transnet.predict(frames)

        return pb2.SceneDetectionResponse(
            probabilities=probabilities.tolist()
        )

    def ComputeSalience(self, request, context):
        context.abort(grpc.StatusCode.UNIMPLEMENTED, "Salience not yet implemented")
        return pb2.SalienceResponse()


def decode_frame_batch(
    raw_frames: list[bytes], width: int, height: int
) -> np.ndarray:
    batch = []
    for raw in raw_frames:
        arr = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)
        batch.append(arr)
    return np.stack(batch)


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=MAX_WORKERS))
    pb2_grpc.add_InferenceServiceServicer_to_server(
        InferenceServicer(), server
    )
    server.add_insecure_port(f"[::]:{GRPC_PORT}")
    server.start()
    logger.info("gRPC inference server started on port %d", GRPC_PORT)
    server.wait_for_termination()

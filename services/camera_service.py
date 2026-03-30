from __future__ import annotations

import logging
import threading
import time
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class CameraService:
    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self.capture: Optional[cv2.VideoCapture] = None
        self.running = False
        self.lock = threading.Lock()
        self.latest_frame: Optional[np.ndarray] = None
        self.thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self.running:
            return
        self.capture = cv2.VideoCapture(self.camera_index)
        if not self.capture.isOpened():
            raise RuntimeError("Unable to open camera")

        # Keep camera frames lightweight and avoid stale buffered frames.
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.running = True
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()
        logger.info("Camera thread started")

    def _read_loop(self) -> None:
        while self.running and self.capture is not None:
            success, frame = self.capture.read()
            if not success:
                time.sleep(0.05)
                continue

            with self.lock:
                self.latest_frame = frame

            time.sleep(0.005)

    def get_frame(self) -> Optional[np.ndarray]:
        with self.lock:
            if self.latest_frame is None:
                return None
            return self.latest_frame.copy()

    def stop(self) -> None:
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)

        if self.capture is not None:
            self.capture.release()
            self.capture = None

        logger.info("Camera stopped")

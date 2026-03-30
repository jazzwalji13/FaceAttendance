from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

from utils.config import CASCADE_PATH

logger = logging.getLogger(__name__)


class FaceEngine:
    def __init__(self) -> None:
        self.detector = self._load_detector()
        self.embedding_mode = self._resolve_embedding_mode()

    def _load_detector(self) -> cv2.CascadeClassifier:
        cascade_path = CASCADE_PATH
        detector = cv2.CascadeClassifier(cascade_path)
        if detector.empty():
            fallback = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            detector = cv2.CascadeClassifier(fallback)

        if detector.empty():
            raise RuntimeError("Unable to load Haar cascade for face detection")
        return detector

    def _resolve_embedding_mode(self) -> str:
        # Try deepface first (best accuracy if available)
        try:
            import deepface  # noqa: F401
            logger.info("Embedding backend: deepface (Facenet512)")
            return "deepface"
        except ImportError:
            logger.debug("deepface not available (requires tensorflow)")

        # Fallback to face_recognition if available
        try:
            import face_recognition  # noqa: F401
            logger.info("Embedding backend: face_recognition (dlib)")
            return "face_recognition"
        except ImportError:
            logger.debug("face_recognition not available (requires dlib)")

        # Default to ORB: pure OpenCV, no external dependencies
        logger.info("Embedding backend: ORB (OpenCV features - default/stable)")
        return "orb"

    def get_backend_diagnostics(self) -> dict:
        info: dict[str, str] = {
            "active_backend": self.embedding_mode,
            "deepface_status": "unknown",
            "face_recognition_status": "unknown",
            "notes": "",
        }

        try:
            import deepface  # noqa: F401

            info["deepface_status"] = "available"
        except Exception as exc:
            info["deepface_status"] = f"unavailable: {exc}"

        try:
            import face_recognition  # noqa: F401

            info["face_recognition_status"] = "available"
        except Exception as exc:
            info["face_recognition_status"] = f"unavailable: {exc}"

        if self.embedding_mode == "orb":
            info["notes"] = "Using ORB fallback. For best face ID accuracy, enable face_recognition backend."
        elif self.embedding_mode == "orb_legacy":
            info["notes"] = "Using ORB legacy mode for existing 4096-d samples. Capture fresh samples for best results."
        elif self.embedding_mode == "deepface":
            info["notes"] = "Using deepface embeddings. Accuracy is usually good but heavier on CPU/GPU."
        else:
            info["notes"] = "Using face_recognition backend (recommended for this project)."

        return info

    @staticmethod
    def is_face_recognition_available() -> bool:
        try:
            import face_recognition  # noqa: F401

            return True
        except Exception:
            return False

    def detect_faces(self, frame_bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80))
        return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]

    def extract_embedding(self, frame_bgr: np.ndarray, box: tuple[int, int, int, int]) -> Optional[np.ndarray]:
        x, y, w, h = box
        x2, y2 = x + w, y + h
        face_crop = frame_bgr[max(y, 0):min(y2, frame_bgr.shape[0]), max(x, 0):min(x2, frame_bgr.shape[1])]
        return self.extract_embedding_from_crop(face_crop)

    def extract_embedding_from_crop(self, face_crop_bgr: np.ndarray) -> Optional[np.ndarray]:
        return self.extract_embedding_from_crop_with_mode(face_crop_bgr, self.embedding_mode)

    def extract_embedding_from_crop_with_mode(self, face_crop_bgr: np.ndarray, mode: str) -> Optional[np.ndarray]:
        if face_crop_bgr is None or face_crop_bgr.size == 0:
            logger.warning("Face crop is empty")
            return None

        if mode == "deepface":
            try:
                from deepface import DeepFace

                reps = DeepFace.represent(
                    img_path=face_crop_bgr,
                    model_name="Facenet512",
                    enforce_detection=False,
                )
                if reps:
                    return np.asarray(reps[0]["embedding"], dtype=np.float32)
                return None
            except Exception as e:
                logger.debug("deepface extraction failed: %s", e)
                return None

        if mode == "face_recognition":
            try:
                import face_recognition

                rgb = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)
                encodings = face_recognition.face_encodings(rgb)
                if not encodings:
                    return None
                return np.asarray(encodings[0], dtype=np.float32)
            except Exception as e:
                logger.debug("face_recognition extraction failed: %s", e)
                return None

        if mode == "orb":
            return self._extract_orb_features_from_crop(face_crop_bgr)

        if mode == "orb_legacy":
            return self._extract_orb_legacy_features_from_crop(face_crop_bgr)

        return None

    @staticmethod
    def _extract_orb_features(frame_bgr: np.ndarray, box: tuple[int, int, int, int]) -> Optional[np.ndarray]:
        x, y, w, h = box
        x2, y2 = x + w, y + h
        face_crop = frame_bgr[max(y, 0):min(y2, frame_bgr.shape[0]), max(x, 0):min(x2, frame_bgr.shape[1])]
        if face_crop.size == 0:
            return None
        return FaceEngine._extract_orb_features_from_crop(face_crop)

    @staticmethod
    def _extract_orb_features_from_crop(face_crop_bgr: np.ndarray) -> Optional[np.ndarray]:
        if face_crop_bgr is None or face_crop_bgr.size == 0:
            return None
        orb = cv2.ORB_create(nfeatures=192)
        gray = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2GRAY)
        kp, des = orb.detectAndCompute(gray, None)
        if des is None or len(des) == 0:
            return None

        # Robust ORB descriptor aggregation: per-byte mean and std removes keypoint-order noise.
        des_float = des.astype(np.float32)
        mean_feat = np.mean(des_float, axis=0)
        std_feat = np.std(des_float, axis=0)
        feature = np.concatenate([mean_feat, std_feat], axis=0)
        norm = np.linalg.norm(feature)
        if norm > 0:
            feature = feature / norm
        return feature.astype(np.float32)

    @staticmethod
    def _extract_orb_legacy_features_from_crop(face_crop_bgr: np.ndarray) -> Optional[np.ndarray]:
        if face_crop_bgr is None or face_crop_bgr.size == 0:
            return None
        orb = cv2.ORB_create(nfeatures=128)
        gray = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2GRAY)
        _, des = orb.detectAndCompute(gray, None)
        if des is None or len(des) == 0:
            return None
        des_float = des.astype(np.float32)
        if des_float.shape[0] < 128:
            padding = np.zeros((128 - des_float.shape[0], 32), dtype=np.float32)
            des_float = np.vstack([des_float, padding])
        return des_float[:128].flatten().astype(np.float32)

    @staticmethod
    def generate_augmented_crops(face_crop_bgr: np.ndarray) -> list[np.ndarray]:
        if face_crop_bgr is None or face_crop_bgr.size == 0:
            return []

        h, w = face_crop_bgr.shape[:2]
        center = (w // 2, h // 2)

        def adjust_gamma(image: np.ndarray, gamma: float) -> np.ndarray:
            inv = 1.0 / max(gamma, 0.01)
            table = np.array([(i / 255.0) ** inv * 255 for i in range(256)], dtype=np.uint8)
            return cv2.LUT(image, table)

        base = face_crop_bgr
        rot_pos = cv2.warpAffine(base, cv2.getRotationMatrix2D(center, 6, 1.0), (w, h), borderMode=cv2.BORDER_REFLECT)
        rot_neg = cv2.warpAffine(base, cv2.getRotationMatrix2D(center, -6, 1.0), (w, h), borderMode=cv2.BORDER_REFLECT)
        flip = cv2.flip(base, 1)
        bright = cv2.convertScaleAbs(base, alpha=1.05, beta=10)
        dark = adjust_gamma(base, 1.2)

        crops = [base, rot_pos, rot_neg, flip, bright, dark]
        return crops

    @staticmethod
    def cosine_similarity(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
        denominator = (np.linalg.norm(vector_a) * np.linalg.norm(vector_b))
        if denominator == 0:
            return 0.0
        return float(np.dot(vector_a, vector_b) / denominator)

    @staticmethod
    def confidence_from_similarity(similarity: float) -> float:
        scaled = ((similarity + 1.0) / 2.0) * 100.0
        return float(max(0.0, min(100.0, scaled)))

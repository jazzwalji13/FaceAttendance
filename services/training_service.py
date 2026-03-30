from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np

from database.repository import AttendanceRepository
from utils.config import MODEL_PATH

logger = logging.getLogger(__name__)


class TrainingService:
    def __init__(self, repository: AttendanceRepository) -> None:
        self.repository = repository

    @staticmethod
    def _pick_algorithm(unique_student_count: int, embedding_count: int) -> str:
        if unique_student_count >= 4 and embedding_count >= 40:
            return "svm"
        return "knn"

    @staticmethod
    def _read_model_metadata() -> dict:
        if not Path(MODEL_PATH).exists():
            return {}
        try:
            payload = joblib.load(MODEL_PATH)
            if isinstance(payload, dict):
                return payload
        except Exception:
            logger.exception("Failed to read model metadata")
        return {}

    def ensure_model_ready(self, force_retrain: bool = False) -> tuple[bool, str, bool]:
        rows = self.repository.get_all_embeddings()
        if len(rows) < 5:
            return False, "Need at least 5 embeddings to auto-train", False

        labels = [student_id for student_id, _ in rows]
        unique_students = set(labels)
        if len(unique_students) < 2:
            return False, "Need embeddings from at least 2 students to auto-train", False

        metadata = self._read_model_metadata()
        model_embedding_count = int(metadata.get("embedding_count", 0))

        if not force_retrain and model_embedding_count == len(rows):
            return True, "Model already up to date", False

        algorithm = self._pick_algorithm(len(unique_students), len(rows))
        ok, message = self.train_classifier(algorithm=algorithm)
        if not ok:
            return False, message, False

        return True, f"{message} (auto)", True

    def train_classifier(self, algorithm: str = "knn") -> tuple[bool, str]:
        rows = self.repository.get_all_embeddings()
        if len(rows) < 5:
            return False, "Need at least 5 embeddings to train"

        labels = [student_id for student_id, _ in rows]
        vectors = np.array([embedding for _, embedding in rows], dtype=np.float32)

        unique_students = set(labels)
        if len(unique_students) < 2:
            return False, "Need embeddings from at least 2 students"

        if algorithm.lower() == "svm":
            try:
                from sklearn.svm import SVC
            except Exception as exc:
                return False, f"scikit-learn unavailable: {exc}"
            classifier = SVC(kernel="rbf", probability=True)
        else:
            try:
                from sklearn.neighbors import KNeighborsClassifier
            except Exception as exc:
                return False, f"scikit-learn unavailable: {exc}"
            classifier = KNeighborsClassifier(n_neighbors=3, metric="euclidean")

        classifier.fit(vectors, labels)

        payload = {
            "classifier": classifier,
            "algorithm": algorithm.lower(),
            "embedding_count": len(rows),
        }
        joblib.dump(payload, MODEL_PATH)
        logger.info("Model saved to %s", MODEL_PATH)

        return True, f"Training complete ({algorithm.upper()}) on {len(rows)} embeddings"

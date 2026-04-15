from __future__ import annotations

import json
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import joblib
import numpy as np

from database.repository import AttendanceRepository
from services.face_engine import FaceEngine
from utils.config import (
    CALIBRATION_PATH,
    CLASSIFIER_MIN_PROB,
    CLASSIFIER_STRONG_PROB,
    COSINE_MARGIN,
    DEEPFACE_COSINE_THRESHOLD,
    FACE_DISTANCE_MARGIN,
    FACE_DISTANCE_THRESHOLD,
    MIN_SAMPLES_PER_STUDENT,
    MIN_MARK_CONFIDENCE,
    MODEL_PATH,
    ORB_COSINE_THRESHOLD,
    ATTENDANCE_COOLDOWN_SECONDS,
)

try:
    import face_recognition
except Exception:  # pragma: no cover - optional dependency runtime guard
    face_recognition = None

logger = logging.getLogger(__name__)


class RecognitionService:
    def __init__(self, repository: AttendanceRepository, face_engine: FaceEngine):
        self.repository = repository
        self.face_engine = face_engine
        self.threshold = FACE_DISTANCE_THRESHOLD
        self.face_distance_threshold = FACE_DISTANCE_THRESHOLD
        self.face_distance_margin = FACE_DISTANCE_MARGIN
        self.deepface_cosine_threshold = DEEPFACE_COSINE_THRESHOLD
        self.orb_cosine_threshold = ORB_COSINE_THRESHOLD
        self.cosine_margin = COSINE_MARGIN
        self.min_mark_confidence = MIN_MARK_CONFIDENCE
        self.per_student_distance_thresholds: dict[str, float] = {}
        self.per_student_cosine_thresholds: dict[str, float] = {}
        self.known_student_ids: list[str] = []
        self.known_encodings: np.ndarray = np.empty((0, 0), dtype=np.float32)
        self.student_vectors: dict[str, np.ndarray] = {}
        self.classifier = None
        self.classifier_dim: Optional[int] = None
        self._marked_today_cache: set[str] = set()
        self._marked_cache_date = date.today().isoformat()
        self._last_marked_lookup: dict[str, datetime] = {}
        self._load_calibration()
        self.refresh_model()

    def _load_calibration(self) -> None:
        path = Path(CALIBRATION_PATH)
        if not path.exists():
            return

        try:
            with path.open("r", encoding="utf-8") as f:
                payload = json.load(f)

            self.face_distance_threshold = float(payload.get("face_distance_threshold", self.face_distance_threshold))
            self.face_distance_margin = float(payload.get("face_distance_margin", self.face_distance_margin))
            self.deepface_cosine_threshold = float(payload.get("deepface_cosine_threshold", self.deepface_cosine_threshold))
            self.orb_cosine_threshold = float(payload.get("orb_cosine_threshold", self.orb_cosine_threshold))
            self.cosine_margin = float(payload.get("cosine_margin", self.cosine_margin))
            self.min_mark_confidence = float(payload.get("min_mark_confidence", self.min_mark_confidence))
            self.per_student_distance_thresholds = {
                str(k): float(v)
                for k, v in payload.get("per_student_distance_thresholds", {}).items()
            }
            self.per_student_cosine_thresholds = {
                str(k): float(v)
                for k, v in payload.get("per_student_cosine_thresholds", {}).items()
            }
            self.threshold = self.face_distance_threshold
        except Exception:
            logger.debug("Could not load calibration file", exc_info=True)

    def _save_calibration(self) -> None:
        payload = {
            "face_distance_threshold": self.face_distance_threshold,
            "face_distance_margin": self.face_distance_margin,
            "deepface_cosine_threshold": self.deepface_cosine_threshold,
            "orb_cosine_threshold": self.orb_cosine_threshold,
            "cosine_margin": self.cosine_margin,
            "min_mark_confidence": self.min_mark_confidence,
            "per_student_distance_thresholds": self.per_student_distance_thresholds,
            "per_student_cosine_thresholds": self.per_student_cosine_thresholds,
            "updated_at": date.today().isoformat(),
        }

        try:
            path = Path(CALIBRATION_PATH)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        except Exception:
            logger.debug("Could not save calibration file", exc_info=True)

    def reset_calibration(self) -> dict:
        self.face_distance_threshold = FACE_DISTANCE_THRESHOLD
        self.face_distance_margin = FACE_DISTANCE_MARGIN
        self.deepface_cosine_threshold = DEEPFACE_COSINE_THRESHOLD
        self.orb_cosine_threshold = ORB_COSINE_THRESHOLD
        self.cosine_margin = COSINE_MARGIN
        self.min_mark_confidence = MIN_MARK_CONFIDENCE
        self.per_student_distance_thresholds = {}
        self.per_student_cosine_thresholds = {}
        self.threshold = self.face_distance_threshold

        try:
            path = Path(CALIBRATION_PATH)
            if path.exists():
                path.unlink()
        except Exception:
            logger.debug("Could not delete calibration file", exc_info=True)

        self._save_calibration()
        return {
            "updated": True,
            "face_distance_threshold": round(self.face_distance_threshold, 4),
            "face_distance_margin": round(self.face_distance_margin, 4),
            "deepface_cosine_threshold": round(self.deepface_cosine_threshold, 4),
            "orb_cosine_threshold": round(self.orb_cosine_threshold, 4),
            "cosine_margin": round(self.cosine_margin, 4),
            "min_mark_confidence": round(self.min_mark_confidence, 2),
        }

    def calibrate_personalized_thresholds(self, target_student_id: Optional[str] = None) -> dict:
        self.refresh_model()
        if len(self.student_vectors) < 2:
            return {
                "updated": False,
                "reason": "Need at least 2 students with samples",
            }

        student_ids = [target_student_id] if target_student_id else list(self.student_vectors.keys())
        student_ids = [sid for sid in student_ids if sid in self.student_vectors]
        if not student_ids:
            return {
                "updated": False,
                "reason": "Selected student has no usable samples",
            }

        mode = self.face_engine.embedding_mode
        updates = 0

        centroids = {sid: np.mean(v, axis=0) for sid, v in self.student_vectors.items()}

        for sid in student_ids:
            vectors = self.student_vectors[sid]
            if vectors.shape[0] < 2:
                continue

            other_ids = [x for x in self.student_vectors.keys() if x != sid]
            if not other_ids:
                continue

            if mode == "face_recognition" and face_recognition is not None:
                intra = self._pairwise_distances(vectors)
                if intra.size == 0:
                    continue

                own_centroid = centroids[sid]
                other_centroids = np.vstack([centroids[x] for x in other_ids])
                inter = face_recognition.face_distance(other_centroids, own_centroid)
                nearest_other = float(np.min(inter)) if inter.size else 1.0

                base = float(np.percentile(intra, 92) + 0.02)
                safe_upper = float(nearest_other - max(self.face_distance_margin, 0.02))
                personalized = float(max(0.34, min(0.55, safe_upper, base)))
                self.per_student_distance_thresholds[sid] = personalized
                updates += 1
            else:
                intra = self._pairwise_cosine(vectors)
                if intra.size == 0:
                    continue

                own_centroid = centroids[sid]
                own_norm = np.linalg.norm(own_centroid)
                other_centroids = np.vstack([centroids[x] for x in other_ids])
                other_norms = np.linalg.norm(other_centroids, axis=1)
                denom = other_norms * max(own_norm, 1e-10)
                denom[denom == 0.0] = 1e-10
                inter = np.dot(other_centroids, own_centroid) / denom
                nearest_other = float(np.max(inter)) if inter.size else 0.0

                base = float(np.percentile(intra, 10) - 0.02)
                safe_lower = min(0.95, nearest_other + max(self.cosine_margin, 0.03))
                personalized = float(min(0.93, max(safe_lower, base)))
                self.per_student_cosine_thresholds[sid] = personalized
                updates += 1

        if updates == 0:
            return {
                "updated": False,
                "reason": "Not enough per-student variation for personalization",
            }

        self._save_calibration()
        return {
            "updated": True,
            "mode": mode,
            "students_updated": updates,
            "target_student": target_student_id or "ALL",
        }

    def _refresh_classifier(self) -> None:
        self.classifier = None
        self.classifier_dim = None

        if not Path(MODEL_PATH).exists():
            return

        try:
            payload = joblib.load(MODEL_PATH)
            if not isinstance(payload, dict):
                return
            classifier = payload.get("classifier")
            if classifier is None:
                return
            self.classifier = classifier

            # Infer classifier expected embedding dimension from a support vector/neighbor matrix if possible.
            if hasattr(classifier, "n_features_in_"):
                self.classifier_dim = int(getattr(classifier, "n_features_in_"))
        except Exception:
            logger.debug("Classifier refresh skipped", exc_info=True)

    def _refresh_marked_cache_date(self) -> None:
        today_iso = date.today().isoformat()
        if today_iso != self._marked_cache_date:
            self._marked_cache_date = today_iso
            self._marked_today_cache = self.repository.get_marked_students_for_date(today_iso)
            self._last_marked_lookup = {}

    def warm_marked_today_cache(self) -> None:
        today_iso = date.today().isoformat()
        self._marked_cache_date = today_iso
        self._marked_today_cache = self.repository.get_marked_students_for_date(today_iso)
        self._last_marked_lookup = {}

    def refresh_model(self) -> None:
        rows = self.repository.get_all_embeddings()
        valid_ids: list[str] = []
        valid_encodings: list[np.ndarray] = []

        for student_id, encoding in rows:
            vector = np.asarray(encoding, dtype=np.float32)
            if vector.size == 0:
                continue
            valid_ids.append(student_id)
            valid_encodings.append(vector)

        if not valid_encodings:
            self.known_student_ids = []
            self.known_encodings = np.empty((0, 0), dtype=np.float32)
            self.student_vectors = {}
            self.classifier = None
            self.classifier_dim = None
            return

        # Keep vectors that match the active backend embedding dimension.
        lengths = [int(v.size) for v in valid_encodings]
        has_128 = any(length == 128 for length in lengths)
        has_legacy_4096 = any(length == 4096 for length in lengths)

        if self.face_engine.embedding_mode == "face_recognition":
            if has_128:
                target_len = 128
            elif has_legacy_4096:
                self.face_engine.embedding_mode = "orb_legacy"
                target_len = 4096
                logger.warning("No 128-d samples found. Auto-switched to legacy ORB compatibility mode.")
            else:
                target_len = max(set(lengths), key=lengths.count)
        elif self.face_engine.embedding_mode == "orb_legacy":
            if has_128 and self.face_engine.is_face_recognition_available():
                self.face_engine.embedding_mode = "face_recognition"
                target_len = 128
                logger.info("Detected 128-d samples. Switched back to face_recognition mode.")
            elif has_legacy_4096:
                target_len = 4096
            else:
                target_len = max(set(lengths), key=lengths.count)
        else:
            target_len = max(set(lengths), key=lengths.count)

        filtered_ids: list[str] = []
        filtered_vectors: list[np.ndarray] = []
        for student_id, vector in zip(valid_ids, valid_encodings):
            if int(vector.size) == target_len:
                filtered_ids.append(student_id)
                filtered_vectors.append(vector)

        if not filtered_vectors:
            self.known_student_ids = []
            self.known_encodings = np.empty((0, 0), dtype=np.float32)
            self.student_vectors = {}
            self.classifier = None
            self.classifier_dim = None
            return

        self.known_student_ids = filtered_ids
        self.known_encodings = np.vstack(filtered_vectors).astype(np.float32)

        grouped: dict[str, list[np.ndarray]] = {}
        for student_id, vector in zip(filtered_ids, filtered_vectors):
            grouped.setdefault(student_id, []).append(vector)

        self.student_vectors = {
            student_id: np.vstack(vectors).astype(np.float32)
            for student_id, vectors in grouped.items()
            if len(vectors) >= MIN_SAMPLES_PER_STUDENT
        }

        if not self.student_vectors:
            logger.warning(
                "No student has enough samples for recognition (need >= %s each)",
                MIN_SAMPLES_PER_STUDENT,
            )
            self.known_student_ids = []
            self.known_encodings = np.empty((0, 0), dtype=np.float32)
            self.classifier = None
            self.classifier_dim = None
            return

        self._refresh_classifier()

    @staticmethod
    def _topk_mean(values: np.ndarray, smallest: bool, k: int = 3) -> float:
        if values.size == 0:
            return 1.0 if smallest else -1.0
        k_eff = min(k, values.size)
        if smallest:
            part = np.partition(values, k_eff - 1)[:k_eff]
        else:
            part = np.partition(values, values.size - k_eff)[-k_eff:]
        return float(np.mean(part))

    @staticmethod
    def _pairwise_distances(vectors: np.ndarray) -> np.ndarray:
        if vectors.shape[0] < 2:
            return np.empty((0,), dtype=np.float32)
        diffs = vectors[:, None, :] - vectors[None, :, :]
        dist_matrix = np.sqrt(np.sum(diffs * diffs, axis=2))
        triu = np.triu_indices(dist_matrix.shape[0], k=1)
        return dist_matrix[triu].astype(np.float32)

    @staticmethod
    def _pairwise_cosine(vectors: np.ndarray) -> np.ndarray:
        if vectors.shape[0] < 2:
            return np.empty((0,), dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1)
        denom = norms[:, None] * norms[None, :]
        denom[denom == 0.0] = 1e-10
        sim_matrix = np.dot(vectors, vectors.T) / denom
        triu = np.triu_indices(sim_matrix.shape[0], k=1)
        return sim_matrix[triu].astype(np.float32)

    def calibrate_thresholds(self) -> dict:
        self.refresh_model()
        student_ids = list(self.student_vectors.keys())
        if len(student_ids) < 2:
            return {
                "updated": False,
                "reason": "Need at least 2 students with samples for calibration",
            }

        total_vectors = sum(v.shape[0] for v in self.student_vectors.values())
        if total_vectors < 8:
            return {
                "updated": False,
                "reason": "Need at least 8 total samples for stable calibration",
            }

        mode = self.face_engine.embedding_mode
        if mode == "face_recognition" and face_recognition is not None:
            intra_distances: list[np.ndarray] = []
            centroids: dict[str, np.ndarray] = {}

            for sid, vectors in self.student_vectors.items():
                intra = self._pairwise_distances(vectors)
                if intra.size > 0:
                    intra_distances.append(intra)
                centroids[sid] = np.mean(vectors, axis=0)

            if not intra_distances:
                return {
                    "updated": False,
                    "reason": "Need at least one student with 2+ samples",
                }

            intra_all = np.concatenate(intra_distances)
            centroid_stack = np.vstack([centroids[sid] for sid in student_ids])
            centroid_dists = self._pairwise_distances(centroid_stack)
            inter_med = float(np.median(centroid_dists)) if centroid_dists.size else 1.0

            new_threshold = float(np.percentile(intra_all, 88) + 0.015)
            new_threshold = float(max(0.34, min(0.52, new_threshold)))

            intra_med = float(np.median(intra_all))
            new_margin = float((inter_med - intra_med) * 0.30)
            new_margin = float(max(0.02, min(0.10, new_margin)))

            self.face_distance_threshold = new_threshold
            self.face_distance_margin = new_margin
            self.threshold = new_threshold
            self._save_calibration()

            return {
                "updated": True,
                "mode": mode,
                "face_distance_threshold": round(self.face_distance_threshold, 4),
                "face_distance_margin": round(self.face_distance_margin, 4),
                "students": len(student_ids),
                "samples": total_vectors,
            }

        # deepface/orb calibration path
        intra_sims: list[np.ndarray] = []
        centroids: dict[str, np.ndarray] = {}
        for sid, vectors in self.student_vectors.items():
            sims = self._pairwise_cosine(vectors)
            if sims.size > 0:
                intra_sims.append(sims)
            centroids[sid] = np.mean(vectors, axis=0)

        if not intra_sims:
            return {
                "updated": False,
                "reason": "Need at least one student with 2+ samples",
            }

        intra_all = np.concatenate(intra_sims)
        centroid_stack = np.vstack([centroids[sid] for sid in student_ids])

        centroid_norms = np.linalg.norm(centroid_stack, axis=1)
        denom = centroid_norms[:, None] * centroid_norms[None, :]
        denom[denom == 0.0] = 1e-10
        centroid_sims = np.dot(centroid_stack, centroid_stack.T) / denom
        tri = np.triu_indices(centroid_sims.shape[0], k=1)
        inter_all = centroid_sims[tri]
        inter_med = float(np.median(inter_all)) if inter_all.size else 0.0

        new_cosine_threshold = float(np.percentile(intra_all, 12) - 0.02)
        if mode == "orb":
            new_cosine_threshold = float(max(0.72, min(0.90, new_cosine_threshold)))
            self.orb_cosine_threshold = new_cosine_threshold
        else:
            new_cosine_threshold = float(max(0.60, min(0.88, new_cosine_threshold)))
            self.deepface_cosine_threshold = new_cosine_threshold

        intra_med = float(np.median(intra_all))
        new_margin = float((intra_med - inter_med) * 0.35)
        self.cosine_margin = float(max(0.03, min(0.14, new_margin)))
        self._save_calibration()

        return {
            "updated": True,
            "mode": mode,
            "cosine_threshold": round(new_cosine_threshold, 4),
            "cosine_margin": round(self.cosine_margin, 4),
            "students": len(student_ids),
            "samples": total_vectors,
        }

    def predict_identity(self, embedding: np.ndarray) -> tuple[Optional[str], float, float]:
        if embedding.size == 0:
            return None, 1.0, 0.0

        if self.known_encodings.size == 0:
            self.refresh_model()
            if self.known_encodings.size == 0:
                return None, 1.0, 0.0

        query = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if query.size != self.known_encodings.shape[1]:
            return None, 1.0, 0.0

        use_face_distance = self.face_engine.embedding_mode == "face_recognition" and face_recognition is not None

        if use_face_distance:
            student_scores: list[tuple[str, float]] = []
            for student_id, vectors in self.student_vectors.items():
                dists = face_recognition.face_distance(vectors, query)
                score = self._topk_mean(dists, smallest=True, k=3)
                student_scores.append((student_id, score))

            if not student_scores:
                return None, 1.0, 0.0

            student_scores.sort(key=lambda item: item[1])
            best_match_id, best_distance = student_scores[0]
            second_best_distance = student_scores[1][1] if len(student_scores) > 1 else 1.0

            confidence = float(max(0.0, min(100.0, (1.0 - best_distance) * 100.0)))
            personal_threshold = self.per_student_distance_thresholds.get(best_match_id, self.face_distance_threshold)
            is_confident = best_distance <= personal_threshold

            # Dynamic confusion guard: require bigger separation when candidate is farther away.
            dynamic_margin = max(self.face_distance_margin, min(0.11, 0.045 + (best_distance * 0.08)))
            has_margin = (second_best_distance - best_distance) >= dynamic_margin

            # Reject ambiguous predictions near threshold regardless of margin.
            near_threshold = best_distance >= (personal_threshold - 0.015)
            weak_runner_up = second_best_distance <= (personal_threshold + 0.025)
            if near_threshold and weak_runner_up:
                return None, best_distance, confidence

            if is_confident and has_margin:
                return best_match_id, best_distance, confidence
            return None, best_distance, confidence

        # Fallback for deepface/orb embeddings where cosine similarity is more stable.
        query_norm = np.linalg.norm(query)
        if query_norm == 0.0:
            return None, 1.0, 0.0

        student_scores: list[tuple[str, float]] = []
        for student_id, vectors in self.student_vectors.items():
            known_norms = np.linalg.norm(vectors, axis=1)
            denom = known_norms * query_norm
            denom[denom == 0.0] = 1e-10
            similarities = np.dot(vectors, query) / denom
            score = self._topk_mean(similarities, smallest=False, k=3)
            student_scores.append((student_id, score))

        if not student_scores:
            return None, 1.0, 0.0

        student_scores.sort(key=lambda item: item[1], reverse=True)
        best_match_id, best_similarity = student_scores[0]
        second_best_similarity = student_scores[1][1] if len(student_scores) > 1 else -1.0

        similarity_confidence = self.face_engine.confidence_from_similarity(best_similarity)
        pseudo_distance = float(1.0 - best_similarity)

        classifier_id: Optional[str] = None
        classifier_prob = 0.0
        if self.classifier is not None and (self.classifier_dim is None or self.classifier_dim == int(query.size)):
            try:
                classifier_id = str(self.classifier.predict([query])[0])
                if hasattr(self.classifier, "predict_proba"):
                    probs = self.classifier.predict_proba([query])[0]
                    classifier_prob = float(np.max(probs))
            except Exception:
                logger.debug("Classifier prediction failed", exc_info=True)

        if self.face_engine.embedding_mode == "deepface":
            similarity_threshold = self.deepface_cosine_threshold
        elif self.face_engine.embedding_mode in ("orb", "orb_legacy"):
            similarity_threshold = self.orb_cosine_threshold
        else:
            similarity_threshold = self.deepface_cosine_threshold

        similarity_threshold = self.per_student_cosine_thresholds.get(best_match_id, similarity_threshold)

        is_confident = best_similarity >= similarity_threshold
        has_margin = (best_similarity - second_best_similarity) >= self.cosine_margin

        # Fuse metric confidence with classifier confidence to reduce "Unknown" for true positives.
        combined_confidence = similarity_confidence
        if classifier_prob > 0.0:
            combined_confidence = (0.65 * similarity_confidence) + (0.35 * (classifier_prob * 100.0))

        if classifier_id is not None and classifier_id == best_match_id:
            if (is_confident and has_margin) or classifier_prob >= CLASSIFIER_MIN_PROB:
                return best_match_id, pseudo_distance, float(max(0.0, min(100.0, combined_confidence)))

        # Allow very strong classifier confidence when metric score is near threshold.
        if classifier_id is not None and classifier_prob >= CLASSIFIER_STRONG_PROB and best_similarity >= (similarity_threshold - 0.03):
            return classifier_id, pseudo_distance, float(max(0.0, min(100.0, combined_confidence)))

        if is_confident and has_margin:
            return best_match_id, pseudo_distance, float(max(0.0, min(100.0, combined_confidence)))

        return None, pseudo_distance, float(max(0.0, min(100.0, combined_confidence)))

    def mark_if_eligible(self, student_id: str, confidence: float) -> tuple[bool, str]:
        self._refresh_marked_cache_date()

        # A strict calibration value (e.g. 78%) can block valid recognitions from being logged.
        # Cap it to a practical runtime threshold and rely on identity distance/margin checks for safety.
        effective_min_confidence = min(self.min_mark_confidence, 55.0)

        if confidence < effective_min_confidence:
            return False, (
                f"Recognition confidence too low ({confidence:.1f}% < {effective_min_confidence:.1f}%)"
            )

        if student_id in self._marked_today_cache:
            today_iso = date.today().isoformat()
            if self.repository.has_marked_today(student_id, today_iso):
                updated = self.repository.update_today_attendance(student_id, confidence, status="Present")
                if updated:
                    now = datetime.now()
                    self._last_marked_lookup[student_id] = now
                    return True, "Attendance updated"
                # Cache can be stale when logs are deleted while app is running.
                self._marked_today_cache.discard(student_id)
                self._last_marked_lookup.pop(student_id, None)
            # Cache can go stale if rows are deleted/reset while app is running.
            self._marked_today_cache.discard(student_id)
            self._last_marked_lookup.pop(student_id, None)

        last_marked_at = self._last_marked_lookup.get(student_id)
        if last_marked_at is None:
            last_marked_at = self.repository.get_last_attendance_timestamp(student_id)
            if last_marked_at is not None:
                self._last_marked_lookup[student_id] = last_marked_at

        if last_marked_at is not None:
            elapsed_seconds = (datetime.now() - last_marked_at).total_seconds()
            if elapsed_seconds < ATTENDANCE_COOLDOWN_SECONDS:
                wait_seconds = int(max(1, ATTENDANCE_COOLDOWN_SECONDS - elapsed_seconds))
                return False, f"Cooldown active ({wait_seconds}s remaining)"

        self.repository.mark_attendance(student_id, confidence, status="Present")
        self._marked_today_cache.add(student_id)
        self._last_marked_lookup[student_id] = datetime.now()
        return True, "Attendance marked"

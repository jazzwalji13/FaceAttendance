from __future__ import annotations

from pathlib import Path

from database.repository import AttendanceRepository
from services.face_engine import FaceEngine
from services.recognition_service import RecognitionService
from utils.config import CALIBRATION_PATH, MIN_SAMPLES_PER_STUDENT


class DiagnosticsService:
    def __init__(
        self,
        repository: AttendanceRepository,
        face_engine: FaceEngine,
        recognition_service: RecognitionService,
    ) -> None:
        self.repository = repository
        self.face_engine = face_engine
        self.recognition_service = recognition_service

    def build_report(self) -> dict:
        diag = self.face_engine.get_backend_diagnostics()
        students = self.repository.list_students()
        embedding_rows = self.repository.get_all_embeddings()

        per_student_counts: dict[str, int] = {}
        dim_breakdown: dict[int, int] = {}
        for student_id, embedding in embedding_rows:
            per_student_counts[student_id] = per_student_counts.get(student_id, 0) + 1
            dim = int(len(embedding))
            dim_breakdown[dim] = dim_breakdown.get(dim, 0) + 1

        lines = [
            f"Active backend: {diag.get('active_backend', 'unknown')}",
            f"deepface: {diag.get('deepface_status', 'unknown')}",
            f"face_recognition: {diag.get('face_recognition_status', 'unknown')}",
            f"Students registered: {len(students)}",
            f"Total samples: {len(embedding_rows)}",
            f"Backend-compatible samples: {self.recognition_service.known_encodings.shape[0]}",
            f"Calibration saved: {'Yes' if Path(CALIBRATION_PATH).exists() else 'No'}",
            f"Current distance threshold: {self.recognition_service.face_distance_threshold:.3f}",
            f"Current cosine threshold (deepface): {self.recognition_service.deepface_cosine_threshold:.3f}",
            f"Current cosine threshold (orb): {self.recognition_service.orb_cosine_threshold:.3f}",
            f"Current margin: {self.recognition_service.cosine_margin:.3f}",
            (
                "Personalized profiles: "
                f"{len(self.recognition_service.per_student_distance_thresholds) + len(self.recognition_service.per_student_cosine_thresholds)}"
            ),
        ]
        if diag.get("notes"):
            lines.append(str(diag["notes"]))

        if dim_breakdown:
            parts = [f"{dim}d={count}" for dim, count in sorted(dim_breakdown.items())]
            lines.append(f"Dimension breakdown: {', '.join(parts)}")

        recommended_setup = [
            "1) Use Python 3.10 or 3.11 for best face_recognition compatibility.",
            "2) In your virtual environment, run: pip install cmake dlib face_recognition",
            "3) Restart the app and check this page until face_recognition is available.",
            f"4) Capture at least {MIN_SAMPLES_PER_STUDENT}-10 clean samples per student.",
            "5) Use Live Attendance page for marking after sample collection.",
        ]

        return {
            "students": students,
            "per_student_counts": per_student_counts,
            "lines": lines,
            "recommended_setup": recommended_setup,
        }

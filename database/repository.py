import json
from datetime import datetime
from typing import Optional

from database.db import Database
from models.entities import AttendanceRecord, Student


class AttendanceRepository:
    def __init__(self, db: Database):
        self.db = db

    def upsert_student(self, student: Student) -> None:
        with self.db.connect() as conn:
            conn.execute(
                """
                INSERT INTO students(student_id, full_name, department, semester, email, phone)
                VALUES(?, ?, ?, ?, ?, ?)
                ON CONFLICT(student_id) DO UPDATE SET
                    full_name=excluded.full_name,
                    department=excluded.department,
                    semester=excluded.semester,
                    email=excluded.email,
                    phone=excluded.phone
                """,
                (
                    student.student_id,
                    student.full_name,
                    student.department,
                    student.semester,
                    student.email,
                    student.phone,
                ),
            )

    def delete_student_data(self, student_id: str) -> tuple[bool, int, int]:
        with self.db.connect() as conn:
            exists = conn.execute(
                "SELECT 1 FROM students WHERE student_id = ? LIMIT 1",
                (student_id,),
            ).fetchone()
            if not exists:
                return False, 0, 0

            embeddings_deleted = conn.execute(
                "DELETE FROM face_embeddings WHERE student_id = ?",
                (student_id,),
            ).rowcount
            attendance_deleted = conn.execute(
                "DELETE FROM attendance_logs WHERE student_id = ?",
                (student_id,),
            ).rowcount
            student_deleted = conn.execute(
                "DELETE FROM students WHERE student_id = ?",
                (student_id,),
            ).rowcount

            return student_deleted > 0, embeddings_deleted, attendance_deleted

    def list_students(self) -> list[dict]:
        with self.db.connect() as conn:
            rows = conn.execute("SELECT * FROM students ORDER BY full_name").fetchall()
            return [dict(r) for r in rows]

    def add_embedding(self, student_id: str, embedding: list[float]) -> None:
        with self.db.connect() as conn:
            conn.execute(
                "INSERT INTO face_embeddings(student_id, embedding) VALUES(?, ?)",
                (student_id, json.dumps(embedding)),
            )

    def list_embedding_samples(self, student_id: Optional[str] = None) -> list[dict]:
        query = "SELECT id, student_id, created_at FROM face_embeddings"
        params: tuple[str, ...] = ()
        if student_id:
            query += " WHERE student_id = ?"
            params = (student_id,)
        query += " ORDER BY created_at DESC, id DESC"

        with self.db.connect() as conn:
            rows = conn.execute(query, params).fetchall()
            return [dict(r) for r in rows]

    def delete_embedding_sample(self, sample_id: int) -> bool:
        with self.db.connect() as conn:
            deleted = conn.execute(
                "DELETE FROM face_embeddings WHERE id = ?",
                (sample_id,),
            ).rowcount
            return deleted > 0

    def delete_embeddings_for_student(self, student_id: str) -> int:
        with self.db.connect() as conn:
            return conn.execute(
                "DELETE FROM face_embeddings WHERE student_id = ?",
                (student_id,),
            ).rowcount

    def get_all_embeddings(self) -> list[tuple[str, list[float]]]:
        with self.db.connect() as conn:
            rows = conn.execute("SELECT student_id, embedding FROM face_embeddings").fetchall()
            return [(r["student_id"], json.loads(r["embedding"])) for r in rows]

    def list_embedding_records(self) -> list[dict]:
        with self.db.connect() as conn:
            rows = conn.execute("SELECT id, student_id, embedding, created_at FROM face_embeddings").fetchall()
            return [
                {
                    "id": int(r["id"]),
                    "student_id": r["student_id"],
                    "embedding": json.loads(r["embedding"]),
                    "created_at": r["created_at"],
                }
                for r in rows
            ]

    def delete_embeddings_by_ids(self, embedding_ids: list[int]) -> int:
        if not embedding_ids:
            return 0

        placeholders = ",".join(["?"] * len(embedding_ids))
        query = f"DELETE FROM face_embeddings WHERE id IN ({placeholders})"
        with self.db.connect() as conn:
            return conn.execute(query, tuple(embedding_ids)).rowcount

    def get_embeddings_for_student(self, student_id: str) -> list[list[float]]:
        with self.db.connect() as conn:
            rows = conn.execute(
                "SELECT embedding FROM face_embeddings WHERE student_id = ?",
                (student_id,),
            ).fetchall()
            return [json.loads(r["embedding"]) for r in rows]

    def get_student_name(self, student_id: str) -> Optional[str]:
        with self.db.connect() as conn:
            row = conn.execute(
                "SELECT full_name FROM students WHERE student_id = ?",
                (student_id,),
            ).fetchone()
            return row["full_name"] if row else None

    def has_marked_today(self, student_id: str, date_iso: str) -> bool:
        with self.db.connect() as conn:
            row = conn.execute(
                """
                SELECT 1 FROM attendance_logs
                WHERE student_id = ? AND DATE(timestamp) = DATE(?)
                LIMIT 1
                """,
                (student_id, date_iso),
            ).fetchone()
            return row is not None

    def mark_attendance(self, student_id: str, confidence: float, status: str = "Present") -> AttendanceRecord:
        timestamp = datetime.now().isoformat(timespec="seconds")
        with self.db.connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO attendance_logs(student_id, timestamp, confidence, status)
                VALUES(?, ?, ?, ?)
                """,
                (student_id, timestamp, confidence, status),
            )
            record_id = cursor.lastrowid

        return AttendanceRecord(
            id=record_id,
            student_id=student_id,
            timestamp=datetime.fromisoformat(timestamp),
            confidence=confidence,
            status=status,
        )

    def get_attendance(self, date_iso: Optional[str] = None, section: Optional[str] = None) -> list[dict]:
        query = """
            SELECT a.id, a.student_id, s.full_name, s.department, a.timestamp, a.confidence, a.status
            FROM attendance_logs a
            JOIN students s ON s.student_id = a.student_id
        """
        params: list[str] = []
        filters: list[str] = []

        if date_iso:
            filters.append("DATE(a.timestamp) = DATE(?)")
            params.append(date_iso)
        if section:
            filters.append("s.semester = ?")
            params.append(section)

        if filters:
            query += " WHERE " + " AND ".join(filters)

        query += " ORDER BY a.timestamp DESC"

        with self.db.connect() as conn:
            rows = conn.execute(query, tuple(params)).fetchall()
            return [dict(r) for r in rows]

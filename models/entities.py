from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class Student:
    student_id: str
    full_name: str
    department: str
    semester: str
    email: str
    phone: str


@dataclass
class EmbeddingRecord:
    id: Optional[int]
    student_id: str
    embedding: list[float]
    created_at: datetime


@dataclass
class AttendanceRecord:
    id: Optional[int]
    student_id: str
    timestamp: datetime
    confidence: float
    status: str

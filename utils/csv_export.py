import csv
from pathlib import Path
from typing import Iterable


def export_attendance_csv(rows: Iterable[dict], output_path: str) -> str:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    headers = ["id", "student_id", "full_name", "department", "timestamp", "confidence", "status"]

    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in headers})

    return str(path.resolve())

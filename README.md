# FaceAttendance

FaceAttendance is a desktop face-recognition attendance system built with Python, Tkinter, OpenCV, and SQLite.

It supports student registration, face-sample capture, model training, real-time attendance marking, and CSV export.

## Features

- Real-time camera feed with non-blocking UI
- Student registry with persistent SQLite storage
- Multi-backend embedding extraction with automatic fallback
- Training pipeline with KNN/SVM options
- Stability-based recognition to reduce false positives
- One-mark-per-day attendance protection
- Attendance log filtering and CSV export

## Tech Stack

- Python
- Tkinter
- OpenCV
- scikit-learn
- SQLite

## Project Structure

```text
FaceAttendance/
  frontend.py
  requirements.txt
  ui/
  services/
  database/
  models/
  utils/
  artifacts/
```

## Quick Start (Windows)

1. Install dependencies

```bash
cd d:\FaceAttendance
pip install -r requirements.txt
```

2. Run the app

```bash
python frontend.py
```

## Optional Backends

For higher-accuracy embeddings (optional):

```bash
pip install deepface tensorflow
```

Alternative backend:

```bash
pip install face-recognition
```

If optional packages are not installed, the app falls back to an OpenCV-based backend.

## Configuration

Tune recognition and stability settings in `utils/config.py`.

## Release Workflow

This repository includes a GitHub Actions workflow that creates a GitHub Release with a source zip whenever a semantic version tag is pushed.

Tag format:

- `v1.0.0`
- `v1.1.0`
- `v2.0.3`

Commands:

```bash
git tag v1.0.0
git push origin v1.0.0
```

The workflow will publish a release automatically for that tag.

## Troubleshooting

- Camera not opening: verify camera access and driver setup.
- Training fails: collect more samples per student before training.
- Recognition is slow: use the default OpenCV fallback backend.

## Notes

- Runtime logs are written to `app.log`.
- SQLite DB file is stored in `database/attendance.db`.

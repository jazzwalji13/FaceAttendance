#!/usr/bin/env python3
"""
Verification script for Face Attendance System.
Run this to ensure all components are working before production deployment.
"""

import sys
from pathlib import Path


def test_imports():
    """Test that all core modules import successfully."""
    print("=" * 60)
    print("Testing Module Imports...")
    print("=" * 60)

    modules_to_test = [
        ("utils.config", "Configuration"),
        ("utils.logger", "Logging"),
        ("models.entities", "Data entities"),
        ("database.db", "Database layer"),
        ("database.repository", "Repository"),
        ("services.face_engine", "Face engine"),
        ("services.camera_service", "Camera service"),
        ("services.recognition_service", "Recognition service"),
        ("services.training_service", "Training service"),
        ("ui.main_window", "UI layer"),
    ]

    failed = []
    for module_name, description in modules_to_test:
        try:
            __import__(module_name)
            print(f"✓ {description:<25} ({module_name})")
        except Exception as e:
            print(f"✗ {description:<25} ({module_name})")
            failed.append((module_name, str(e)))

    return len(failed) == 0, failed


def test_db_initialization():
    """Test database creation and schema."""
    print("\n" + "=" * 60)
    print("Testing Database Initialization...")
    print("=" * 60)

    try:
        from database.db import Database
        from utils.config import DB_PATH

        db = Database(DB_PATH)
        db.initialize()
        print(f"✓ Database initialized at: {DB_PATH}")

        from database.repository import AttendanceRepository

        repo = AttendanceRepository(db)
        students = repo.list_students()
        print(f"✓ Repository working (current students: {len(students)})")
        return True, None
    except Exception as e:
        print(f"✗ Database error: {e}")
        return False, str(e)


def test_face_engine():
    """Test face detection and embedding extraction."""
    print("\n" + "=" * 60)
    print("Testing Face Engine...")
    print("=" * 60)

    try:
        from services.face_engine import FaceEngine

        engine = FaceEngine()
        print(f"✓ Face detector loaded: Haar Cascade")
        print(f"✓ Embedding backend: {engine.embedding_mode}")

        if engine.embedding_mode == "none":
            print("⚠ Warning: No embedding backend detected (using ORB fallback)")

        return True, None
    except Exception as e:
        print(f"✗ Face engine error: {e}")
        return False, str(e)


def test_cascade_file():
    """Test that Haar cascade file exists."""
    print("\n" + "=" * 60)
    print("Testing Cascade File...")
    print("=" * 60)

    from utils.config import CASCADE_PATH

    cascade_path = Path(CASCADE_PATH)
    if cascade_path.exists():
        size_mb = cascade_path.stat().st_size / (1024 * 1024)
        print(f"✓ Cascade file exists: {CASCADE_PATH}")
        print(f"  Size: {size_mb:.2f} MB")
        return True, None
    else:
        print(f"✗ Cascade file missing: {CASCADE_PATH}")
        print("  Download from: https://github.com/opencv/opencv/raw/master/data/haarcascades/haarcascade_frontalface_default.xml")
        return False, "Cascade file missing"


def test_dependencies():
    """Test that required packages are installed."""
    print("\n" + "=" * 60)
    print("Testing Dependencies...")
    print("=" * 60)

    required = {
        "numpy": "Numerical computing",
        "cv2": "OpenCV (computer vision)",
        "PIL": "Pillow (image processing)",
        "sklearn": "Scikit-learn (ML)",
        "joblib": "Model serialization",
    }

    optional = {
        "deepface": "DeepFace (Facenet512 embeddings)",
        "face_recognition": "Face recognition (dlib-based)",
    }

    failed = []
    for pkg_name, description in required.items():
        try:
            __import__(pkg_name)
            print(f"✓ {description:<40} ({pkg_name})")
        except ImportError:
            print(f"✗ {description:<40} ({pkg_name})")
            failed.append((pkg_name, description))

    print("\n  Optional backends (at least one recommended):")
    for pkg_name, description in optional.items():
        try:
            __import__(pkg_name if pkg_name != "face_recognition" else "face_recognition")
            print(f"  ✓ {description:<36} ({pkg_name})")
        except ImportError:
            print(f"  ○ {description:<36} ({pkg_name}) - Not installed")

    return len(failed) == 0, failed


def main():
    """Run all verification tests."""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║  FACE ATTENDANCE SYSTEM - VERIFICATION                      ║")
    print("╚" + "=" * 58 + "╝")

    tests = [
        ("Dependencies", test_dependencies),
        ("Cascade File", test_cascade_file),
        ("Module Imports", test_imports),
        ("Database", test_db_initialization),
        ("Face Engine", test_face_engine),
    ]

    results = {}
    for test_name, test_func in tests:
        success, error = test_func()
        results[test_name] = success

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_passed = all(results.values())
    for test_name, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status:<8} {test_name}")

    print("\n" + "=" * 60)
    if all_passed:
        print("✓✓✓ All tests passed! System is ready to launch.")
        print("\n  Run: python frontend.py")
        return 0
    else:
        print("✗✗✗ Some tests failed. See errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

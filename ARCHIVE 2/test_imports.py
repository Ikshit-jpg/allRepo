#!/usr/bin/env python
"""Test if all required packages are installed."""

import sys

packages_to_test = [
    ("Flask", "flask"),
    ("Flask-CORS", "flask_cors"),
    ("librosa", "librosa"),
    ("moviepy", "moviepy"),
    ("OpenCV", "cv2"),
    ("NumPy", "numpy"),
    ("SciPy", "scipy"),
    ("scikit-image", "skimage"),
    ("scikit-learn", "sklearn"),
    ("joblib", "joblib"),
    ("resampy", "resampy"),
    ("soxr", "soxr"),
    ("ImageIO", "imageio"),
    ("imageio-ffmpeg", "imageio_ffmpeg"),
    ("PyAV", "av"),
    ("ffmpeg-python", "ffmpeg"),
]

failed = []
passed = []

for package_name, import_name in packages_to_test:
    try:
        __import__(import_name)
        passed.append(package_name)
        print(f"[OK] {package_name}")
    except ImportError as e:
        failed.append((package_name, str(e)))
        print(f"[FAIL] {package_name} - {e}")

print(f"\n{'='*60}")
print(f"PASSED: {len(passed)}/{len(packages_to_test)}")
print(f"FAILED: {len(failed)}/{len(packages_to_test)}")

if failed:
    print(f"\nMissing packages:")
    for pkg, err in failed:
        print(f"  - {pkg}")
    sys.exit(1)
else:
    print("\n[OK] All required packages are installed!")
    sys.exit(0)

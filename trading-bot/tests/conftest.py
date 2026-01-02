import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.dont_write_bytecode = True
for path in (SRC, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# Clean stale root-level pyc artifacts for removed modules (prevents coverage parse warnings).
cache_dir = ROOT / "__pycache__"
if cache_dir.exists():
    for pattern in ("config.cpython-*.pyc", "config-*.pyc"):
        for pyc in cache_dir.glob(pattern):
            pyc.unlink(missing_ok=True)

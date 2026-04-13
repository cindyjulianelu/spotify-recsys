"""
Shared pytest configuration.

Adds all src sub-packages to sys.path so that tests can import internal
modules with the same bare-name imports used in the source files.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# src/ — enables `from features.engineer import ...`
sys.path.insert(0, str(ROOT / "src"))
# src/data/ — enables `from quality import ...` (used inside cleaner.py)
sys.path.insert(0, str(ROOT / "src" / "data"))
# src/features/ — enables `from engineer import ...`
sys.path.insert(0, str(ROOT / "src" / "features"))
# src/models/ — enables `from classifier import ...`
sys.path.insert(0, str(ROOT / "src" / "models"))

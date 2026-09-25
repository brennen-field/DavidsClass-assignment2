import sys
from pathlib import Path

# Make app.py and course_assistant importable from tests.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

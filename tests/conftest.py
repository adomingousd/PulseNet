import sys
from pathlib import Path

# make world_model importable when running pytest from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

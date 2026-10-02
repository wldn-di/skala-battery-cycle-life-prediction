"""Edit data placement here or set SKALA_BATTERY_DATA_DIR."""
from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get('SKALA_BATTERY_DATA_DIR', str(BASE_DIR / 'data'))).expanduser()
SCRATCH_PATH = DATA_DIR / '30-ESSHealth-scratch.ipynb'
DAY1_OUTPUT_DIR = BASE_DIR / 'reproduced' / 'day1'
AUDIT_OUTPUT_DIR = BASE_DIR / 'reproduced' / 'audit'

"""
Shared pytest fixtures for all TraCA tests.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure the repo root is on sys.path so `import traca` works
sys.path.insert(0, str(Path(__file__).parent.parent))

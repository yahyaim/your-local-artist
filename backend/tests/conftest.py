import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from make_sample_clip import make_sample_clip  # noqa: E402


@pytest.fixture(scope="session")
def sample_clip(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return make_sample_clip(tmp_path_factory.mktemp("clips") / "sample.mp4", seconds=1.0, fps=12)

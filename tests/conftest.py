from pathlib import Path

import pytest

from hospital_intelligence.config import Settings
from hospital_intelligence.generate import generate
from hospital_intelligence.pipeline import build
from hospital_intelligence.services import AnalyticsService, precompute


@pytest.fixture(scope="session")
def dataset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("hospital")
    settings = Settings(root=root, encounters=600, seed=17)
    generate(settings)
    build(settings)
    precompute(settings)
    return root


@pytest.fixture(scope="session")
def service(dataset: Path) -> AnalyticsService:
    return AnalyticsService(dataset)

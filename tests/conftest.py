import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient
    from api.main import app

    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="session")
def features():
    import pandas as pd
    from config import DATA_FEATURES_DIR

    return pd.read_parquet(DATA_FEATURES_DIR / "training_features.parquet")

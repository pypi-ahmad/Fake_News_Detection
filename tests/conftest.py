from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest


def load_notebook_module() -> SimpleNamespace:
    notebook_path = Path(__file__).resolve().parents[1] / "Fake_news_Detection.ipynb"
    data = json.loads(notebook_path.read_text(encoding="utf-8"))

    code_cells = [cell for cell in data.get("cells", []) if cell.get("cell_type") == "code"]
    if not code_cells:
        raise RuntimeError("No code cells found in notebook")

    source = "\n".join(code_cells[0]["source"])
    namespace: dict[str, object] = {}
    exec(source, namespace)
    return SimpleNamespace(**namespace)


@pytest.fixture(scope="session")
def nb() -> SimpleNamespace:
    return load_notebook_module()


@pytest.fixture()
def sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "text": [
                "real economy growth is steady",
                "fake celebrity arrest rumor",
                "government releases official report",
                "shocking hoax spreads online",
                "scientists publish peer reviewed study",
                "fabricated election conspiracy",
                "verified weather advisory issued",
                "clickbait health cure claim",
                "local team wins championship",
                "unfounded scandal accusation",
            ],
            "label": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        }
    )


@pytest.fixture()
def sample_csv_path(tmp_path: Path, sample_df: pd.DataFrame) -> Path:
    csv_path = tmp_path / "news.csv"
    sample_df.to_csv(csv_path, index=False)
    return csv_path

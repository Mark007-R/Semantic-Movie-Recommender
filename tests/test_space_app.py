"""
Smoke tests for the Hugging Face Space UI (src/serving/space_app.py + views/).

Each page is run headless with Streamlit's AppTest against the real champion
stack, so they need the Space artifacts (catalog, cached e5-base-v2 vectors,
CF model) and skip cleanly without them.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

NEEDS = [ROOT / "data" / "9000plus.csv",
         ROOT / "results" / "emb_cache" / "intfloat__e5-base-v2.npy",
         ROOT / "models" / "cf_meta.json"]
pytestmark = pytest.mark.skipif(not all(p.exists() for p in NEEDS),
                                reason="Space artifacts (catalog, emb cache, CF model) not present")

APP = str(ROOT / "src" / "serving" / "space_app.py")


def _index(title: str, year: str) -> int:
    cat = pd.read_csv(ROOT / "data" / "9000plus.csv").fillna("")
    hit = cat.index[(cat["Title"] == title) & cat["Release_Date"].astype(str).str.startswith(year)]
    return int(hit[0])


def _run(page: str | None = None, **query) -> str:
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP, default_timeout=300)
    for k, v in query.items():
        at.query_params[k] = v
    if page:
        # st.navigation registers custom url_paths (e.g. for-you) on the first
        # run; before it, switch_page can only match a page by its filename.
        at.run()
        at.switch_page(page)
    at.run()
    assert not at.exception, at.exception
    return "\n".join(m.value for m in at.markdown)


def test_discover_landing_shows_popular_films():
    html = _run()
    assert "Describe a mood" in html
    assert "Most-rated" in html and "cs-card" in html


def test_discover_search_returns_posters():
    html = _run(q="space adventure with robots")
    assert "Top 12 for" in html
    assert html.count("class='cs-card'") == 12


def test_similar_renders_seed_and_neighbours():
    html = _run("views/similar.py", movie=str(_index("Toy Story", "1995")))
    assert "Seed film" in html and "Nearest" in html
    assert html.count("class='cs-card'") == 12


def test_for_you_explains_itemknn_recommendations():
    liked = ",".join(str(_index(t, y)) for t, y in [("Toy Story", "1995"), ("Finding Nemo", "2003")])
    html = _run("views/for_you.py", liked=liked)
    assert "Because you liked" in html
    assert html.count("class='cs-card'") == 12


def test_evaluation_reads_results():
    html = _run("views/evaluation.py")
    assert "Only collaborative filtering moves the needle" in html
    assert "0.1059" in html

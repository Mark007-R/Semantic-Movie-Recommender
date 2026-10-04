"""
CineSemantics — self-contained Streamlit demo (Hugging Face Space entrypoint).

The full Streamlit app (pages/movieflix.py) is Milvus-backed and needs the
docker-compose stack. This app serves the SAME champion components the FastAPI
service uses — e5-base-v2 embeddings + faiss HNSW + metadata rerank + ItemKNN
CF — entirely from on-disk artifacts, so it runs on a single free CPU with no
vector database. Wiring mirrors api.py's lifespan exactly (see space_kit.py);
nothing here is a second implementation.

Sets the page config and theme once, then routes to the pages in views/ via
top navigation:

    Discover        free-text semantic search with genre / rating / year filters
    More like this  item-to-item neighbours of one film
    For you         ItemKNN recommendations from a few liked films
    Evaluation      the offline numbers, read from results/

Run:  streamlit run src/serving/space_app.py
Artifacts required (all shipped with the repo / space):
  data/9000plus.csv, results/emb_cache/intfloat__e5-base-v2.npy,
  models/cf_interactions.npz + cf_meta.json, results/phase6_metrics.json,
  results/phase2a_embeddings.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.serving import space_kit as kit  # noqa: E402
from src.serving import ui_theme          # noqa: E402

st.set_page_config(
    page_title="CineSemantics · Semantic movie recommender",
    page_icon=str(ROOT / "assets" / "logo-mark.svg"),
    layout="wide",
    initial_sidebar_state="collapsed",
)
ui_theme.apply_theme()    # shared mark.dev paper tokens, type and widgets
kit.apply_components()    # CineSemantics cards, grid, tables, top-nav shell

st.logo(str(ROOT / "assets" / "logo.svg"), size="large")

pages = [
    st.Page("views/discover.py", title="Discover", icon=":material/search:", default=True),
    st.Page("views/similar.py", title="More like this", icon=":material/movie:", url_path="similar"),
    st.Page("views/for_you.py", title="For you", icon=":material/favorite:", url_path="for-you"),
    st.Page("views/evaluation.py", title="Evaluation", icon=":material/insights:", url_path="evaluation"),
]

st.navigation(pages, position="top").run()
kit.footer()

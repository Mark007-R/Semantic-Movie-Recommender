"""
CineSemantics — self-contained Streamlit demo (Hugging Face Space entrypoint).

The full Streamlit app (pages/movieflix.py) is Milvus-backed and needs the
docker-compose stack. This app serves the SAME champion components the FastAPI
service uses — e5-base-v2 embeddings + faiss HNSW + metadata rerank + ItemKNN
CF — entirely from on-disk artifacts, so it runs on a single free CPU with no
vector database. Wiring mirrors api.py's lifespan exactly; nothing here is a
second implementation.

Run:  streamlit run src/serving/space_app.py
Artifacts required (all shipped with the repo / space):
  data/9000plus.csv, results/emb_cache/intfloat__e5-base-v2.npy,
  models/cf_interactions.npz + cf_meta.json (+ data/eval/cf_split.json fallback)
"""
from __future__ import annotations

import html
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.retrieval.embedder import ChampionEmbedder          # noqa: E402
from src.retrieval.index import MovieIndex                    # noqa: E402
from src.rerank.metadata_rerank import MetadataReranker       # noqa: E402
from src.recsys.recommender import ItemKNNRecommender         # noqa: E402
from src.serving.offline_metrics import panel as offline_panel  # noqa: E402
from src.serving import ui_theme                              # noqa: E402

st.set_page_config(page_title="CineSemantics", page_icon="🎬", layout="wide")
ui_theme.apply_theme()


def _results_css() -> str:
    """Result-row cards and the metrics panel, written against ui_theme's tokens."""
    mono, display = ui_theme.FONT_MONO, ui_theme.FONT_DISPLAY
    return f"""
<style>
.stApp h1 em {{ font-style: italic; font-weight: 600; color: var(--accent); }}

.cs-row {{
  display: flex; align-items: center; gap: 1rem;
  background: var(--card); border: 1px solid var(--line); border-radius: 16px;
  padding: .7rem 1.1rem .7rem .7rem; box-shadow: var(--shadow-sm);
  transition: border-color .25s ease, box-shadow .25s ease, transform .25s ease;
}}
.cs-row:hover {{ border-color: var(--line-strong); box-shadow: var(--shadow-md); transform: translateY(-1px); }}
.cs-poster {{
  flex: 0 0 64px; width: 64px; height: 96px; object-fit: cover;
  border-radius: 10px; border: 1px solid var(--line); background: var(--paper-alt);
}}
.cs-body {{ flex: 1 1 auto; min-width: 0; }}
.cs-title {{ font-family: {display}; font-weight: 600; font-size: 1.1rem; line-height: 1.3; color: var(--ink); overflow-wrap: anywhere; }}
.cs-year {{ font-family: {mono}; font-size: .72rem; font-weight: 500; letter-spacing: .04em; color: var(--ink-3); }}
.cs-genre {{
  margin-top: .35rem; font-family: {mono}; font-size: .66rem; font-weight: 500;
  text-transform: uppercase; letter-spacing: .12em; color: var(--ink-3);
}}
.cs-score {{ flex: 0 0 auto; display: flex; flex-direction: column; align-items: flex-end; gap: .35rem; }}
.cs-score-value {{
  font-family: {mono}; font-size: .8rem; font-weight: 600; color: var(--accent);
  background: var(--accent-tint); border: 1px solid var(--accent-line); border-radius: 999px; padding: .2rem .7rem;
}}
.cs-method {{ font-family: {mono}; font-size: .6rem; font-weight: 500; text-transform: uppercase; letter-spacing: .12em; color: var(--ink-3); }}

[data-testid="stJson"] {{
  background: var(--card); border: 1px solid var(--line); border-radius: 16px;
  padding: .85rem 1.1rem; box-shadow: var(--shadow-sm);
}}

@media (max-width: 480px) {{
  .cs-row {{ gap: .7rem; padding-right: .8rem; }}
  .cs-poster {{ flex-basis: 48px; width: 48px; height: 72px; }}
}}
</style>
"""


st.markdown(_results_css(), unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading catalog, embeddings and index…")
def load_stack():
    """Same wiring as api.py's lifespan — catalog → embedder → HNSW → CF."""
    catalog = pd.read_csv(ROOT / "data" / "9000plus.csv").fillna("")
    embedder = ChampionEmbedder()
    emb = embedder.encode_catalog(catalog)          # served from the .npy cache
    index = MovieIndex(catalog, emb, embedder)
    index.build_faiss()
    reranker = MetadataReranker(catalog)
    try:
        rec = ItemKNNRecommender.load(ROOT / "models", catalog_embeddings=emb)
    except Exception:                                # noqa: BLE001
        rec = None
    genres = sorted({g.strip() for gs in catalog["Genre"].astype(str)
                     for g in gs.split(",") if g.strip()})
    return catalog, index, reranker, rec, genres


def movie_row(meta: dict | pd.Series, score: float, method: str | None = None):
    """One result row, as a card: poster thumbnail + title/genre/year + score."""
    url = str(meta.get("Poster_Url", "") or "")
    year = str(meta.get("Release_Date", ""))[:4]
    poster = (f"<img class='cs-poster' src='{html.escape(url, quote=True)}' alt=''>"
              if url.startswith("http") else "<div class='cs-poster'></div>")
    method_html = f"<span class='cs-method'>{html.escape(str(method))}</span>" if method else ""
    st.markdown(
        "<div class='cs-row'>"
        f"{poster}"
        "<div class='cs-body'>"
        f"<div class='cs-title'>{html.escape(str(meta.get('Title', '')))} "
        f"<span class='cs-year'>({html.escape(year)})</span></div>"
        f"<div class='cs-genre'>{html.escape(str(meta.get('Genre', '')))}</div>"
        "</div>"
        "<div class='cs-score'>"
        f"<span class='cs-score-value'>{score:.3f}</span>{method_html}"
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )


def main() -> None:
    catalog, index, reranker, rec, genres = load_stack()

    st.title("🎬 Cine*Semantics*")
    st.caption(
        f"Semantic search + personalized recommendation over {len(catalog):,} TMDB movies — "
        "e5-base-v2 embeddings, faiss HNSW, metadata rerank, ItemKNN collaborative filtering. "
        "No Milvus required: this Space serves the exact champion stack the offline "
        "evaluation measured. Source: "
        "[Mark007-R/Semantic-Movie-Recommender](https://github.com/Mark007-R/Semantic-Movie-Recommender)"
    )

    tab_search, tab_similar, tab_foryou, tab_metrics = st.tabs(
        ["🔍 Semantic search", "🎞️ More like this", "🤝 For you", "📊 Honest metrics"])

    # ---------------------------------------------------------------- search
    with tab_search:
        q = st.text_input("Describe what you feel like watching",
                          placeholder="a heist that goes sideways, dark humour")
        f1, f2, f3, f4 = st.columns([3, 2, 2, 1])
        sel_genres = f1.multiselect("Genres (proper filter — not a substring match)", genres)
        min_rating = f2.slider("Min rating", 0.0, 10.0, 0.0, 0.5)
        yr = f3.slider("Year range", 1900, 2026, (1900, 2026))
        rerank = f4.toggle("Rerank", value=True,
                           help="Day-4 metadata reranker: cosine + popularity prior + genre Jaccard")
        if q.strip():
            hits = index.search(q, top_k=10, genres=sel_genres or None,
                                min_rating=min_rating or None,
                                min_year=yr[0] if yr[0] > 1900 else None,
                                max_year=yr[1] if yr[1] < 2026 else None,
                                overfetch=30 if rerank else 20)
            if rerank and hits:
                ordered = reranker.rerank_query([h["index"] for h in hits],
                                                [h["score"] for h in hits],
                                                query_genres=sel_genres or None)
                by_idx = {h["index"]: h for h in hits}
                hits = [by_idx[i] for i in ordered]
            if not hits:
                st.info("No catalog match under those filters — relax them and retry.")
            for h in hits[:10]:
                movie_row(catalog.iloc[h["index"]], h["score"])

    # --------------------------------------------------------------- similar
    with tab_similar:
        title = st.selectbox("Pick a movie", catalog["Title"].astype(str).tolist(),
                             index=None, placeholder="Start typing a title…")
        if title:
            qi = index.title_to_index(title)
            if qi is None:
                st.warning("Title not found in the index.")
            else:
                for h in index.similar(qi, top_k=10):
                    movie_row(catalog.iloc[h["index"]], h["score"])

    # ---------------------------------------------------------------- for you
    with tab_foryou:
        st.markdown("Personalized ranking from the **ItemKNN CF champion** "
                    "(Day-3 winner on the held-out MovieLens split), with a "
                    "content cold-start fallback for titles without interactions.")
        liked = st.multiselect("Movies you liked", catalog["Title"].astype(str).tolist())
        if liked and rec is None:
            st.error("CF artifacts not loaded — models/cf_interactions.npz missing.")
        elif liked:
            liked_idx = [i for t in liked if (i := index.title_to_index(t)) is not None]
            for r in rec.recommend(liked_idx, top_k=10):
                movie_row(catalog.iloc[r["index"]], r["score"], r.get("method"))

    # ---------------------------------------------------------------- metrics
    with tab_metrics:
        st.markdown(
            "This project's Day-1 audit finding was that a *'recommendation engine'* "
            "shipped with **zero evaluation**. This panel is the fix, live: the same "
            "offline numbers the API serves at `/metrics`, sourced from the persisted "
            "`results/` artifacts — including the honest ones."
        )
        st.json(offline_panel())


if __name__ == "__main__":
    main()

"""Discover — free-text semantic search with metadata filters (mirrors api.py /search)."""
import time

import streamlit as st

from src.serving import space_kit as kit

kit.page_header(
    f"Semantic search · {kit.catalog_size():,} films",
    "Describe a mood. <em>Find the film.</em>",
    "Your words are embedded with <b>e5-base-v2</b>, matched against every film's title, genres "
    "and overview in a <b>faiss HNSW</b> index, then reranked with genre overlap and a popularity prior.",
    hero=True,
)

s = kit.stack()
lo, hi = s.years

# URL first (shareable searches), then whatever the session already holds.
if "q" not in st.session_state:
    st.session_state.q = st.query_params.get("q", "")
st.session_state.setdefault("f_genres", [])
st.session_state.setdefault("f_rating", 0.0)
st.session_state.setdefault("f_years", (lo, hi))
st.session_state.setdefault("f_rerank", True)


def _take_prompt() -> None:
    if st.session_state.prompt:
        st.session_state.q = st.session_state.prompt
    st.session_state.prompt = None


f = st.session_state
active = (bool(f.f_genres) + (f.f_rating > 0) + (tuple(f.f_years) != (lo, hi)) + (not f.f_rerank))

bar, filt = st.columns([6, 1], vertical_alignment="center")
with bar:
    st.text_input("What do you feel like watching?", key="q", label_visibility="collapsed",
                  placeholder="a heist that goes sideways, with dark humour",
                  icon=":material/search:", persist_state="session")
with filt.popover(f"Filters · {active}" if active else "Filters", icon=":material/tune:",
                  width="stretch", key="filters"):
    st.multiselect("Genres", s.genres, key="f_genres", placeholder="Any genre",
                   persist_state="session")
    st.slider("Minimum rating", 0.0, 10.0, step=0.5, key="f_rating", persist_state="session")
    st.slider("Release years", lo, hi, key="f_years", persist_state="session")
    st.toggle("Metadata rerank", key="f_rerank", persist_state="session",
              help="Cosine + genre overlap with the chosen genres + a popularity prior. "
                   "The Day-4 champion: +42% NDCG@10 over raw cosine at ~49 ms p95.")

st.pills("Or start from a prompt", kit.QUICK_PROMPTS, key="prompt", on_change=_take_prompt)

q = (f.q or "").strip()
if q:
    st.query_params["q"] = q
else:
    st.query_params.pop("q", None)

if not q:
    kit.result_head("Most-rated <em>in the catalog</em>", ["click any poster for more like it"])
    kit.movie_grid(s, [{"index": i} for i in s.order[:12]], ranked=False)
    st.stop()

genres = f.f_genres or None
t0 = time.perf_counter()
hits = s.index.search(q, top_k=12, genres=genres,
                      min_rating=f.f_rating or None,
                      min_year=f.f_years[0] if f.f_years[0] > lo else None,
                      max_year=f.f_years[1] if f.f_years[1] < hi else None,
                      overfetch=30 if f.f_rerank else 20)
if f.f_rerank and hits:
    ordered = s.reranker.rerank_query([h["index"] for h in hits], [h["score"] for h in hits],
                                      query_genres=genres)
    by_idx = {h["index"]: h for h in hits}
    hits = [by_idx[i] for i in ordered]
ms = (time.perf_counter() - t0) * 1000

if not hits:
    kit.result_head(f"No matches for <em>“{kit.esc(q)}”</em>")
    kit.note("<b>Nothing passes those filters.</b> Loosen the genre, rating or year filters and try again.")
    st.stop()

order = "reranked" if f.f_rerank else "cosine order"
filters = f"{active} filter{'s' if active != 1 else ''}" if active else "no filters"
kit.result_head(f"Top {len(hits)} for <em>“{kit.esc(q)}”</em>",
                [order, filters, f"{ms:.0f} ms"], accent="reranked")
kit.movie_grid(s, hits, score_label="cosine similarity to your query")

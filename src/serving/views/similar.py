"""More like this — item-to-item neighbours in e5-base-v2 space (mirrors api.py /similar)."""
import time

import streamlit as st

from src.serving import space_kit as kit

kit.page_header(
    "Item-to-item",
    "More <em>like this</em>",
    "Pick a film and get its nearest neighbours in <b>e5-base-v2</b> embedding space: cosine over "
    "title, genres, year and overview, served from the same <b>faiss HNSW</b> index as the API.",
)

s = kit.stack()

if "movie" not in st.session_state:  # poster links arrive as ?movie=<catalog index>
    picked = kit.qp_ints("movie", len(s.catalog))
    st.session_state.movie = picked[0] if picked else None

movie = st.selectbox("Seed film", s.order, index=None, key="movie",
                     format_func=s.labels.__getitem__, placeholder="Start typing a title…",
                     persist_state="session")

if movie is None:
    st.query_params.pop("movie", None)
    kit.result_head("Or start from <em>a well-known film</em>", ["click any poster"])
    kit.movie_grid(s, [{"index": i} for i in s.order[:12]], ranked=False)
    st.stop()

st.query_params["movie"] = str(movie)
kit.seed_card(s, movie)

t0 = time.perf_counter()
hits = s.index.similar(movie, top_k=12)
ms = (time.perf_counter() - t0) * 1000

kit.result_head("Nearest <em>neighbours</em>", ["cosine · e5-base-v2", "faiss HNSW", f"{ms:.1f} ms"],
                accent="cosine · e5-base-v2")
kit.movie_grid(s, hits, score_label="cosine similarity to the seed film")

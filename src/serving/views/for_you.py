"""For you — ItemKNN recommendations from a few liked films (mirrors api.py /recommend)."""
import time

import streamlit as st

from src.serving import space_kit as kit

kit.page_header(
    "Collaborative filtering",
    "Recommended <em>for you</em>",
    "Pick a few films you liked. <b>ItemKNN</b>, the ranker that won the held-out evaluation, scores "
    "every candidate by how strongly MovieLens users who liked your films also liked it. Films with "
    "no MovieLens history fall back to content similarity.",
)

s = kit.stack()

if s.rec is None:
    kit.note("<b>CF artifacts not loaded.</b> Recommendations need <code>models/cf_interactions.npz</code> "
             "and <code>models/cf_meta.json</code>.")
    st.stop()

if "liked" not in st.session_state:  # shareable as ?liked=12,34
    st.session_state.liked = kit.qp_ints("liked", len(s.catalog))


def _take_taste() -> None:
    if st.session_state.taste:
        st.session_state.liked = [i for t, y in kit.TASTES[st.session_state.taste]
                                  if (i := kit.lookup(s, t, y)) is not None]
    st.session_state.taste = None


st.pills("Start from a taste", list(kit.TASTES), key="taste", on_change=_take_taste)
liked = st.multiselect("Films you liked", s.order, key="liked", format_func=s.labels.__getitem__,
                       placeholder="Search the catalog…", max_selections=20, persist_state="session")
diverse = st.toggle("Diversify genres (MMR, λ = 0.7)", key="diverse", persist_state="session",
                    help="Maximal Marginal Relevance over the top 60 candidates, genre overlap as the "
                         "redundancy term. On the held-out split: −0.3 pp NDCG@10 for +6.4 pp "
                         "intra-list diversity.")

if liked:
    st.query_params["liked"] = ",".join(str(i) for i in liked)
else:
    st.query_params.pop("liked", None)
    n_users, n_items = s.rec.R.shape
    kit.section("How it works", "Three steps, no training at request time")
    kit.html_block(
        "<div class='cs-steps'>"
        "<div class='cs-step'><div class='cs-step-n'>01</div><div class='cs-step-t'>Your picks</div>"
        f"<div class='cs-step-d'>Each film you pick is a column in the MovieLens interaction matrix: "
        f"{n_users:,} users × {n_items:,} films, aligned to this catalog by title and year.</div></div>"
        "<div class='cs-step'><div class='cs-step-n'>02</div><div class='cs-step-t'>Item-item cosine</div>"
        "<div class='cs-step-d'>Every candidate is scored by the summed cosine similarity of its column "
        "to your picks: films liked by the same people score high.</div></div>"
        "<div class='cs-step'><div class='cs-step-n'>03</div><div class='cs-step-t'>Rank</div>"
        "<div class='cs-step-d'>The top 12 are shown, excluding your picks. MMR can trade a little "
        "relevance for genre variety.</div></div></div>"
    )
    st.stop()

t0 = time.perf_counter()
recs = s.rec.recommend(liked, top_k=12, diversity=0.7 if diverse else None)
ms = (time.perf_counter() - t0) * 1000

in_cf = [i for i in liked if i in s.cf_items]
method = recs[0]["method"] if recs else "itemknn"
cold = method == "content_coldstart"
items = []
for r in recs:
    why = None if cold else s.rec.explain(in_cf, r["index"])
    note = (f"Because you liked <b>{kit.esc(s.catalog.iloc[why]['Title'])}</b>" if why is not None
            else "Close to your picks in embedding space" if cold else None)
    items.append({**r, "note": note})

label = {"itemknn": "ItemKNN", "itemknn+mmr": "ItemKNN + MMR",
         "content_coldstart": "content cold-start"}.get(method, method)
kit.result_head("Your <em>top 12</em>",
                [label, f"{len(in_cf)} of {len(liked)} picks in CF", f"{ms:.0f} ms"], accent=label)
if cold:
    kit.note("<b>Cold start.</b> None of your picks has MovieLens history, so there is nothing for "
             "collaborative filtering to work with. These come from the centroid of your picks' "
             "e5-base-v2 embeddings instead.")
elif len(in_cf) < len(liked):
    missing = [kit.esc(s.catalog.iloc[i]["Title"]) for i in liked if i not in s.cf_items]
    kit.note(f"<b>Partial coverage.</b> {', '.join(missing)} {'has' if len(missing) == 1 else 'have'} "
             "no MovieLens interactions, so only your other picks drive these scores.")
kit.movie_grid(s, items, score_label=("cosine similarity to your picks' centroid" if cold
                                      else "ItemKNN score: summed item-item cosine"))

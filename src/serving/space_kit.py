"""Shared building blocks for the CineSemantics Space pages (src/serving/views/).

The cached champion stack (the same wiring as api.py's lifespan: catalog ->
e5-base-v2 -> faiss HNSW -> metadata rerank -> ItemKNN), catalog helpers, the
readers for the persisted results/, and the small HTML components that
components_css() styles on top of ui_theme's tokens. Every page imports from
here so numbers and markup stay consistent.
"""
from __future__ import annotations

import html
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.serving import ui_theme  # noqa: E402

RESULTS = ROOT / "results"
REPO_URL = "https://github.com/Mark007-R/Semantic-Movie-Recommender"

# Starting points for the Discover page.
QUICK_PROMPTS = [
    "space adventure with robots",
    "a heist that goes sideways",
    "feel-good animated family film",
    "slow-burn psychological thriller",
    "coming-of-age summer romance",
    "dystopian future rebellion",
]

# One-click tastes for the For-you page, as (title, year) so they survive a
# catalog re-order. Every title here has MovieLens interactions (CF universe).
TASTES = {
    "Pixar night": [("Toy Story", "1995"), ("Finding Nemo", "2003"), ("Up", "2009"),
                    ("Monsters, Inc.", "2001")],
    "Mind-benders": [("Inception", "2010"), ("The Matrix", "1999"), ("Memento", "2000"),
                     ("The Prestige", "2006")],
    "Crime classics": [("The Godfather", "1972"), ("Pulp Fiction", "1994"),
                       ("Reservoir Dogs", "1992"), ("Heat", "1995")],
    "Date night": [("The Notebook", "2004"), ("Pride & Prejudice", "2005"),
                   ("La La Land", "2016"), ("Notting Hill", "1999")],
    "Big adventure": [("The Lord of the Rings: The Fellowship of the Ring", "2001"),
                      ("Jurassic Park", "1993"), ("Back to the Future", "1985"),
                      ("Mad Max: Fury Road", "2015")],
}


# ---------------------------------------------------------------------------
# The champion stack
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def _load_stack() -> SimpleNamespace:
    from src.recsys.recommender import ItemKNNRecommender
    from src.rerank.metadata_rerank import MetadataReranker
    from src.retrieval.embedder import ChampionEmbedder
    from src.retrieval.index import MovieIndex

    catalog = pd.read_csv(ROOT / "data" / "9000plus.csv").fillna("")
    embedder = ChampionEmbedder()
    emb = embedder.encode_catalog(catalog)          # served from the .npy cache
    index = MovieIndex(catalog, emb, embedder)
    index.build_faiss()
    reranker = MetadataReranker(catalog)
    try:
        rec = ItemKNNRecommender.load(ROOT / "models", catalog_embeddings=emb)
        rec.attach_genres(catalog)                  # enables the MMR diversity option
    except Exception:                               # noqa: BLE001
        rec = None
    embedder.encode_query("warm up")                # load the weights now, not on the first search

    years = catalog["Release_Date"].astype(str).str[:4]
    labels = [f"{t} ({y})" if y.isdigit() else str(t)
              for t, y in zip(catalog["Title"].astype(str), years)]
    votes = pd.to_numeric(catalog["Vote_Count"], errors="coerce").fillna(0)
    numeric_years = pd.to_numeric(years, errors="coerce").dropna().astype(int)
    by_title_year = {(str(t).strip().lower(), y): i
                     for i, (t, y) in enumerate(zip(catalog["Title"], years))}
    return SimpleNamespace(
        catalog=catalog, index=index, reranker=reranker, rec=rec,
        genres=sorted({g.strip() for gs in catalog["Genre"].astype(str)
                       for g in gs.split(",") if g.strip()}),
        labels=labels,
        order=[int(i) for i in votes.sort_values(ascending=False, kind="stable").index],
        cf_items=set(rec.item_universe) if rec is not None else set(),
        years=(int(numeric_years.min()), int(numeric_years.max())),
        by_title_year=by_title_year,
    )


def stack() -> SimpleNamespace:
    """The loaded stack; the first call of a fresh container shows a loader."""
    with st.spinner("Loading the catalog, embeddings and HNSW index…"):
        return _load_stack()


@st.cache_data(show_spinner=False)
def catalog_size() -> int:
    """Row count for page headers, cheap enough to show before the stack loads."""
    return len(pd.read_csv(ROOT / "data" / "9000plus.csv", usecols=["Title"]))


def lookup(s: SimpleNamespace, title: str, year: str) -> int | None:
    return s.by_title_year.get((title.strip().lower(), year))


def qp_ints(name: str, n: int) -> list[int]:
    """Catalog indices from a comma-separated query parameter (invalid ones dropped)."""
    raw = st.query_params.get(name, "")
    out = []
    for part in str(raw).split(","):
        part = part.strip()
        if part.isdigit() and int(part) < n and int(part) not in out:
            out.append(int(part))
    return out


# ---------------------------------------------------------------------------
# Results artifacts
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def read_json(name: str) -> dict:
    try:
        return json.loads((RESULTS / name).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


@st.cache_data(show_spinner=False)
def read_csv(name: str) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(RESULTS / name)
    except Exception:                               # noqa: BLE001
        return None
    return None if df.empty else df


# ---------------------------------------------------------------------------
# HTML components
# ---------------------------------------------------------------------------


def esc(value) -> str:
    """Escape for HTML inside st.markdown ($ would otherwise start KaTeX)."""
    return html.escape(" ".join(str(value).split()), quote=True).replace("$", "&#36;")


def poster_url(url, size: str = "w342") -> str:
    """TMDB serves every poster at several widths; the catalog stores the original."""
    url = str(url or "")
    if not url.startswith("http"):
        return ""
    return url.replace("/t/p/original/", f"/t/p/{size}/")


def _year(row) -> str:
    y = str(row.get("Release_Date", ""))[:4]
    return y if y.isdigit() else ""


def _rating(row) -> float | None:
    try:
        v = float(row.get("Vote_Average"))
    except (TypeError, ValueError):
        return None
    return v if v > 0 else None


def _genres(row) -> list[str]:
    return [g.strip() for g in str(row.get("Genre", "")).split(",") if g.strip()]


def html_block(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


def page_header(eyebrow: str, title_html: str, lede: str, *, hero: bool = False) -> None:
    cls = "hero-title" if hero else "ph-title"
    html_block(
        f"<div class='ph'><span class='eyebrow'><span class='pulse'></span>{eyebrow}</span>"
        f"<h1 class='{cls}'>{title_html}</h1><p class='ph-lede'>{lede}</p></div>"
    )


def section(index: str, title: str, sub: str = "") -> None:
    sub_html = f"<p class='sec-sub'>{sub}</p>" if sub else ""
    html_block(f"<div class='sec'><div class='sec-index'>{index}</div>"
               f"<h2 class='sec-title'>{title}</h2>{sub_html}</div>")


def result_head(title_html: str, badges: list[str] = (), *, accent: str | None = None) -> None:
    """A results title with small mono badges; the `accent` badge is highlighted."""
    chips = "".join(f"<span class='is-accent'>{b}</span>" if b == accent else f"<span>{b}</span>"
                    for b in badges)
    html_block(f"<div class='cs-resulthead'><h3>{title_html}</h3>"
               f"<div class='cs-badges'>{chips}</div></div>")


def movie_grid(s: SimpleNamespace, items: list[dict], *, score_label: str | None = None,
               ranked: bool = True) -> None:
    """Poster cards. items: {"index", optional "score", optional "note" (HTML)}.

    Each card links to the More-like-this page for that film."""
    cards = []
    for rank, item in enumerate(items, 1):
        row = s.catalog.iloc[item["index"]]
        title = esc(row.get("Title", ""))
        src = poster_url(row.get("Poster_Url"))
        art = (f"<img src='{esc(src)}' alt='' loading='lazy'>" if src
               else f"<div class='cs-noposter'>{title}</div>")
        chip = ""
        if score_label and item.get("score") is not None:
            chip = (f"<span class='cs-chip' title='{esc(score_label)}'>"
                    f"{float(item['score']):.3f}</span>")
        rank_html = f"<span class='cs-rank'>{rank:02d}</span>" if ranked else ""
        meta = [y for y in [_year(row)] if y]
        rating = _rating(row)
        if rating is not None:
            meta.append(f"<span class='star'>★</span> {rating:.1f}")
        tags = " · ".join(esc(g) for g in _genres(row)[:3])
        note = f"<div class='cs-why'>{item['note']}</div>" if item.get("note") else ""
        cards.append(
            f"<a class='cs-card' href='similar?movie={int(item['index'])}' target='_self' "
            f"aria-label='More like {title}'>"
            f"<div class='cs-poster'>{art}{rank_html}{chip}"
            f"<span class='cs-more'>More like this →</span></div>"
            f"<div class='cs-name'>{title}</div>"
            f"<div class='cs-meta'>{' · '.join(meta)}</div>"
            f"<div class='cs-tags'>{tags}</div>{note}</a>"
        )
    html_block(f"<div class='cs-grid'>{''.join(cards)}</div>")


def seed_card(s: SimpleNamespace, idx: int) -> None:
    """The chosen film, large: poster, facts, genres and overview."""
    row = s.catalog.iloc[idx]
    src = poster_url(row.get("Poster_Url"), "w500")
    art = (f"<img src='{esc(src)}' alt=''>" if src
           else f"<div class='cs-noposter'>{esc(row.get('Title', ''))}</div>")
    facts = []
    rating = _rating(row)
    if rating is not None:
        facts.append(f"<span class='cs-fact is-accent'>★ {rating:.1f} / 10</span>")
    try:
        facts.append(f"<span class='cs-fact'>{int(float(row.get('Vote_Count'))):,} votes</span>")
    except (TypeError, ValueError):
        pass
    if str(row.get("Original_Language", "")).strip():
        facts.append(f"<span class='cs-fact'>{esc(str(row.get('Original_Language')).upper())}</span>")
    in_cf = idx in s.cf_items
    facts.append(f"<span class='cs-fact'>{'In the CF universe' if in_cf else 'Content-only'}</span>")
    genres = "".join(f"<span class='cs-genre'>{esc(g)}</span>" for g in _genres(row))
    year = _year(row)
    html_block(
        f"<div class='cs-seed'><div class='cs-seed-art'>{art}</div><div class='cs-seed-body'>"
        f"<span class='eyebrow'>Seed film</span>"
        f"<div class='cs-seed-title'>{esc(row.get('Title', ''))}"
        f"{f' <span>({year})</span>' if year else ''}</div>"
        f"<div class='cs-facts'>{''.join(facts)}</div>"
        f"<div class='cs-genres'>{genres}</div>"
        f"<p class='cs-overview'>{esc(row.get('Overview', ''))}</p>"
        f"<a class='cs-link' href='for-you?liked={idx}' target='_self'>Recommend from this film →</a>"
        f"</div></div>"
    )


def stat_tiles(tiles: list[tuple[str, str, str]]) -> None:
    """(value, label, note) tiles in an auto-fitting row."""
    cells = "".join(
        f"<div class='cs-stat'><div class='cs-stat-value'>{v}</div>"
        f"<div class='cs-stat-label'>{label}</div><div class='cs-stat-note'>{note}</div></div>"
        for v, label, note in tiles
    )
    html_block(f"<div class='cs-stats'>{cells}</div>")


def note(text_html: str) -> None:
    html_block(f"<div class='cs-note'>{text_html}</div>")


def footer() -> None:
    html_block(
        "<div class='cs-footer'><span>CineSemantics · e5-base-v2 · faiss HNSW · ItemKNN</span>"
        f"<span><a href='{REPO_URL}' target='_blank' rel='noopener'>Source &amp; evaluation harness ↗</a>"
        "</span></div>"
    )


# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------


def components_css() -> str:
    """CineSemantics components on the shared tokens (ui_theme.theme_css)."""
    mono, display = ui_theme.FONT_MONO, ui_theme.FONT_DISPLAY
    neutral = ui_theme._palette()["neutrals"][3]
    return f"""
<style>
:root {{ --display: {display}; --mono: {mono}; --neutral: {neutral}; }}

/* ---- shell: top navigation + page width ---------------------------------- */
[data-testid="stMainBlockContainer"] {{ max-width: 1240px; padding-top: 5.25rem; padding-bottom: 2.5rem; }}
[data-testid="stHeader"] {{ border-bottom: 1px solid var(--line); }}
[data-testid="stHeaderLogo"], [data-testid="stLogo"] {{ height: 2rem; max-width: 13rem; }}
[data-testid="stTopNavLink"] {{ border-radius: 999px; padding-left: .8rem; padding-right: .8rem; transition: background .2s ease; }}
[data-testid="stTopNavLink"]:hover {{ background: var(--accent-tint); }}
[data-testid="stTopNavLink"]:hover span {{ color: var(--accent) !important; }}
[data-testid="stTopNavLink"][aria-current="page"] {{ background: var(--accent-tint); box-shadow: inset 0 0 0 1px var(--accent-line); }}
[data-testid="stTopNavLink"][aria-current="page"] span {{ color: var(--accent) !important; font-weight: 600; }}

/* ---- headers --------------------------------------------------------------- */
.eyebrow {{
  display: inline-flex; align-items: center; gap: .55rem;
  font-family: var(--mono); font-size: .68rem; font-weight: 600; letter-spacing: .14em;
  text-transform: uppercase; color: var(--accent);
}}
.pulse {{ width: 8px; height: 8px; border-radius: 50%; background: var(--accent); position: relative; flex: none; }}
.pulse::after {{
  content: ''; position: absolute; inset: -4px; border-radius: 50%; border: 1px solid var(--accent);
  animation: cs-pulse 2.4s ease-out infinite;
}}
@keyframes cs-pulse {{ 0% {{ transform: scale(.6); opacity: .9; }} 100% {{ transform: scale(2.2); opacity: 0; }} }}
@media (prefers-reduced-motion: reduce) {{ .pulse::after {{ animation: none; display: none; }} }}
.ph {{ margin: .25rem 0 1.5rem; }}
.stApp .hero-title, .stApp .ph-title {{
  font-family: var(--display); font-weight: 600; letter-spacing: -.035em; color: var(--ink);
  margin: .9rem 0 .8rem; padding: 0;
}}
.stApp .hero-title {{ font-size: clamp(2.5rem, 5.6vw, 4.4rem); line-height: 1.02; max-width: 15ch; }}
.stApp .ph-title {{ font-size: clamp(2.1rem, 4.2vw, 3.1rem); line-height: 1.05; }}
.stApp .hero-title em, .stApp .ph-title em {{ font-weight: 500; }}
.stApp .ph-lede {{ color: var(--ink-2); font-size: 1.04rem; line-height: 1.7; max-width: 44rem; margin: 0; }}
.stApp .ph-lede b {{ color: var(--ink); font-weight: 600; }}

.sec {{ margin: 3rem 0 1.1rem; }}
.sec-index {{
  display: flex; align-items: center; gap: .8rem; font-family: var(--mono); font-size: .68rem; font-weight: 600;
  letter-spacing: .14em; text-transform: uppercase; color: var(--accent);
}}
.sec-index::after {{ content: ''; flex: 1; height: 1px; background: linear-gradient(90deg, var(--line-strong), transparent); }}
.stApp .sec-title {{
  font-family: var(--display); font-weight: 600; font-size: clamp(1.45rem, 2.6vw, 1.9rem);
  letter-spacing: -.02em; color: var(--ink); margin: .55rem 0 .35rem; padding: 0;
}}
.stApp .sec-sub {{ color: var(--ink-2); font-size: .95rem; line-height: 1.65; max-width: 52rem; margin: 0; }}

/* ---- search bar, filters, prompt pills -------------------------------------- */
/* Streamlit 1.6x text inputs are react-aria: the frame is stTextInputRootElement, not baseweb */
.stApp [data-testid="stTextInputRootElement"] {{ background: var(--card); border: 1px solid var(--line); border-radius: 10px; }}
.stApp [data-testid="stTextInputRootElement"]:focus-within {{ border-color: var(--accent); box-shadow: 0 0 0 3px var(--accent-glow); }}
.stApp .st-key-q [data-testid="stTextInputRootElement"] {{
  height: 3.3rem; border-radius: 999px; border-color: var(--line-strong); box-shadow: var(--shadow-md); padding: 0 .75rem 0 1.15rem;
}}
.stApp .st-key-q [data-testid="stTextInputRootElement"]:focus-within {{ border-color: var(--accent); box-shadow: 0 0 0 4px var(--accent-glow), var(--shadow-md); }}
.stApp .st-key-q input {{ font-size: 1.06rem; }}
.stApp .st-key-q svg, .stApp .st-key-q [data-testid="stIconMaterial"] {{ color: var(--accent); }}
/* ...and so are select boxes: the frame is the role=group child, tags carry data-tag */
.stApp [data-testid="stSelectbox"] [role="group"], .stApp [data-testid="stMultiSelect"] [role="group"]:has([data-testid="stMultiSelectTagsContainer"]) {{
  background: var(--card); border: 1px solid var(--line-strong); border-radius: 14px; min-height: 2.9rem; box-shadow: var(--shadow-sm);
}}
.stApp [data-testid="stSelectbox"] [role="group"]:focus-within, .stApp [data-testid="stMultiSelect"] [role="group"]:has([data-testid="stMultiSelectTagsContainer"]):focus-within {{
  border-color: var(--accent); box-shadow: 0 0 0 3px var(--accent-glow);
}}
.stApp [data-testid="stMultiSelect"] [data-tag] {{
  background: var(--accent-tint); color: var(--accent); border: 1px solid var(--accent-line); border-radius: 999px; font-weight: 500;
}}
.stApp [data-testid="stMultiSelect"] [data-tag] button {{ color: var(--accent); }}
.stApp .st-key-filters button {{
  border-radius: 999px; min-height: 3.2rem; background: var(--card); border: 1px solid var(--line-strong);
  box-shadow: var(--shadow-sm); color: var(--ink-2); font-weight: 600;
}}
.stApp .st-key-filters button:hover {{ border-color: var(--accent); color: var(--accent); background: var(--accent-tint); }}
[data-testid="stPopoverBody"] {{ border-radius: 16px; border: 1px solid var(--line); box-shadow: var(--shadow-md); background: var(--card); }}
.stApp [data-testid="stBaseButton-pills"], .stApp [data-testid="stBaseButton-pillsActive"] {{
  border-radius: 999px; border: 1px solid var(--line); background: var(--card); color: var(--ink-2);
  font-size: .86rem; padding: .3rem .95rem; box-shadow: var(--shadow-sm);
  transition: border-color .2s ease, color .2s ease, background .2s ease;
}}
.stApp [data-testid="stBaseButton-pills"]:hover {{ border-color: var(--accent); color: var(--accent); background: var(--accent-tint); }}
.stApp [data-testid="stBaseButton-pillsActive"] {{ border-color: var(--accent); color: var(--accent); background: var(--accent-tint); }}
.stApp [data-testid="stBaseButton-pills"] p, .stApp [data-testid="stBaseButton-pillsActive"] p {{ color: inherit; }}

/* ---- result header ------------------------------------------------------------- */
.cs-resulthead {{ display: flex; flex-wrap: wrap; align-items: baseline; justify-content: space-between; gap: .5rem 1.2rem; margin: 2.2rem 0 1.1rem; }}
.stApp .cs-resulthead h3 {{ font-family: var(--display); font-weight: 600; font-size: 1.45rem; letter-spacing: -.015em; margin: 0; padding: 0; color: var(--ink); }}
.stApp .cs-resulthead h3 em {{ font-weight: 500; }}
.cs-badges {{ display: flex; flex-wrap: wrap; gap: .4rem; }}
.cs-badges span {{
  font-family: var(--mono); font-size: .64rem; font-weight: 600; letter-spacing: .1em; text-transform: uppercase;
  color: var(--ink-3); border: 1px solid var(--line); background: var(--card); border-radius: 999px; padding: .22rem .65rem;
}}
.cs-badges span.is-accent {{ color: var(--accent); border-color: var(--accent-line); background: var(--accent-tint); }}

/* ---- poster grid ----------------------------------------------------------------- */
.cs-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(156px, 1fr)); gap: 1.7rem 1.15rem; }}
.stApp a.cs-card {{ display: block; color: var(--ink); text-decoration: none; min-width: 0; }}
.stApp a.cs-card:hover {{ color: var(--ink); }}
.cs-poster {{
  position: relative; aspect-ratio: 2 / 3; border-radius: 14px; overflow: hidden;
  background: var(--paper-alt); border: 1px solid var(--line); box-shadow: var(--shadow-sm);
  transition: transform .3s ease, box-shadow .3s ease, border-color .3s ease;
}}
.cs-poster img {{ width: 100%; height: 100%; object-fit: cover; display: block; transition: transform .6s ease; }}
.cs-card:hover .cs-poster, .cs-card:focus-visible .cs-poster {{
  transform: translateY(-4px); border-color: var(--accent-line); box-shadow: var(--shadow-md), 0 10px 30px var(--accent-glow);
}}
.cs-card:hover .cs-poster img {{ transform: scale(1.045); }}
.stApp a.cs-card:focus-visible {{ outline: none; }}
.cs-card:focus-visible .cs-poster {{ box-shadow: 0 0 0 3px var(--accent-glow); }}
.cs-noposter {{
  display: flex; align-items: center; justify-content: center; height: 100%; padding: 1rem; text-align: center;
  font-family: var(--display); font-style: italic; font-size: 1.05rem; color: var(--ink-3);
  background: radial-gradient(circle at 30% 20%, var(--accent-tint), transparent 60%), var(--paper-alt);
}}
.cs-chip, .cs-rank {{
  position: absolute; top: .5rem; font-family: var(--mono); font-size: .64rem; font-weight: 600; letter-spacing: .04em;
  border-radius: 999px; padding: .2rem .5rem; backdrop-filter: blur(6px); -webkit-backdrop-filter: blur(6px);
}}
.cs-chip {{ right: .5rem; color: var(--accent); background: rgba(255, 253, 250, .92); border: 1px solid var(--accent-line); }}
.cs-rank {{ left: .5rem; color: #fefaf5; background: rgba(28, 23, 20, .72); }}
.cs-more {{
  position: absolute; left: 0; right: 0; bottom: 0; padding: 2.6rem .8rem .75rem;
  background: linear-gradient(to top, rgba(28, 23, 20, .86), rgba(28, 23, 20, 0));
  color: #fefaf5; font-size: .78rem; font-weight: 600; letter-spacing: .01em;
  opacity: 0; transform: translateY(6px); transition: opacity .3s ease, transform .3s ease;
}}
.cs-card:hover .cs-more, .cs-card:focus-visible .cs-more {{ opacity: 1; transform: none; }}
.cs-name {{
  font-family: var(--display); font-weight: 600; font-size: 1rem; line-height: 1.25; color: var(--ink);
  margin: .75rem 0 .25rem; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden;
  transition: color .2s ease;
}}
.cs-card:hover .cs-name {{ color: var(--accent); }}
.cs-meta {{ font-family: var(--mono); font-size: .68rem; letter-spacing: .04em; color: var(--ink-3); }}
.cs-meta .star {{ color: var(--accent); }}
.cs-tags {{ font-size: .76rem; color: var(--ink-3); margin-top: .2rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
.cs-why {{ margin-top: .45rem; font-size: .76rem; line-height: 1.4; color: var(--ink-2); }}
.cs-why b {{ color: var(--accent); font-weight: 600; }}

/* ---- seed film ------------------------------------------------------------------- */
.cs-seed {{
  display: grid; grid-template-columns: 210px minmax(0, 1fr); gap: 1.9rem; align-items: start;
  background: linear-gradient(160deg, var(--accent-tint), var(--card) 58%); border: 1px solid var(--line);
  border-radius: 22px; padding: 1.4rem; box-shadow: var(--shadow-md); margin-top: 1rem;
}}
.cs-seed-art img, .cs-seed-art .cs-noposter {{ width: 100%; aspect-ratio: 2 / 3; object-fit: cover; border-radius: 14px; box-shadow: var(--shadow-md); display: block; }}
.cs-seed-body {{ padding-top: .3rem; min-width: 0; }}
.cs-seed-title {{
  font-family: var(--display); font-weight: 600; font-size: clamp(1.7rem, 3.2vw, 2.5rem); line-height: 1.08;
  letter-spacing: -.025em; color: var(--ink); margin: .55rem 0 .8rem;
}}
.cs-seed-title span {{ color: var(--ink-3); font-weight: 400; }}
.cs-facts, .cs-genres {{ display: flex; flex-wrap: wrap; gap: .4rem; margin-bottom: .6rem; }}
.cs-fact {{
  font-family: var(--mono); font-size: .64rem; font-weight: 600; letter-spacing: .08em; text-transform: uppercase;
  color: var(--ink-2); background: var(--card); border: 1px solid var(--line); border-radius: 999px; padding: .24rem .65rem;
}}
.cs-fact.is-accent {{ color: var(--accent); border-color: var(--accent-line); background: var(--accent-tint); }}
.cs-genre {{ font-size: .8rem; color: var(--ink-2); border-bottom: 1px solid var(--line-strong); padding-bottom: 1px; margin-right: .35rem; }}
.stApp p.cs-overview {{ color: var(--ink-2); font-size: .98rem; line-height: 1.75; max-width: 46rem; margin: .5rem 0 1.1rem; }}
.stApp a.cs-link {{
  display: inline-block; font-weight: 600; font-size: .9rem; text-decoration: none; color: var(--on-accent);
  background: var(--accent); border-radius: 999px; padding: .5rem 1.15rem; box-shadow: 0 4px 16px var(--accent-glow);
  transition: background .2s ease, transform .2s ease;
}}
.stApp a.cs-link:hover {{ background: var(--accent-strong); color: var(--on-accent); transform: translateY(-1px); }}

/* ---- stat tiles -------------------------------------------------------------------- */
.cs-stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(205px, 1fr)); gap: 1rem; }}
.cs-stat {{ background: var(--card); border: 1px solid var(--line); border-radius: 18px; padding: 1.15rem 1.25rem 1.1rem; box-shadow: var(--shadow-sm); }}
.cs-stat-value {{ font-family: var(--display); font-weight: 600; font-size: clamp(1.9rem, 3vw, 2.5rem); line-height: 1; letter-spacing: -.02em; color: var(--accent); }}
.cs-stat-value small {{ font-size: .5em; color: var(--ink-3); font-weight: 500; letter-spacing: 0; }}
.cs-stat-label {{ font-family: var(--mono); font-size: .64rem; font-weight: 600; letter-spacing: .1em; text-transform: uppercase; color: var(--ink-3); margin-top: .75rem; }}
.cs-stat-note {{ color: var(--ink-2); font-size: .85rem; line-height: 1.5; margin-top: .3rem; }}

/* ---- ablation ladder (HTML bars, values direct-labelled) ----------------------------- */
.cs-panel {{ background: var(--card); border: 1px solid var(--line); border-radius: 18px; padding: .5rem 1.4rem; box-shadow: var(--shadow-sm); }}
.cs-rung {{ display: grid; grid-template-columns: minmax(0, 17rem) minmax(0, 1fr) 5rem; gap: .35rem 1.2rem; align-items: center; padding: .85rem 0; border-top: 1px solid var(--line); }}
.cs-rung:first-child {{ border-top: 0; }}
.cs-rung-name {{ color: var(--ink-2); font-size: .92rem; line-height: 1.35; }}
.cs-rung-name small {{ display: block; font-family: var(--mono); font-size: .6rem; letter-spacing: .12em; text-transform: uppercase; color: var(--ink-3); margin-bottom: .15rem; }}
.cs-track {{ height: 10px; border-radius: 4px; background: var(--paper-alt); overflow: hidden; }}
.cs-fill {{ height: 100%; border-radius: 0 4px 4px 0; background: var(--neutral); }}
.cs-rung.is-key .cs-fill {{ background: var(--accent); }}
.cs-rung.is-key .cs-rung-name {{ color: var(--ink); font-weight: 600; }}
.cs-val {{ font-family: var(--mono); font-size: .82rem; font-weight: 600; text-align: right; color: var(--ink); }}
.cs-val small {{ display: block; font-size: .62rem; font-weight: 500; color: var(--ink-3); }}

/* ---- tables ----------------------------------------------------------------------------- */
.cs-table-wrap {{ overflow-x: auto; background: var(--card); border: 1px solid var(--line); border-radius: 18px; box-shadow: var(--shadow-sm); }}
.stApp table.cs-table {{ width: 100%; border-collapse: collapse; font-size: .9rem; margin: 0; border: 0; }}
.stApp .cs-table th {{
  font-family: var(--mono); font-size: .62rem; font-weight: 600; letter-spacing: .1em; text-transform: uppercase;
  color: var(--ink-3); text-align: right; padding: .85rem 1rem; background: var(--paper-alt);
  border: 0; border-bottom: 1px solid var(--line); white-space: nowrap;
}}
.stApp .cs-table td {{
  font-family: var(--mono); font-size: .82rem; text-align: right; color: var(--ink-2); padding: .8rem 1rem;
  border: 0; border-bottom: 1px solid var(--line); white-space: nowrap; font-variant-numeric: tabular-nums;
}}
.stApp .cs-table th:first-child, .stApp .cs-table td:first-child {{ text-align: left; }}
.stApp .cs-table td:first-child {{ font-family: inherit; font-size: .9rem; color: var(--ink); white-space: normal; min-width: 12rem; }}
.stApp .cs-table tr:last-child td {{ border-bottom: 0; }}
.stApp .cs-table tr.is-key td {{ background: var(--accent-tint); color: var(--ink); }}
.stApp .cs-table tr.is-key td:first-child {{ box-shadow: inset 3px 0 0 var(--accent); font-weight: 600; }}
.stApp .cs-table td.is-flag {{ color: var(--accent); font-weight: 600; }}
.stApp .cs-table td small {{ display: block; font-family: var(--mono); font-size: .62rem; letter-spacing: .06em; color: var(--ink-3); font-weight: 500; }}

/* ---- protocol steps, notes ---------------------------------------------------------------- */
.cs-steps {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 1rem; }}
.cs-step {{ background: var(--card); border: 1px solid var(--line); border-radius: 18px; padding: 1.15rem 1.25rem; box-shadow: var(--shadow-sm); }}
.cs-step-n {{ font-family: var(--mono); font-size: .64rem; font-weight: 600; letter-spacing: .14em; color: var(--accent); }}
.cs-step-t {{ font-family: var(--display); font-weight: 600; font-size: 1.1rem; color: var(--ink); margin: .4rem 0 .3rem; }}
.cs-step-d {{ color: var(--ink-2); font-size: .88rem; line-height: 1.6; }}
.cs-note {{
  border: 1px dashed var(--line-strong); border-radius: 14px; padding: .85rem 1.1rem; margin: 1rem 0 0;
  color: var(--ink-2); font-size: .88rem; line-height: 1.6; background: rgba(255, 253, 250, .65);
}}
.cs-note b {{ color: var(--ink); font-weight: 600; }}

/* ---- footer -------------------------------------------------------------------------------- */
.cs-footer {{
  margin-top: 4.5rem; padding-top: 1.3rem; border-top: 1px solid var(--line);
  display: flex; flex-wrap: wrap; justify-content: space-between; gap: .6rem 1.5rem;
  font-family: var(--mono); font-size: .64rem; font-weight: 500; letter-spacing: .1em; text-transform: uppercase; color: var(--ink-3);
}}
.stApp .cs-footer a {{ color: var(--accent); text-decoration: none; }}

@media (max-width: 640px) {{
  [data-testid="stMainBlockContainer"] {{ padding-top: 4.5rem; }}
  .cs-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 1.4rem .9rem; }}
  .cs-seed {{ grid-template-columns: minmax(0, 1fr); gap: .9rem; padding: 1.1rem; border-radius: 18px; }}
  .cs-seed-art {{ width: 132px; }}
  .stApp p.cs-overview {{ font-size: .9rem; }}
  .cs-rung {{ grid-template-columns: minmax(0, 1fr) auto; }}
  .cs-track {{ grid-column: 1 / -1; grid-row: 2; }}
}}
</style>
"""


def apply_components() -> None:
    html_block(components_css())

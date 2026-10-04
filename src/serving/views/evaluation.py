"""Evaluation — the offline numbers, read from the persisted results/ artifacts."""
import json

import streamlit as st

from src.serving import space_kit as kit

kit.page_header(
    "Offline evaluation",
    "Measured, <em>not claimed</em>",
    "The evaluation harness was built <b>before</b> any tuning. Every number on this page comes from "
    "it, on a held-out split, read straight from the repo's <code>results/</code> files, including "
    "the components that didn't help.",
)

p6 = kit.read_json("phase6_metrics.json")
p2 = kit.read_json("phase2a_metrics.json")
try:
    split = json.loads((kit.ROOT / "data" / "eval" / "cf_manifest.json").read_text(encoding="utf-8"))
except (OSError, ValueError):
    split = {}

ladder = [r for r in p6.get("ablation_ladder", []) if r.get("ndcg@10") not in (None, "")]
frontier = p6.get("frontier", {})
systems = frontier.get("systems", {})


def _system(prefix: str) -> dict:
    return next((v for k, v in systems.items() if k.lower().startswith(prefix)), {})


ranker = _system("specialized")
popularity = _system("reference")
llm = next((v for k, v in systems.items()
            if not k.lower().startswith(("specialized", "reference"))), {})


def _num(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


# ---- headline tiles ----------------------------------------------------------------------
cf_rung = next((r for r in ladder if "CF" in r["system"]), None)
cf_index = ladder.index(cf_rung) if cf_rung else -1
tiles = []
if cf_rung and cf_index > 0:
    before, after = _num(ladder[cf_index - 1]["ndcg@10"]), _num(cf_rung["ndcg@10"])
    tiles.append((f"{after / before:.1f}×", "Lift from collaborative filtering",
                  f"NDCG@10 {before:.4f} → {after:.4f} when ItemKNN joins the stack"))
    tiles.append((f"{after:.4f}", "Champion NDCG@10", "ItemKNN on the per-user temporal hold-out"))
if ranker and llm:
    tiles.append((f"{ranker['off_catalog_pct']:.0f}% <small>vs {llm['off_catalog_pct']:.1f}%</small>",
                  "Off-catalog picks", "The ranker can only return real items; the LLM can't make that promise"))
    tiles.append((f"{ranker['latency_ms']:.1f} ms", "Ranker latency per query",
                  f"vs ~{llm['latency_ms']:,.0f} ms for the LLM (estimated)"))
if tiles:
    kit.stat_tiles(tiles)

# ---- 01 ablation ladder -----------------------------------------------------------------------
NAMES = {
    "semantic (MiniLM centroid)": "Semantic search · MiniLM centroid",
    "+better_embeddings (e5-base-v2 centroid)": "+ better embeddings · e5-base-v2",
    "+CF (ItemKNN)": "+ collaborative filtering · ItemKNN",
    "+tuning (ALS Optuna-tuned)": "+ tuning · ALS with Optuna",
    "+sequential (BERT4Rec)": "+ sequential ranker · BERT4Rec",
    "+diversity (ItemKNN + MMR)": "+ diversity · ItemKNN + MMR",
}
if ladder:
    kit.section("01 · Capability ablation", "Only collaborative filtering moves the needle",
                "Each rung adds one capability on the personalized split (the last 20% of each user's "
                "ratings held out, by time). CF is the only jump; every rung after it is flat or lower.")
    top = max(_num(r["ndcg@10"]) for r in ladder)
    rows = []
    for r in ladder:
        ndcg, recall = _num(r["ndcg@10"]), _num(r.get("recall@20"))
        key = r is cf_rung
        rows.append(
            f"<div class='cs-rung{' is-key' if key else ''}' "
            f"title='NDCG@10 {ndcg:.4f} · recall@20 {recall:.4f}'>"
            f"<div class='cs-rung-name'><small>Rung {kit.esc(r['rung'])}</small>"
            f"{kit.esc(NAMES.get(r['system'], r['system']))}</div>"
            f"<div class='cs-track'><div class='cs-fill' style='width:{ndcg / top * 100:.1f}%'></div></div>"
            f"<div class='cs-val'>{ndcg:.4f}<small>recall {recall:.3f}</small></div></div>"
        )
    kit.html_block(f"<div class='cs-panel'>{''.join(rows)}</div>")
    kit.note("Bars show <b>NDCG@10</b> on one axis from zero; recall@20 is printed under each value. "
             "Source: <code>results/phase6_metrics.json</code>.")

# ---- 02 frontier LLM ------------------------------------------------------------------------------
if ranker and llm:
    n_users = frontier.get("n_users", 30)
    kit.section("02 · Frontier comparison", "An LLM matches the ranker, and can't be served",
                f"A frontier LLM, zero-shot and without catalog access, against the same split on a "
                f"deterministic {n_users}-user sample. It edges ItemKNN on NDCG, then recommends films "
                f"that aren't in the catalog.")

    def _ms(v: float) -> str:
        return f"{v:.2f} ms" if v < 1 else f"{v:,.0f} ms"

    def _usd(v: float) -> str:
        return "$0" if v == 0 else f"${v:.3f}"

    def _row(name, m, *, key=False, flag=False, estimated=False):
        est = "<small>estimated</small>" if estimated else ""
        off = f"<td class='is-flag'>{m['off_catalog_pct']:.1f}%</td>" if flag else \
              f"<td>{m['off_catalog_pct']:.1f}%</td>"
        return (f"<tr{' class=is-key' if key else ''}><td>{name}</td>"
                f"<td>{m['ndcg@10']:.4f}</td><td>{m['recall@20']:.4f}</td>{off}"
                f"<td>{_ms(m['latency_ms'])}{est}</td><td>{_usd(m['cost_usd_per_query'])}{est}</td></tr>")

    body = _row("ItemKNN · this project's ranker", ranker, key=True)
    body += _row("Frontier LLM · zero-shot", llm, flag=True, estimated=True)
    if popularity:
        body += _row("Popularity · reference", popularity)
    kit.html_block(
        "<div class='cs-table-wrap'><table class='cs-table'><thead><tr><th>System</th><th>NDCG@10</th>"
        "<th>Recall@20</th><th>Off-catalog</th><th>Latency</th><th>Cost / query</th></tr></thead>"
        f"<tbody>{body.replace('$', '&#36;')}</tbody></table></div>"
    )
    b = frontier.get("hallucination_breakdown", {})
    if b:
        kit.note(f"Of {b.get('total_reco', 0):,} LLM recommendations, <b>{b.get('off_catalog', 0)} were "
                 f"off-catalog</b>: {b.get('genuine_absent', 0)} titles that don't exist in the catalog "
                 f"and {b.get('title_variant', 0)} title variants. A CF model can only return items it "
                 "has seen, so that failure is structurally impossible for it. LLM latency and cost "
                 "are estimates; the ranker's are measured.")

# ---- 03 embedding bake-off --------------------------------------------------------------------------
models = p2.get("models", {})
if models:
    champ = p2.get("champion", {}).get("model")
    ev = p2.get("eval", {})
    kit.section("03 · Embedding bake-off", "Five encoders, a small spread",
                f"{ev.get('n_queries', 0):,} held-out “more like this” queries against "
                f"{ev.get('catalog_size', 0):,} films, relevance from MovieLens co-ratings.")
    ranked = sorted(models.items(), key=lambda kv: -kv[1]["ndcg@10"])
    body = ""
    for name, m in ranked:
        label = name.replace(" (current)", "")
        tag = " <small>champion · served here</small>" if name == champ else \
              " <small>originally shipped</small>" if "(current)" in name else ""
        body += (f"<tr{' class=is-key' if name == champ else ''}><td>{kit.esc(label)}{tag}</td>"
                 f"<td>{m['dim']}</td><td>{m['ndcg@10']:.4f}</td><td>{m['recall@20']:.4f}</td>"
                 f"<td>{m['search_ms_per_query']:.2f} ms</td><td>{m['encode_sec_9837docs']:.0f} s</td></tr>")
    kit.html_block(
        "<div class='cs-table-wrap'><table class='cs-table'><thead><tr><th>Model</th><th>Dim</th>"
        "<th>NDCG@10</th><th>Recall@20</th><th>Search</th><th>Encode catalog</th></tr></thead>"
        f"<tbody>{body}</tbody></table></div>"
    )
    base = next((m for n, m in models.items() if "(current)" in n), None)
    if champ in models and base:
        c = models[champ]
        kit.note(f"The champion is <b>{c['ndcg@10'] / base['ndcg@10']:.1f}×</b> the originally shipped "
                 f"MiniLM on NDCG@10 but <b>{c['encode_sec_9837docs'] / base['encode_sec_9837docs']:.1f}×</b> "
                 "slower to encode the catalog, and the whole spread is small next to what CF adds.")

# ---- 04 protocol ------------------------------------------------------------------------------------
kit.section("04 · Protocol", "How the numbers are kept honest")
users = f"{split['n_users_eval']:,} MovieLens users" if split.get("n_users_eval") else "Every MovieLens user"
held = (f" ({split['n_train_interactions']:,} train / {split['n_test_interactions']:,} test interactions)"
        if split.get("n_test_interactions") else "")
cands = f"{split['candidate_items']:,} training items" if split.get("candidate_items") else "training items"
kit.html_block(
    "<div class='cs-steps'>"
    "<div class='cs-step'><div class='cs-step-n'>01</div><div class='cs-step-t'>Temporal hold-out</div>"
    f"<div class='cs-step-d'>{users} each lose the last 20% of their ratings by time{held}. "
    "Models only ever see what came before.</div></div>"
    "<div class='cs-step'><div class='cs-step-n'>02</div><div class='cs-step-t'>No leakage</div>"
    f"<div class='cs-step-d'>Candidates are the {cands} only, and the split is checked for "
    f"train/test overlap: {split.get('train_test_overlap', 0)} shared interactions.</div></div>"
    "<div class='cs-step'><div class='cs-step-n'>03</div><div class='cs-step-t'>One harness</div>"
    "<div class='cs-step-d'>Every system on this page is scored by the same code on the same split, "
    "and the ones that lost are reported next to the winner.</div></div></div>"
)

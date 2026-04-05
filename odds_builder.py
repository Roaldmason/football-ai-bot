# ============================================================
#  odds_builder.py  –  Build smart accumulator tickets
# ============================================================

import logging
import itertools
import pandas as pd
from config import ACCA_TARGET_ODDS, ACCA_MIN_LEGS, ACCA_MAX_LEGS, ACCA_MIN_CONF

logger = logging.getLogger(__name__)


# ── Extract the best single pick per fixture ──────────────────
def best_pick(pred: dict) -> dict | None:
    """
    From a prediction row, extract the single highest-confidence,
    best-value pick across all five markets.
    Returns dict with: match, market, pick_label, conf, model_odds, bm_odds, is_value
    """
    candidates = []

    # 1X2
    r = pred["result_pick"]
    bm_r = pred.get(f"bm_{r}", 0) or pred.get(f"odds_{r}", 0)
    candidates.append({
        "market":     "1X2",
        "pick_label": {"home": f"{pred['home']} Win",
                       "draw": "Draw",
                       "away": f"{pred['away']} Win"}[r],
        "conf":       pred["result_conf"],
        "model_odds": pred[f"odds_{r}"],
        "bm_odds":    bm_r,
        "is_value":   pred.get("value_result", False),
    })

    # Over/Under
    ou = pred["ou_pick"]
    bm_ou = pred.get("bm_over25" if ou == "over" else "bm_under25", 0)
    candidates.append({
        "market":     "Over/Under 2.5",
        "pick_label": "Over 2.5 Goals" if ou == "over" else "Under 2.5 Goals",
        "conf":       pred["ou_conf"],
        "model_odds": pred[f"odds_{ou}25"],
        "bm_odds":    bm_ou or pred[f"odds_{ou}25"],
        "is_value":   pred.get("value_over25", False),
    })

    # BTTS
    btts = pred["btts_pick"]
    bm_btts = pred.get(f"bm_btts_{btts}", 0)
    candidates.append({
        "market":     "BTTS",
        "pick_label": "Both Teams to Score – Yes" if btts == "yes" else "Both Teams to Score – No",
        "conf":       pred["btts_conf"],
        "model_odds": pred[f"odds_btts_{btts}"],
        "bm_odds":    bm_btts or pred[f"odds_btts_{btts}"],
        "is_value":   pred.get("value_btts", False),
    })

    # Double Chance (only if result is uncertain)
    if pred["result_conf"] < 0.60:
        dc_map  = {"1x": ("dc_1x", "Home or Draw"), "x2": ("dc_x2", "Draw or Away"), "12": ("dc_12", "Home or Away")}
        best_dc = max(["1x", "x2", "12"], key=lambda k: pred.get(f"prob_dc_{k}", 0))
        key, label = dc_map[best_dc]
        candidates.append({
            "market":     "Double Chance",
            "pick_label": label,
            "conf":       pred.get(f"prob_dc_{best_dc}", 0),
            "model_odds": pred.get(f"odds_{key}", 1.30),
            "bm_odds":    pred.get(f"bm_{key}", 0) or pred.get(f"odds_{key}", 1.30),
            "is_value":   False,
        })

    # Filter by minimum confidence
    candidates = [c for c in candidates if c["conf"] >= ACCA_MIN_CONF]
    if not candidates:
        return None

    # Prefer value bets; otherwise pick highest confidence
    value_cands = [c for c in candidates if c["is_value"]]
    best = max(value_cands or candidates, key=lambda c: (c["is_value"], c["conf"]))
    best["match"] = f"{pred['home']} vs {pred['away']}"
    best["league"] = pred["league"]
    best["kickoff"] = pred["kickoff"]
    best["fixture_id"] = pred["fixture_id"]
    best["xg_home"] = pred.get("xg_home", "?")
    best["xg_away"] = pred.get("xg_away", "?")
    return best


# ── Accumulator builder ───────────────────────────────────────
def build_accumulator(predictions_df: pd.DataFrame) -> list[dict]:
    """
    From a predictions DataFrame, build the best accumulator
    ticket(s) targeting ACCA_TARGET_ODDS ≈ 5.00.

    Returns list of acca dicts, best first.
    """
    if predictions_df.empty:
        logger.warning("No predictions available.")
        return []

    # Get best pick per fixture
    picks = []
    for _, row in predictions_df.iterrows():
        p = best_pick(row.to_dict())
        if p:
            picks.append(p)

    if not picks:
        logger.warning("No picks meet confidence threshold.")
        return []

    # Sort by confidence descending
    picks.sort(key=lambda x: x["conf"], reverse=True)

    # ── Search for best combo closest to target odds ──────────
    best_accas = []

    for n_legs in range(ACCA_MIN_LEGS, ACCA_MAX_LEGS + 1):
        if len(picks) < n_legs:
            continue

        best_diff = float("inf")
        best_combo = None

        # Try top-15 picks only for performance
        pool = picks[:15]
        for combo in itertools.combinations(pool, n_legs):
            total = 1.0
            for leg in combo:
                odds = leg["bm_odds"] if leg["bm_odds"] and leg["bm_odds"] > 1.01 else leg["model_odds"]
                total *= odds
            diff = abs(total - ACCA_TARGET_ODDS)
            if diff < best_diff:
                best_diff = diff
                best_combo = list(combo)
                best_total = round(total, 2)

        if best_combo:
            avg_conf = sum(l["conf"] for l in best_combo) / len(best_combo)
            best_accas.append({
                "legs":       best_combo,
                "total_odds": best_total,
                "n_legs":     n_legs,
                "avg_conf":   round(avg_conf, 3),
                "diff":       round(best_diff, 3),
            })

    if not best_accas:
        return []

    # Sort: closest to target first, then by confidence
    best_accas.sort(key=lambda x: (x["diff"], -x["avg_conf"]))
    return best_accas


# ── Format for display ────────────────────────────────────────
def format_acca(acca: dict, idx: int = 1) -> str:
    """Return a pretty string for one accumulator ticket."""
    lines = [
        f"🎫 ACCUMULATOR #{idx}",
        f"{'─' * 38}",
    ]
    for i, leg in enumerate(acca["legs"], 1):
        odds = leg["bm_odds"] if leg["bm_odds"] and leg["bm_odds"] > 1.01 else leg["model_odds"]
        val_tag = " ⭐ VALUE" if leg["is_value"] else ""
        lines += [
            f"\n🔹 Leg {i}:  {leg['match']}",
            f"   📋 Market:     {leg['market']}",
            f"   ✅ Pick:       {leg['pick_label']}{val_tag}",
            f"   📊 Confidence: {leg['conf']:.0%}",
            f"   💰 Odds:       {odds:.2f}",
            f"   ⚽ xG:         {leg['xg_home']} – {leg['xg_away']}",
        ]

    total_odds = acca["total_odds"]
    avg_conf   = acca["avg_conf"]
    lines += [
        f"\n{'─' * 38}",
        f"📈 TOTAL ODDS:  {total_odds:.2f}",
        f"🧠 AVG CONF:   {avg_conf:.0%}",
        f"🏷  LEGS:       {acca['n_legs']}",
    ]
    return "\n".join(lines)


def format_all_accas(accas: list[dict]) -> str:
    if not accas:
        return "⚠️ No accumulators could be built today."
    parts = [format_acca(a, i + 1) for i, a in enumerate(accas[:3])]
    return "\n\n".join(parts)


if __name__ == "__main__":
    import os
    from config import DATA_DIR
    path = f"{DATA_DIR}/latest_predictions.csv"
    if os.path.exists(path):
        df = pd.read_csv(path)
        accas = build_accumulator(df)
        print(format_all_accas(accas))
    else:
        print("Run predictor.py first to generate predictions.")

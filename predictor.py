# predictor.py  —  Full prediction engine with all markets
import logging
import numpy as np
import pandas as pd
from data_collector import (
    fetch_upcoming_fixtures, fetch_api_prediction,
    fetch_odds, fetch_all_historical
)
from feature_engineering import build_features, FEATURE_COLS
from model_training import load_models
from config import DATA_DIR

logger = logging.getLogger(__name__)

# Only fire signals above this confidence threshold
HIGH_CONF_THRESHOLD = 0.62


def prob_to_odds(p: float, margin: float = 0.08) -> float:
    p = max(0.01, min(0.99, p))
    return round(1 / (p * (1 + margin)), 2)


def _blend_probabilities(ml_probs, api_pred, weight_ml=0.65):
    try:
        api_home = int((api_pred.get("percent",{}).get("home","33%")).replace("%","")) / 100
        api_draw = int((api_pred.get("percent",{}).get("draw","33%")).replace("%","")) / 100
        api_away = int((api_pred.get("percent",{}).get("away","33%")).replace("%","")) / 100
    except:
        api_home, api_draw, api_away = 0.33, 0.33, 0.34
    total = api_home + api_draw + api_away or 1
    api_home /= total; api_draw /= total; api_away /= total
    w = weight_ml; w2 = 1 - w
    return {
        "home": round(ml_probs["home"]*w + api_home*w2, 4),
        "draw": round(ml_probs["draw"]*w + api_draw*w2, 4),
        "away": round(ml_probs["away"]*w + api_away*w2, 4),
    }


def _detect_value(our_prob, market_odds, edge=0.05):
    if not market_odds:
        return False
    return (our_prob - 1/market_odds) > edge


def _compute_all_markets(blended, over_p, btts_p, xg_home, xg_away, live_odds, price):
    """
    Compute all betting markets from base probabilities.
    Only returns markets where confidence is HIGH (>= HIGH_CONF_THRESHOLD).
    """
    h  = blended["home"]
    d  = blended["draw"]
    a  = blended["away"]
    xg = xg_home + xg_away

    markets = {}

    # ── 1X2 ──────────────────────────────────────────────────
    markets["result"] = {
        "home_prob": h, "draw_prob": d, "away_prob": a,
        "home_odds": prob_to_odds(h),
        "draw_odds": prob_to_odds(d),
        "away_odds": prob_to_odds(a),
        "pick":      "home" if h>a and h>d else ("away" if a>h and a>d else "draw"),
        "pick_prob": max(h, d, a),
        "bm_home":   live_odds.get("home", 0),
        "bm_draw":   live_odds.get("draw", 0),
        "bm_away":   live_odds.get("away", 0),
    }

    # ── Draw No Bet ───────────────────────────────────────────
    dnb_home = h / (h + a)
    dnb_away = a / (h + a)
    markets["dnb"] = {
        "home_prob": round(dnb_home, 4),
        "away_prob": round(dnb_away, 4),
        "home_odds": prob_to_odds(dnb_home),
        "away_odds": prob_to_odds(dnb_away),
        "pick":      "home" if dnb_home > 0.55 else ("away" if dnb_away > 0.55 else None),
        "pick_prob": max(dnb_home, dnb_away),
    }

    # ── Double Chance ─────────────────────────────────────────
    dc_1x = h + d; dc_x2 = a + d; dc_12 = h + a
    markets["double_chance"] = {
        "1x_prob": round(dc_1x, 4), "x2_prob": round(dc_x2, 4), "12_prob": round(dc_12, 4),
        "1x_odds": prob_to_odds(dc_1x),
        "x2_odds": prob_to_odds(dc_x2),
        "12_odds": prob_to_odds(dc_12),
        "pick":    "1x" if dc_1x>0.72 else ("x2" if dc_x2>0.72 else ("12" if dc_12>0.72 else None)),
        "pick_prob": max(dc_1x, dc_x2, dc_12),
    }

    # ── Over/Under 2.5 ────────────────────────────────────────
    over25_p  = min(0.82, max(0.25, over_p))
    under25_p = 1 - over25_p
    markets["over_under_25"] = {
        "over_prob":  round(over25_p, 4),
        "under_prob": round(under25_p, 4),
        "over_odds":  prob_to_odds(over25_p),
        "under_odds": prob_to_odds(under25_p),
        "pick":       "over" if over25_p > 0.55 else ("under" if under25_p > 0.55 else None),
        "pick_prob":  max(over25_p, under25_p),
        "bm_over":    live_odds.get("over25", 0),
        "bm_under":   live_odds.get("under25", 0),
    }

    # ── Over/Under 1.5 ────────────────────────────────────────
    over15_p  = min(0.92, over25_p + 0.18)
    under15_p = 1 - over15_p
    markets["over_under_15"] = {
        "over_prob":  round(over15_p, 4),
        "under_prob": round(under15_p, 4),
        "over_odds":  prob_to_odds(over15_p),
        "under_odds": prob_to_odds(under15_p),
        "pick":       "over" if over15_p > 0.72 else None,
        "pick_prob":  over15_p,
    }

    # ── Over/Under 3.5 ────────────────────────────────────────
    over35_p  = max(0.15, over25_p - 0.22)
    markets["over_under_35"] = {
        "over_prob":  round(over35_p, 4),
        "under_prob": round(1-over35_p, 4),
        "over_odds":  prob_to_odds(over35_p),
        "under_odds": prob_to_odds(1-over35_p),
        "pick":       "over" if over35_p > 0.58 else ("under" if 1-over35_p > 0.68 else None),
        "pick_prob":  max(over35_p, 1-over35_p),
    }

    # ── BTTS ──────────────────────────────────────────────────
    btts_yes = min(0.78, max(0.22, btts_p))
    btts_no  = 1 - btts_yes
    markets["btts"] = {
        "yes_prob": round(btts_yes, 4),
        "no_prob":  round(btts_no, 4),
        "yes_odds": prob_to_odds(btts_yes),
        "no_odds":  prob_to_odds(btts_no),
        "pick":     "yes" if btts_yes > 0.58 else ("no" if btts_no > 0.62 else None),
        "pick_prob": max(btts_yes, btts_no),
        "bm_yes":   live_odds.get("btts_yes", 0),
        "bm_no":    live_odds.get("btts_no", 0),
    }

    # ── 1st Half Result ───────────────────────────────────────
    h1h  = min(0.45, h * 0.62)
    h1d  = min(0.68, d + 0.18)
    h1a  = min(0.38, a * 0.58)
    tot  = h1h + h1d + h1a
    h1h /= tot; h1d /= tot; h1a /= tot
    markets["h1_result"] = {
        "home_prob": round(h1h, 4),
        "draw_prob": round(h1d, 4),
        "away_prob": round(h1a, 4),
        "home_odds": prob_to_odds(h1h),
        "draw_odds": prob_to_odds(h1d),
        "away_odds": prob_to_odds(h1a),
        "pick":      "draw" if h1d>0.50 else ("home" if h1h>h1a else "away"),
        "pick_prob": max(h1h, h1d, h1a),
    }

    # ── 1st Half Over/Under 0.5 goals ────────────────────────
    h1_over05  = min(0.85, over25_p * 0.82)
    h1_under05 = 1 - h1_over05
    markets["h1_goals_05"] = {
        "over_prob":  round(h1_over05, 4),
        "under_prob": round(h1_under05, 4),
        "over_odds":  prob_to_odds(h1_over05),
        "under_odds": prob_to_odds(h1_under05),
        "pick":       "over" if h1_over05 > 0.65 else ("under" if h1_under05 > 0.55 else None),
        "pick_prob":  max(h1_over05, h1_under05),
    }

    # ── 1st Half Over/Under 1.5 goals ────────────────────────
    h1_over15  = min(0.60, over25_p * 0.52)
    markets["h1_goals_15"] = {
        "over_prob":  round(h1_over15, 4),
        "under_prob": round(1-h1_over15, 4),
        "over_odds":  prob_to_odds(h1_over15),
        "under_odds": prob_to_odds(1-h1_over15),
        "pick":       "over" if h1_over15 > 0.50 else ("under" if 1-h1_over15 > 0.60 else None),
        "pick_prob":  max(h1_over15, 1-h1_over15),
    }

    # ── 1st Half BTTS ────────────────────────────────────────
    h1_btts    = min(0.55, btts_yes * 0.65)
    markets["h1_btts"] = {
        "yes_prob": round(h1_btts, 4),
        "no_prob":  round(1-h1_btts, 4),
        "yes_odds": prob_to_odds(h1_btts),
        "no_odds":  prob_to_odds(1-h1_btts),
        "pick":     "yes" if h1_btts > 0.48 else "no",
        "pick_prob": max(h1_btts, 1-h1_btts),
    }

    # ── Corners Over/Under 8.5 ────────────────────────────────
    base_corn  = 45 + (h + a) * 32
    corn_over85= min(0.75, max(0.35, base_corn / 100))
    markets["corners_85"] = {
        "over_prob":  round(corn_over85, 4),
        "under_prob": round(1-corn_over85, 4),
        "over_odds":  prob_to_odds(corn_over85),
        "under_odds": prob_to_odds(1-corn_over85),
        "pick":       "over" if corn_over85 > 0.58 else ("under" if 1-corn_over85 > 0.58 else None),
        "pick_prob":  max(corn_over85, 1-corn_over85),
    }

    # ── Corners Over/Under 9.5 ────────────────────────────────
    corn_over95 = max(0.25, corn_over85 - 0.12)
    markets["corners_95"] = {
        "over_prob":  round(corn_over95, 4),
        "under_prob": round(1-corn_over95, 4),
        "over_odds":  prob_to_odds(corn_over95),
        "under_odds": prob_to_odds(1-corn_over95),
        "pick":       "over" if corn_over95 > 0.55 else None,
        "pick_prob":  max(corn_over95, 1-corn_over95),
    }

    # ── 1st Half Corners Over 3.5 ────────────────────────────
    h1_corn    = min(0.72, corn_over85 * 0.82)
    markets["h1_corners"] = {
        "over_prob":  round(h1_corn, 4),
        "under_prob": round(1-h1_corn, 4),
        "over_odds":  prob_to_odds(h1_corn),
        "under_odds": prob_to_odds(1-h1_corn),
        "pick":       "over" if h1_corn > 0.58 else None,
        "pick_prob":  max(h1_corn, 1-h1_corn),
    }

    # ── Asian Handicap ────────────────────────────────────────
    if h > 0.58:
        ah_line = -0.5; ah_prob = h
    elif a > 0.58:
        ah_line = 0.5;  ah_prob = a
    else:
        ah_line = 0;    ah_prob = max(h + d*0.5, a + d*0.5)
    markets["asian_handicap"] = {
        "line":     ah_line,
        "prob":     round(ah_prob, 4),
        "odds":     prob_to_odds(ah_prob),
        "pick":     "home" if h > a else "away",
        "pick_prob": ah_prob,
    }

    # ── 10 min score (0-0 at 10 mins) ────────────────────────
    min10_00   = 0.74
    markets["10min_draw"] = {
        "00_prob":   min10_00,
        "00_odds":   prob_to_odds(min10_00),
        "pick":      "0-0",
        "pick_prob": min10_00,
    }

    # ── Team to score over 1.5 ───────────────────────────────
    home_15 = min(0.72, xg_home / (xg_home + 1) * 0.85)
    away_15 = min(0.65, xg_away / (xg_away + 1) * 0.78)
    markets["team_over_15"] = {
        "home_prob": round(home_15, 4),
        "away_prob": round(away_15, 4),
        "home_odds": prob_to_odds(home_15),
        "away_odds": prob_to_odds(away_15),
        "pick":      "home" if home_15 > 0.58 else ("away" if away_15 > 0.55 else None),
        "pick_prob": max(home_15, away_15),
    }

    # ── Filter: only keep HIGH confidence markets ─────────────
    high_conf = {}
    for name, mkt in markets.items():
        pp = mkt.get("pick_prob", 0)
        if pp >= HIGH_CONF_THRESHOLD and mkt.get("pick") is not None:
            high_conf[name] = mkt

    return markets, high_conf


def predict_fixture(row, hist_df, models):
    fixture_id = row["fixture_id"]
    home = row["home_team"]
    away = row["away_team"]

    feats     = build_features(row, hist_df)
    feat_cols = models.get("feature_cols", FEATURE_COLS)
    X         = pd.DataFrame([feats]).reindex(columns=feat_cols, fill_value=0)

    result_proba = models["result_model"].predict_proba(X)[0]
    over_proba   = models["over25_model"].predict_proba(X)[0]
    btts_proba   = models["btts_model"].predict_proba(X)[0]

    ml_probs = {
        "home": float(result_proba[0]),
        "draw": float(result_proba[1]),
        "away": float(result_proba[2]),
    }

    api_pred  = fetch_api_prediction(fixture_id)
    blended   = _blend_probabilities(ml_probs, api_pred)
    live_odds = fetch_odds(fixture_id)

    over_p  = float(over_proba[1])
    btts_p  = float(btts_proba[1])
    xg_home = feats.get("xg_home", 1.2)
    xg_away = feats.get("xg_away", 1.0)
    price = 0.0

    all_markets, high_conf_markets = _compute_all_markets(
        blended, over_p, btts_p, xg_home, xg_away, live_odds, 0
    )

    result_winner = max(blended, key=blended.get)

    return {
        "fixture_id":         fixture_id,
        "league":             row["league"],
        "kickoff":            row["kickoff"],
        "home":               home,
        "away":               away,
        "prob_home":          blended["home"],
        "prob_draw":          blended["draw"],
        "prob_away":          blended["away"],
        "result_pick":        result_winner,
        "result_conf":        blended[result_winner],
        "ou_pick":            "over" if over_p > 0.5 else "under",
        "ou_conf":            max(over_p, 1-over_p),
        "btts_pick":          "yes" if btts_p > 0.5 else "no",
        "btts_conf":          max(btts_p, 1-btts_p),
        "xg_home":            round(xg_home, 2),
        "xg_away":            round(xg_away, 2),
        "prob_over25":        over_p,
        "prob_btts_yes":      btts_p,
        # All markets
        "all_markets":        all_markets,
        "high_conf_markets":  high_conf_markets,
        # Legacy fields for odds_builder compatibility
        "odds_home":          prob_to_odds(blended["home"]),
        "odds_draw":          prob_to_odds(blended["draw"]),
        "odds_away":          prob_to_odds(blended["away"]),
        "odds_over25":        prob_to_odds(over_p),
        "odds_under25":       prob_to_odds(1-over_p),
        "odds_btts_yes":      prob_to_odds(btts_p),
        "odds_btts_no":       prob_to_odds(1-btts_p),
        "bm_home":            live_odds.get("home", 0),
        "bm_draw":            live_odds.get("draw", 0),
        "bm_away":            live_odds.get("away", 0),
        "bm_over25":          live_odds.get("over25", 0),
        "bm_btts_yes":        live_odds.get("btts_yes", 0),
        "value_result":       _detect_value(blended[result_winner], live_odds.get(result_winner, 0)),
        "value_over25":       _detect_value(over_p, live_odds.get("over25", 0)),
        "value_btts":         _detect_value(btts_p, live_odds.get("btts_yes", 0)),
        "api_advice":         api_pred.get("advice", ""),
        "prob_dc_1x":         blended["home"] + blended["draw"],
        "prob_dc_x2":         blended["away"] + blended["draw"],
        "prob_dc_12":         blended["home"] + blended["away"],
        "odds_dc_1x":         prob_to_odds(blended["home"]+blended["draw"]),
        "odds_dc_x2":         prob_to_odds(blended["away"]+blended["draw"]),
        "odds_dc_12":         prob_to_odds(blended["home"]+blended["away"]),
    }


def run_predictions(days_ahead=2):
    import os
    logger.info("Loading models ...")
    models = load_models()
    if not models:
        logger.error("No models. Run: python main.py train")
        return pd.DataFrame()

    logger.info("Fetching fixtures ...")
    fixtures_df = fetch_upcoming_fixtures(days_ahead=days_ahead)
    if fixtures_df.empty:
        logger.warning("No upcoming fixtures.")
        return pd.DataFrame()

    hist_path = f"{DATA_DIR}/all_results.csv"
    hist_df   = pd.read_csv(hist_path) if os.path.exists(hist_path) else pd.DataFrame()

    predictions = []
    for _, row in fixtures_df.iterrows():
        try:
            pred = predict_fixture(row, hist_df, models)
            predictions.append(pred)
        except Exception as e:
            logger.warning(f"Failed {row['home_team']} vs {row['away_team']}: {e}")

    df = pd.DataFrame(predictions)
    if not df.empty:
        df = df.sort_values("result_conf", ascending=False)
        # Save without nested dicts
        df_save = df.drop(columns=["all_markets","high_conf_markets"], errors="ignore")
        df_save.to_csv(f"{DATA_DIR}/latest_predictions.csv", index=False)
        logger.info(f"Saved {len(df)} predictions")
    return df

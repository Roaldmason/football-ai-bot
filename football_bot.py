#!/usr/bin/env python3
# ============================================================
#  football_bot.py  —  All-in-one Football AI Prediction Bot
#  Usage:
#    python football_bot.py train     # Fetch data & train ML models
#    python football_bot.py predict   # Run predictions (prints to console)
#    python football_bot.py acca      # Print accumulator to console
#    python football_bot.py bot       # Start Telegram bot (polling)
#    python football_bot.py schedule  # Auto-run daily pipeline (VPS)
# ============================================================

import os, sys, time, json, logging, pickle, asyncio, itertools, threading
import numpy as np
import pandas as pd
import requests
import schedule as schedule_lib
from datetime import datetime, timedelta, timezone

# ── Telegram ─────────────────────────────────────────────────
from telegram import Bot, Update, BotCommand
from telegram.ext import Application, CommandHandler, ContextTypes

# ── ML ───────────────────────────────────────────────────────
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report, accuracy_score

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

# ─────────────────────────────────────────────────────────────
#  SECTION 1 — CONFIGURATION  (edit these values)
# ─────────────────────────────────────────────────────────────

API_FOOTBALL_KEY    = "e0afc3e5ec334d44bc3d207811fbc764"
API_FOOTBALL_BASE   = "https://api.football-data.org/v4"
API_FOOTBALL_SEASON = 2025          # 2025 = 2025/26 season

TELEGRAM_BOT_TOKEN  = "8573812227:AAGyn4ab2tkWxhG8Kn5W7DbhYQoLGaDIlnI"
TELEGRAM_CHANNEL_ID = "8573812227"  # Your channel/chat ID (no leading space)

LEAGUES = {
    "Premier League":    {"id": "PL",   "flag": "PL"},
    "La Liga":           {"id": "PD",   "flag": "LL"},
    "Bundesliga":        {"id": "BL1",  "flag": "BL"},
    "Eredivisie":        {"id": "DED",  "flag": "ED"},
    "Liga Portugal":     {"id": "PPL",  "flag": "LP"},
    "Ligue 1":           {"id": "FL1",  "flag": "L1"},
    "Champions League":  {"id": "CL",   "flag": "UCL"},
    "Europa League":     {"id": "EL",   "flag": "UEL"},
    "Conference League": {"id": "UECL", "flag": "UECL"},
        "Saudi Pro League":  {"id": "SPL",   "flag": "SPL"},
}

ACCA_TARGET_ODDS = 5.00
ACCA_MIN_LEGS    = 3
ACCA_MAX_LEGS    = 5
ACCA_MIN_CONF    = 0.55
DAILY_RUN_TIME   = "08:00"          # UTC
HIGH_CONF_THRESHOLD = 0.62

DATA_DIR  = "data"
MODEL_DIR = "models"
LOG_FILE  = "bot.log"
SUBS_FILE = "subscribers.json"

os.makedirs(DATA_DIR,  exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────
#  SECTION 2 — LOGGING
# ─────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(LOG_FILE)],
)
logger = logging.getLogger("football_bot")

# ─────────────────────────────────────────────────────────────
#  SECTION 3 — DATA COLLECTOR
# ─────────────────────────────────────────────────────────────

API_HEADERS = {"X-Auth-Token": API_FOOTBALL_KEY}

# Timeouts per attempt: start short so failures are detected quickly
_TIMEOUTS = [8, 12, 20]


def check_connectivity() -> bool:
    """Quick check: can we reach the API at all?"""
    try:
        r = requests.get(
            "https://api.football-data.org/v4/competitions",
            headers=API_HEADERS, timeout=8
        )
        if r.status_code == 200:
            logger.info("API connectivity OK.")
            return True
        if r.status_code == 403:
            logger.error(
                "403 Forbidden — your API key is invalid or expired.\n"
                "  1. Go to https://www.football-data.org/client/login\n"
                "  2. Log in and copy your API token\n"
                "  3. Paste it into API_FOOTBALL_KEY at the top of this file"
            )
        else:
            logger.error(f"API returned HTTP {r.status_code}")
        return False
    except requests.exceptions.ConnectTimeout:
        logger.error(
            "Cannot reach api.football-data.org (connection timed out).\n"
            "  Possible causes:\n"
            "    1. No internet connection\n"
            "    2. Firewall / antivirus blocking outbound HTTPS\n"
            "    3. The site is down — check https://www.football-data.org\n"
            "  Fix options:\n"
            "    A. Try a VPN (e.g. Proton VPN free tier)\n"
            "    B. Temporarily disable Windows Defender Firewall\n"
            "    C. Run from a different network (phone hotspot)\n"
            "    D. Run: python football_bot.py test    to diagnose further"
        )
        return False
    except requests.RequestException as e:
        logger.error(f"Connectivity check failed: {e}")
        return False


def _api_get(path: str, params: dict = None) -> dict:
    """GET football-data.org v4 with rate-limit handling (free tier = 10 req/min)."""
    url = f"{API_FOOTBALL_BASE}/{path}"
    for attempt in range(3):
        timeout = _TIMEOUTS[attempt]
        try:
            r = requests.get(url, headers=API_HEADERS, params=params, timeout=timeout)
            if r.status_code == 429:
                wait = int(r.headers.get("X-RequestCounter-Reset", 60))
                logger.warning(f"Rate limited — waiting {wait}s ...")
                time.sleep(wait)
                continue
            if r.status_code == 403:
                logger.error("403 Forbidden — invalid API key")
                return {}
            if r.status_code == 404:
                logger.warning(f"404 Not Found: {url} — league may not be available on free tier")
                return {}
            r.raise_for_status()
            time.sleep(7)   # respect 10 req/min limit
            return r.json()
        except requests.exceptions.ConnectTimeout:
            logger.error(f"Timeout attempt {attempt+1}/3 for {path} (waited {timeout}s)")
            time.sleep(2 ** attempt)
        except requests.RequestException as e:
            logger.error(f"Request failed ({attempt+1}/3): {e}")
            time.sleep(2 ** attempt)
    logger.error(f"All 3 attempts failed for: {path}")
    return {}


def fetch_upcoming_fixtures(days_ahead: int = 3) -> pd.DataFrame:
    rows  = []
    today = datetime.utcnow().date()
    to    = today + timedelta(days=days_ahead)
    for name, info in LEAGUES.items():
        code = info["id"]
        data = _api_get(f"competitions/{code}/matches", {
            "dateFrom": str(today), "dateTo": str(to),
            "status":   "SCHEDULED,TIMED",
        })
        for m in data.get("matches", []):
            rows.append({
                "fixture_id": m["id"],
                "league":     name,
                "league_id":  code,
                "home_team":  m["homeTeam"]["name"],
                "home_id":    m["homeTeam"]["id"],
                "away_team":  m["awayTeam"]["name"],
                "away_id":    m["awayTeam"]["id"],
                "kickoff":    m["utcDate"],
                "round":      m.get("matchday", ""),
                "stage":      m.get("stage", ""),
            })
        logger.info(f"{name}: {len(data.get('matches', []))} upcoming fixtures")
    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_csv(f"{DATA_DIR}/upcoming_fixtures.csv", index=False)
    return df


def fetch_historical_results(code: str, season: int = None) -> pd.DataFrame:
    season = season or API_FOOTBALL_SEASON
    data   = _api_get(f"competitions/{code}/matches", {
        "season": season, "status": "FINISHED",
    })
    rows = []
    for m in data.get("matches", []):
        ft = m.get("score", {}).get("fullTime", {})
        hg = ft.get("home") or 0
        ag = ft.get("away") or 0
        rows.append({
            "fixture_id": m["id"],
            "date":       m["utcDate"][:10],
            "home_team":  m["homeTeam"]["name"],
            "home_id":    m["homeTeam"]["id"],
            "away_team":  m["awayTeam"]["name"],
            "away_id":    m["awayTeam"]["id"],
            "home_goals": hg, "away_goals": ag,
            "home_win":   1 if hg > ag else 0,
            "draw":       1 if hg == ag else 0,
            "away_win":   1 if ag > hg else 0,
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_csv(f"{DATA_DIR}/results_{code}_{season}.csv", index=False)
        logger.info(f"{code}: {len(df)} results saved")
    return df


def fetch_all_historical() -> pd.DataFrame:
    if not check_connectivity():
        logger.error(
            "Aborting — no API connection. Run: python football_bot.py test"
        )
        return pd.DataFrame()
    frames = []
    for name, info in LEAGUES.items():
        # Try current season first, fall back to 2024 if empty
        df = fetch_historical_results(info["id"], API_FOOTBALL_SEASON)
        if df.empty:
            logger.info(f"{name}: no data for season {API_FOOTBALL_SEASON}, trying 2024 ...")
            df = fetch_historical_results(info["id"], 2024)
        if not df.empty:
            df["league"] = name
            frames.append(df)
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not combined.empty:
        combined.to_csv(f"{DATA_DIR}/all_results.csv", index=False)
        logger.info(f"Saved {len(combined)} total results across all leagues")
    else:
        logger.error("No data collected. API may be unreachable. Run: python football_bot.py test")
    return combined


def fetch_h2h(home_id: int, away_id: int, last: int = 10) -> pd.DataFrame:
    data = _api_get(f"teams/{home_id}/matches", {"status": "FINISHED", "limit": 50})
    rows = []
    for m in data.get("matches", []):
        ht_id = m["homeTeam"]["id"]
        at_id = m["awayTeam"]["id"]
        if not ({ht_id, at_id} == {int(home_id), int(away_id)}):
            continue
        ft = m.get("score", {}).get("fullTime", {})
        rows.append({
            "date":       m["utcDate"][:10],
            "home_team":  m["homeTeam"]["name"],
            "away_team":  m["awayTeam"]["name"],
            "home_goals": ft.get("home") or 0,
            "away_goals": ft.get("away") or 0,
        })
        if len(rows) >= last:
            break
    return pd.DataFrame(rows)

# ─────────────────────────────────────────────────────────────
#  SECTION 4 — FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────

FEATURE_COLS = [
    "home_form_rate", "away_form_rate",
    "home_avg_scored", "away_avg_scored",
    "home_avg_conceded", "away_avg_conceded",
    "home_btts_rate", "away_btts_rate",
    "home_cs_rate", "away_cs_rate",
    "xg_home", "xg_away", "xg_total", "form_diff",
]


def _form_from_results(df, team, n=5):
    default = {"form_pts": 0, "form_gf": 0, "form_ga": 0, "form_rate": 0.0}
    if df is None or df.empty or "home_team" not in df.columns:
        return default
    mask  = (df["home_team"] == team) | (df["away_team"] == team)
    games = df[mask].sort_values("date", ascending=False).head(n)
    if games.empty:
        return default
    pts, gf, ga = 0, 0, 0
    for _, row in games.iterrows():
        if row["home_team"] == team:
            gf += row["home_goals"]; ga += row["away_goals"]
            if   row["home_goals"] > row["away_goals"]:  pts += 3
            elif row["home_goals"] == row["away_goals"]: pts += 1
        else:
            gf += row["away_goals"]; ga += row["home_goals"]
            if   row["away_goals"] > row["home_goals"]:  pts += 3
            elif row["away_goals"] == row["home_goals"]: pts += 1
    return {
        "form_pts":  pts,
        "form_gf":   round(gf / len(games), 2),
        "form_ga":   round(ga / len(games), 2),
        "form_rate": round(pts / (n * 3), 3),
    }


def _season_averages(df, team):
    default = {"avg_scored":1.2,"avg_conceded":1.2,"home_avg_scored":1.2,
               "away_avg_scored":1.0,"clean_sheet_rate":0.3,"btts_rate":0.5}
    if df is None or df.empty or "home_team" not in df.columns:
        return default
    hg    = df[df["home_team"] == team]
    ag    = df[df["away_team"] == team]
    all_g = pd.concat([hg, ag])
    if all_g.empty:
        return default
    scored   = list(hg["home_goals"]) + list(ag["away_goals"])
    conceded = list(hg["away_goals"]) + list(ag["home_goals"])
    btts = sum(1 for _, r in all_g.iterrows()
               if r["home_goals"] > 0 and r["away_goals"] > 0) / max(len(all_g), 1)
    cs   = sum(1 for g in conceded if g == 0) / max(len(conceded), 1)
    return {
        "avg_scored":       round(np.mean(scored),   2) if scored   else 1.2,
        "avg_conceded":     round(np.mean(conceded), 2) if conceded else 1.2,
        "home_avg_scored":  round(hg["home_goals"].mean(), 2) if not hg.empty else 1.2,
        "away_avg_scored":  round(ag["away_goals"].mean(), 2) if not ag.empty else 1.0,
        "clean_sheet_rate": round(cs,   3),
        "btts_rate":        round(btts, 3),
    }


def _h2h_features(h2h_df, home_team):
    if h2h_df is None or h2h_df.empty:
        return {"h2h_home_win_rate": 0.33, "h2h_avg_goals": 2.5}
    results, total_goals = [], []
    for _, row in h2h_df.iterrows():
        total_goals.append((row["home_goals"] or 0) + (row["away_goals"] or 0))
        if row["home_team"] == home_team:
            results.append(1 if row["home_goals"] > row["away_goals"] else 0)
        else:
            results.append(1 if row["away_goals"] > row["home_goals"] else 0)
    return {
        "h2h_home_win_rate": round(np.mean(results),     3),
        "h2h_avg_goals":     round(np.mean(total_goals), 2),
    }


def build_features(fixture_row, hist_df):
    home = fixture_row["home_team"]
    away = fixture_row["away_team"]
    hid  = fixture_row.get("home_id", 0)
    aid  = fixture_row.get("away_id", 0)
    if hist_df is not None and not hist_df.empty and "league" in hist_df.columns:
        lhist = hist_df[hist_df["league"] == fixture_row["league"]]
        if lhist.empty:
            lhist = hist_df
    else:
        lhist = pd.DataFrame()
    hf  = _form_from_results(lhist, home)
    af  = _form_from_results(lhist, away)
    hs  = _season_averages(lhist, home)
    as_ = _season_averages(lhist, away)
    h2h_df = fetch_h2h(hid, aid) if hid and aid else pd.DataFrame()
    h2h    = _h2h_features(h2h_df, home)
    ha  = hs["avg_scored"]   / 1.35
    hd  = hs["avg_conceded"] / 1.35
    aa  = as_["avg_scored"]  / 1.35
    ad  = as_["avg_conceded"]/ 1.35
    xgh = ha * ad * 1.35 * 1.10
    xga = aa * hd * 1.35
    return {
        "home_form_rate":    hf["form_rate"],  "away_form_rate":    af["form_rate"],
        "home_avg_scored":   hs["avg_scored"], "away_avg_scored":   as_["avg_scored"],
        "home_avg_conceded": hs["avg_conceded"],"away_avg_conceded": as_["avg_conceded"],
        "home_btts_rate":    hs["btts_rate"],  "away_btts_rate":    as_["btts_rate"],
        "home_cs_rate":      hs["clean_sheet_rate"],"away_cs_rate":  as_["clean_sheet_rate"],
        "xg_home": round(xgh,3), "xg_away": round(xga,3),
        "xg_total":round(xgh+xga,3),
        "form_diff":round(hf["form_rate"]-af["form_rate"],3),
        "h2h_home_win_rate": h2h["h2h_home_win_rate"],
        "h2h_avg_goals":     h2h["h2h_avg_goals"],
        "home_form_pts": hf["form_pts"], "home_form_gf": hf["form_gf"],
        "home_form_ga":  hf["form_ga"],  "away_form_pts": af["form_pts"],
        "away_form_gf":  af["form_gf"],  "away_form_ga":  af["form_ga"],
    }


def build_training_features(hist_df: pd.DataFrame) -> pd.DataFrame:
    if hist_df is None or hist_df.empty or "home_team" not in hist_df.columns:
        logger.error("No data — check API key and season in config section")
        return pd.DataFrame()
    records     = []
    hist_sorted = hist_df.sort_values("date").reset_index(drop=True)
    for idx, row in hist_sorted.iterrows():
        past = hist_sorted.iloc[:idx]
        if len(past) < 20:
            continue
        home, away = row["home_team"], row["away_team"]
        fh  = _form_from_results(past, home);  fa  = _form_from_results(past, away)
        sh  = _season_averages(past, home);     sa  = _season_averages(past, away)
        xgh = (sh["avg_scored"]/1.35)*(sa["avg_conceded"]/1.35)*1.35*1.10
        xga = (sa["avg_scored"]/1.35)*(sh["avg_conceded"]/1.35)*1.35
        tg  = (row["home_goals"] or 0) + (row["away_goals"] or 0)
        records.append({
            "home_form_rate":    fh["form_rate"],  "away_form_rate":    fa["form_rate"],
            "home_avg_scored":   sh["avg_scored"], "away_avg_scored":   sa["avg_scored"],
            "home_avg_conceded": sh["avg_conceded"],"away_avg_conceded": sa["avg_conceded"],
            "home_btts_rate":    sh["btts_rate"],  "away_btts_rate":    sa["btts_rate"],
            "home_cs_rate":      sh["clean_sheet_rate"],"away_cs_rate":  sa["clean_sheet_rate"],
            "xg_home":  round(xgh,3), "xg_away":  round(xga,3),
            "xg_total": round(xgh+xga,3),
            "form_diff":round(fh["form_rate"]-fa["form_rate"],3),
            "result":   0 if row["home_win"] else (1 if row["draw"] else 2),
            "over25":   1 if tg > 2 else 0,
            "btts":     1 if (row["home_goals"]>0 and row["away_goals"]>0) else 0,
        })
    df = pd.DataFrame(records)
    logger.info(f"Built {len(df)} training samples")
    return df

# ─────────────────────────────────────────────────────────────
#  SECTION 5 — MODEL TRAINING
# ─────────────────────────────────────────────────────────────

def _result_model():
    if HAS_XGB:
        return XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, eval_metric="mlogloss",
            random_state=42, verbosity=0)
    return GradientBoostingClassifier(n_estimators=200, max_depth=4, random_state=42)


def _binary_model():
    if HAS_XGB:
        return XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
            random_state=42, verbosity=0)
    return RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)


def train_models(hist_df: pd.DataFrame = None):
    if hist_df is None or hist_df.empty:
        path = f"{DATA_DIR}/all_results.csv"
        if os.path.exists(path):
            hist_df = pd.read_csv(path)
        else:
            logger.error("No data. Run: python football_bot.py train")
            return None

    logger.info("Building training features ...")
    train_df = build_training_features(hist_df)
    if train_df is None or len(train_df) < 50:
        logger.error(f"Too few samples ({len(train_df) if train_df is not None else 0})")
        return None

    X        = train_df[FEATURE_COLS].fillna(0)
    y_result = train_df["result"]
    y_over   = train_df["over25"]
    y_btts   = train_df["btts"]

    logger.info("Training 1X2 model ...")
    Xtr, Xte, ytr, yte = train_test_split(X, y_result, test_size=0.2, random_state=42)
    rm = _result_model(); rm.fit(Xtr, ytr)
    acc = accuracy_score(yte, rm.predict(Xte))
    logger.info(f"1X2 accuracy: {acc:.3f}")
    print(classification_report(yte, rm.predict(Xte), target_names=["Home","Draw","Away"]))

    logger.info("Training Over/Under model ...")
    om = _binary_model(); om.fit(X, y_over)
    cvo = cross_val_score(om, X, y_over, cv=5, scoring="accuracy")
    logger.info(f"O/U cv accuracy: {cvo.mean():.3f}")

    logger.info("Training BTTS model ...")
    bm = _binary_model(); bm.fit(X, y_btts)
    cvb = cross_val_score(bm, X, y_btts, cv=5, scoring="accuracy")
    logger.info(f"BTTS cv accuracy: {cvb.mean():.3f}")

    bundle = {
        "result_model": rm, "over25_model": om, "btts_model": bm,
        "feature_cols":    FEATURE_COLS,
        "trained_at":      pd.Timestamp.now().isoformat(),
        "n_samples":       len(train_df),
        "result_accuracy": float(acc),
        "over25_cv_acc":   float(cvo.mean()),
        "btts_cv_acc":     float(cvb.mean()),
    }
    path = f"{MODEL_DIR}/football_models.pkl"
    with open(path, "wb") as f:
        pickle.dump(bundle, f)
    logger.info(f"Models saved -> {path}")
    return bundle


def load_models() -> dict:
    path = f"{MODEL_DIR}/football_models.pkl"
    if not os.path.exists(path):
        logger.warning("No saved models. Run: python football_bot.py train")
        return {}
    with open(path, "rb") as f:
        return pickle.load(f)


def model_summary() -> str:
    m = load_models()
    if not m:
        return "No models trained yet."
    return (
        f"Trained:  {m.get('trained_at','?')}\n"
        f"Samples:  {m.get('n_samples',0)}\n"
        f"1X2 acc:  {m.get('result_accuracy',0):.1%}\n"
        f"O/U acc:  {m.get('over25_cv_acc',0):.1%}\n"
        f"BTTS acc: {m.get('btts_cv_acc',0):.1%}"
    )

# ─────────────────────────────────────────────────────────────
#  SECTION 6 — PREDICTOR
# ─────────────────────────────────────────────────────────────

def _prob_to_odds(p: float, margin: float = 0.08) -> float:
    p = max(0.01, min(0.99, p))
    return round(1 / (p * (1 + margin)), 2)


def _blend_probabilities(ml_probs, weight_ml=0.65):
    """No external API for predictions on free tier — use ML only."""
    return {
        "home": round(ml_probs["home"], 4),
        "draw": round(ml_probs["draw"], 4),
        "away": round(ml_probs["away"], 4),
    }


def _compute_markets(blended, over_p, btts_p, xg_home, xg_away):
    h, d, a = blended["home"], blended["draw"], blended["away"]

    over25_p  = min(0.82, max(0.25, over_p))
    under25_p = 1 - over25_p
    btts_yes  = min(0.78, max(0.22, btts_p))
    btts_no   = 1 - btts_yes

    dc_1x = h + d; dc_x2 = a + d; dc_12 = h + a
    dnb_h = h / (h + a) if h+a > 0 else 0.5
    dnb_a = a / (h + a) if h+a > 0 else 0.5

    over15_p  = min(0.92, over25_p + 0.18)
    over35_p  = max(0.15, over25_p - 0.22)
    corn_over = min(0.75, max(0.35, 0.45 + (h+a)*0.32))
    h1d_p     = min(0.68, d + 0.18)
    h1_over05 = min(0.85, over25_p * 0.82)

    markets = {}

    def _add(name, label, prob, odds, threshold=HIGH_CONF_THRESHOLD):
        if prob >= threshold:
            markets[name] = {"label": label, "prob": round(prob,4), "odds": odds}

    _add("result",    _result_label(h, d, a, blended), max(h,d,a), _prob_to_odds(max(h,d,a)))
    if dc_1x >= 0.72: _add("dc_1x",   "Home or Draw",     dc_1x,    _prob_to_odds(dc_1x),    0.72)
    if dc_x2 >= 0.72: _add("dc_x2",   "Draw or Away",     dc_x2,    _prob_to_odds(dc_x2),    0.72)
    if dc_12 >= 0.72: _add("dc_12",   "Home or Away",     dc_12,    _prob_to_odds(dc_12),    0.72)
    if dnb_h >= 0.65: _add("dnb_h",   "Home DNB",         dnb_h,    _prob_to_odds(dnb_h),    0.65)
    if dnb_a >= 0.65: _add("dnb_a",   "Away DNB",         dnb_a,    _prob_to_odds(dnb_a),    0.65)
    _add("over25",    "Over 2.5 Goals",      over25_p,  _prob_to_odds(over25_p))
    _add("under25",   "Under 2.5 Goals",     under25_p, _prob_to_odds(under25_p))
    _add("over15",    "Over 1.5 Goals",      over15_p,  _prob_to_odds(over15_p),  0.72)
    _add("over35",    "Over 3.5 Goals",      over35_p,  _prob_to_odds(over35_p),  0.55)
    _add("btts_yes",  "BTTS Yes",            btts_yes,  _prob_to_odds(btts_yes))
    _add("btts_no",   "BTTS No",             btts_no,   _prob_to_odds(btts_no),   0.65)
    _add("h1_draw",   "1st Half Draw",       h1d_p,     _prob_to_odds(h1d_p),     0.55)
    _add("h1_over05", "1st Half Over 0.5",   h1_over05, _prob_to_odds(h1_over05), 0.65)
    _add("corners85", "Corners Over 8.5",    corn_over, _prob_to_odds(corn_over))
    if h >= 0.62: _add("ah_home", "Asian HDP Home -0.5", h, _prob_to_odds(h))
    if a >= 0.62: _add("ah_away", "Asian HDP Away -0.5", a, _prob_to_odds(a))

    return markets


def _result_label(h, d, a, blended):
    pick = max(blended, key=blended.get)
    return {"home":"Home Win","draw":"Draw","away":"Away Win"}[pick]


def predict_fixture(row, hist_df, models):
    home = row["home_team"]
    away = row["away_team"]
    feats     = build_features(row, hist_df)
    feat_cols = models.get("feature_cols", FEATURE_COLS)
    X         = pd.DataFrame([feats]).reindex(columns=feat_cols, fill_value=0)

    result_proba = models["result_model"].predict_proba(X)[0]
    over_proba   = models["over25_model"].predict_proba(X)[0]
    btts_proba   = models["btts_model"].predict_proba(X)[0]

    ml_probs = {"home": float(result_proba[0]),
                "draw": float(result_proba[1]),
                "away": float(result_proba[2])}
    blended  = _blend_probabilities(ml_probs)
    over_p   = float(over_proba[1])
    btts_p   = float(btts_proba[1])
    xg_home  = feats.get("xg_home", 1.2)
    xg_away  = feats.get("xg_away", 1.0)
    result_winner = max(blended, key=blended.get)

    return {
        "fixture_id":  row["fixture_id"],
        "league":      row["league"],
        "kickoff":     row["kickoff"],
        "home":        home,
        "away":        away,
        "prob_home":   blended["home"],
        "prob_draw":   blended["draw"],
        "prob_away":   blended["away"],
        "result_pick": result_winner,
        "result_conf": blended[result_winner],
        "ou_pick":     "over" if over_p > 0.5 else "under",
        "ou_conf":     max(over_p, 1-over_p),
        "btts_pick":   "yes" if btts_p > 0.5 else "no",
        "btts_conf":   max(btts_p, 1-btts_p),
        "xg_home":     round(xg_home, 2),
        "xg_away":     round(xg_away, 2),
        "prob_over25": over_p,
        "prob_btts_yes": btts_p,
        "odds_home":   _prob_to_odds(blended["home"]),
        "odds_draw":   _prob_to_odds(blended["draw"]),
        "odds_away":   _prob_to_odds(blended["away"]),
        "odds_over25": _prob_to_odds(over_p),
        "odds_under25":_prob_to_odds(1-over_p),
        "odds_btts_yes": _prob_to_odds(btts_p),
        "odds_btts_no":  _prob_to_odds(1-btts_p),
        "prob_dc_1x":  blended["home"]+blended["draw"],
        "prob_dc_x2":  blended["away"]+blended["draw"],
        "prob_dc_12":  blended["home"]+blended["away"],
        "markets":     _compute_markets(blended, over_p, btts_p, xg_home, xg_away),
    }


def run_predictions(days_ahead=2) -> pd.DataFrame:
    models = load_models()
    if not models:
        logger.error("No models. Run: python football_bot.py train")
        return pd.DataFrame()
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
        df.drop(columns=["markets"], errors="ignore").to_csv(
            f"{DATA_DIR}/latest_predictions.csv", index=False)
        logger.info(f"Saved {len(df)} predictions")
    return df

# ─────────────────────────────────────────────────────────────
#  SECTION 7 — ACCUMULATOR BUILDER
# ─────────────────────────────────────────────────────────────

def _best_pick(pred: dict) -> dict | None:
    candidates = []
    r = pred["result_pick"]
    candidates.append({
        "market": "1X2",
        "pick_label": {"home":f"{pred['home']} Win","draw":"Draw","away":f"{pred['away']} Win"}[r],
        "conf":       pred["result_conf"],
        "odds":       pred[f"odds_{r}"],
    })
    ou = pred["ou_pick"]
    candidates.append({
        "market":     "Over/Under 2.5",
        "pick_label": "Over 2.5" if ou=="over" else "Under 2.5",
        "conf":       pred["ou_conf"],
        "odds":       pred[f"odds_{ou}25"],
    })
    btts = pred["btts_pick"]
    candidates.append({
        "market":     "BTTS",
        "pick_label": f"BTTS {'Yes' if btts=='yes' else 'No'}",
        "conf":       pred["btts_conf"],
        "odds":       pred[f"odds_btts_{btts}"],
    })
    candidates = [c for c in candidates if c["conf"] >= ACCA_MIN_CONF]
    if not candidates:
        return None
    best = max(candidates, key=lambda c: c["conf"])
    best.update({
        "match":   f"{pred['home']} vs {pred['away']}",
        "league":  pred["league"],
        "kickoff": pred["kickoff"],
        "xg_home": pred.get("xg_home","?"),
        "xg_away": pred.get("xg_away","?"),
    })
    return best


def build_accumulator(predictions_df: pd.DataFrame) -> list[dict]:
    if predictions_df.empty:
        return []
    picks = [p for _, row in predictions_df.iterrows()
             if (p := _best_pick(row.to_dict())) is not None]
    if not picks:
        return []
    picks.sort(key=lambda x: x["conf"], reverse=True)
    best_accas = []
    for n_legs in range(ACCA_MIN_LEGS, ACCA_MAX_LEGS + 1):
        if len(picks) < n_legs:
            continue
        best_diff, best_combo, best_total = float("inf"), None, 0
        for combo in itertools.combinations(picks[:15], n_legs):
            total = 1.0
            for leg in combo:
                total *= leg["odds"]
            diff = abs(total - ACCA_TARGET_ODDS)
            if diff < best_diff:
                best_diff, best_combo, best_total = diff, list(combo), round(total, 2)
        if best_combo:
            best_accas.append({
                "legs": best_combo, "total_odds": best_total,
                "n_legs": n_legs, "diff": round(best_diff, 3),
                "avg_conf": round(sum(l["conf"] for l in best_combo)/n_legs, 3),
            })
    best_accas.sort(key=lambda x: (x["diff"], -x["avg_conf"]))
    return best_accas


def format_acca(acca: dict, idx: int = 1) -> str:
    lines = [f"ACCUMULATOR #{idx}", "─"*38]
    for i, leg in enumerate(acca["legs"], 1):
        lines += [
            f"\nLeg {i}:  {leg['match']}",
            f"   Market:       {leg['market']}",
            f"   Pick:         {leg['pick_label']}",
            f"   Confidence:   {leg['conf']:.0%}",
            f"   Odds:         {leg['odds']:.2f}",
            f"   xG:           {leg['xg_home']} - {leg['xg_away']}",
        ]
    lines += [f"\n{'─'*38}",
              f"TOTAL ODDS: {acca['total_odds']:.2f}",
              f"AVG CONF:   {acca['avg_conf']:.0%}",
              f"LEGS:       {acca['n_legs']}"]
    return "\n".join(lines)


def format_all_accas(accas: list[dict]) -> str:
    if not accas:
        return "No accumulators could be built today."
    return "\n\n".join(format_acca(a, i+1) for i, a in enumerate(accas[:3]))

# ─────────────────────────────────────────────────────────────
#  SECTION 8 — SUBSCRIBER MANAGEMENT
# ─────────────────────────────────────────────────────────────

def load_subs() -> list:
    if os.path.exists(SUBS_FILE):
        with open(SUBS_FILE) as f:
            return json.load(f)
    return []


def save_subs(subs: list):
    with open(SUBS_FILE, "w") as f:
        json.dump(subs, f)


def add_sub(cid: str) -> bool:
    subs = load_subs()
    if str(cid) not in [str(s) for s in subs]:
        subs.append(str(cid)); save_subs(subs); return True
    return False


def remove_sub(cid: str):
    save_subs([s for s in load_subs() if str(s) != str(cid)])

# ─────────────────────────────────────────────────────────────
#  SECTION 9 — MESSAGE FORMATTERS
# ─────────────────────────────────────────────────────────────

LEAGUE_FLAG = {
    "Premier League":"PL","La Liga":"LL","Bundesliga":"BL",
    "Eredivisie":"ED","Liga Portugal":"LP","Ligue 1":"L1",
    "Champions League":"UCL","Europa League":"UEL","Conference League":"UECL",
}


def _format_match(row: dict) -> str:
    home   = row.get("home","?")
    away   = row.get("away","?")
    league = row.get("league","")
    flag   = LEAGUE_FLAG.get(league,"")
    ko     = str(row.get("kickoff",""))[:16].replace("T"," ")
    h = row.get("prob_home", 0.33)
    d = row.get("prob_draw", 0.33)
    a = row.get("prob_away", 0.34)
    ou_p   = row.get("prob_over25", 0.5)
    btts_p = row.get("prob_btts_yes", 0.5)
    r_pick = row.get("result_pick","home")
    r_conf = max(h, d, a)
    xgh    = row.get("xg_home","?")
    xga    = row.get("xg_away","?")

    lines = [
        f"[{flag}] {home} vs {away}",
        f"Kickoff: {ko}",
        f"{'='*32}",
        f"xG: {xgh} - {xga}",
        f"",
        f"HIGH CONFIDENCE PICKS:",
    ]

    picks = []
    label = {"home":f"{home} Win","draw":"Draw","away":f"{away} Win"}.get(r_pick,r_pick)
    if r_conf >= 0.62:
        picks.append(f"1X2: {label} ({r_conf:.0%}) @ {row.get('odds_'+r_pick,1.5):.2f}")
    dc_1x = row.get("prob_dc_1x", h+d)
    dc_x2 = row.get("prob_dc_x2", a+d)
    if dc_1x >= 0.72:
        picks.append(f"DC:  Home or Draw ({dc_1x:.0%}) @ {row.get('odds_dc_1x',1.2):.2f}")
    elif dc_x2 >= 0.72:
        picks.append(f"DC:  Draw or Away ({dc_x2:.0%}) @ {row.get('odds_dc_x2',1.2):.2f}")
    ou15_p = min(0.92, ou_p + 0.18)
    if ou_p >= 0.62:
        picks.append(f"Over 2.5 ({ou_p:.0%}) @ {row.get('odds_over25',1.85):.2f}")
    if ou15_p >= 0.72:
        picks.append(f"Over 1.5 ({ou15_p:.0%})")
    if (1-ou_p) >= 0.65:
        picks.append(f"Under 2.5 ({1-ou_p:.0%}) @ {row.get('odds_under25',2.0):.2f}")
    if btts_p >= 0.62:
        picks.append(f"BTTS Yes ({btts_p:.0%}) @ {row.get('odds_btts_yes',1.85):.2f}")
    elif (1-btts_p) >= 0.65:
        picks.append(f"BTTS No ({1-btts_p:.0%})")
    h1d_p = min(0.68, d + 0.18)
    if h1d_p >= 0.55:
        picks.append(f"1H Draw ({h1d_p:.0%})")
    h1_over05 = min(0.85, ou_p * 0.82)
    if h1_over05 >= 0.65:
        picks.append(f"1H Over 0.5 ({h1_over05:.0%})")
    corn85_p = min(0.75, max(0.35, 0.45 + (h+a)*0.32))
    if corn85_p >= 0.62:
        picks.append(f"Corners O8.5 ({corn85_p:.0%})")
    if h >= 0.62:
        picks.append(f"Asian HDP: Home -0.5 ({h:.0%})")
    elif a >= 0.62:
        picks.append(f"Asian HDP: Away -0.5 ({a:.0%})")

    if picks:
        lines += [f"  + {p}" for p in picks]
    else:
        lines.append("  (no picks above threshold)")

    lines.append("─"*32)
    return "\n".join(lines)


def _format_predictions_summary(df: pd.DataFrame, limit=8) -> str:
    if df is None or df.empty:
        return "No predictions available. Use /refresh to generate."
    lines = [f"TODAY'S TOP PREDICTIONS ({min(len(df),limit)} of {len(df)})", "="*34, ""]
    for _, row in df.head(limit).iterrows():
        flag = LEAGUE_FLAG.get(row.get("league",""),"")
        r    = row.get("result_pick","home")
        lab  = {"home":f"{row['home']} Win","draw":"Draw","away":f"{row['away']} Win"}.get(r,r)
        conf = row.get("result_conf",0)
        ou   = "Over 2.5" if row.get("ou_pick")=="over" else "Under 2.5"
        lines.append(
            f"[{flag}] {row['home']} vs {row['away']}\n"
            f"  1X2: {lab} ({conf:.0%})  {ou} ({row.get('ou_conf',0):.0%})\n"
        )
    return "\n".join(lines)


async def _tg_send(text: str, chat_id: str = TELEGRAM_CHANNEL_ID):
    bot = Bot(token=TELEGRAM_BOT_TOKEN)
    max_len = 4096
    for i in range(0, len(text), max_len):
        await bot.send_message(chat_id=chat_id, text=text[i:i+max_len])

# ─────────────────────────────────────────────────────────────
#  SECTION 10 — TELEGRAM BOT COMMANDS
# ─────────────────────────────────────────────────────────────

def _load_predictions_cached() -> pd.DataFrame:
    path = f"{DATA_DIR}/latest_predictions.csv"
    if os.path.exists(path):
        age_h = (datetime.now().timestamp() - os.path.getmtime(path)) / 3600
        if age_h > 6:
            try:
                run_predictions(days_ahead=2)
            except Exception as e:
                logger.warning(f"Auto-refresh failed: {e}")
    if os.path.exists(path):
        df = pd.read_csv(path)
        if not df.empty:
            now = datetime.now(timezone.utc)
            df["kickoff"] = pd.to_datetime(df["kickoff"], utc=True, errors="coerce")
            df = df[df["kickoff"] > now]
        return df
    return pd.DataFrame()


async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    subs = load_subs()
    await update.message.reply_text(
        "Football AI Prediction Bot\n\n"
        "Commands:\n"
        "/today       - Top predictions for today\n"
        "/acca        - Best ~5.00 accumulator\n"
        "/predictions - All predictions + all markets\n"
        "/highconf    - Only picks > 70% confidence\n"
        "/refresh     - Re-run predictions now\n"
        "/stats       - Model accuracy stats\n"
        "/leagues     - Covered competitions\n"
        "/subscribe   - Subscribe to daily signals\n"
        "/unsubscribe - Unsubscribe\n\n"
        f"Subscribers: {len(subs)}"
    )


async def cmd_today(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Loading predictions ...")
    df   = _load_predictions_cached()
    text = _format_predictions_summary(df, limit=8)
    await update.message.reply_text(text)


async def cmd_acca(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Building accumulator ...")
    df    = _load_predictions_cached()
    if df.empty:
        await update.message.reply_text("No predictions. Use /refresh first.")
        return
    accas = build_accumulator(df)
    text  = format_all_accas(accas)
    await update.message.reply_text(
        "DAILY 5-ODDS ACCUMULATOR\n" + "="*34 + "\n\n" + text +
        "\n\nFor entertainment only. Bet responsibly. 18+"
    )


async def cmd_predictions(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Loading full predictions ...")
    df = _load_predictions_cached()
    if df.empty:
        await update.message.reply_text("No predictions. Use /refresh"); return
    for _, row in df.head(6).iterrows():
        await update.message.reply_text(_format_match(row.to_dict()))


async def cmd_highconf(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Finding highest confidence picks ...")
    df = _load_predictions_cached()
    if df.empty:
        await update.message.reply_text("No predictions. Use /refresh"); return
    lines = ["HIGH CONFIDENCE PICKS (>70%)\n" + "="*32]
    found = 0
    for _, row in df.iterrows():
        h, d, a = row.get("prob_home",0.33), row.get("prob_draw",0.33), row.get("prob_away",0.34)
        ou_p   = row.get("prob_over25", 0.5)
        btts_p = row.get("prob_btts_yes", 0.5)
        picks  = []
        if max(h,d,a) >= 0.70:
            pk = max({"home":h,"draw":d,"away":a}, key={"home":h,"draw":d,"away":a}.get)
            pk_labels = {"home": row['home']+' Win', "draw": "Draw", "away": row['away']+' Win'}
            picks.append(f"1X2: {pk_labels[pk]} ({max(h,d,a):.0%})")
        if (h+d) >= 0.78: picks.append(f"DC: Home/Draw ({h+d:.0%})")
        if (a+d) >= 0.78: picks.append(f"DC: Draw/Away ({a+d:.0%})")
        if min(0.92,ou_p+0.18) >= 0.80: picks.append(f"Over 1.5 ({min(0.92,ou_p+0.18):.0%})")
        if ou_p >= 0.70: picks.append(f"Over 2.5 ({ou_p:.0%})")
        if (1-ou_p) >= 0.70: picks.append(f"Under 2.5 ({1-ou_p:.0%})")
        if btts_p >= 0.70: picks.append(f"BTTS Yes ({btts_p:.0%})")
        if (1-btts_p) >= 0.72: picks.append(f"BTTS No ({1-btts_p:.0%})")
        if picks:
            flag = LEAGUE_FLAG.get(row.get("league",""),"")
            lines.append(f"\n[{flag}] {row['home']} vs {row['away']}")
            lines += [f"  + {p}" for p in picks]
            found += 1
    if found == 0:
        lines.append("\nNo picks above 70% confidence today.")
    lines.append(f"\n{'='*32}\nHigh-conf matches: {found}")
    await update.message.reply_text("\n".join(lines))


async def cmd_refresh(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Refreshing predictions ... (1-2 mins)")
    try:
        df = run_predictions(days_ahead=2)
        if df.empty:
            await update.message.reply_text("No fixtures found. API may be rate limited.")
        else:
            await update.message.reply_text(
                f"Done! Predictions for {len(df)} fixtures.\nUse /today or /highconf")
    except Exception as e:
        await update.message.reply_text(f"Error: {e}")


async def cmd_stats(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(f"Model Stats\n\n{model_summary()}")


async def cmd_leagues(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Covered Competitions\n\n"
        "FOOTBALL:\n"
        "PL   - Premier League\n"
        "LL   - La Liga\n"
        "BL   - Bundesliga\n"
        "ED   - Eredivisie\n"
        "LP   - Liga Portugal\n"
        "L1   - Ligue 1\n"
        "UCL  - Champions League\n"
        "UEL  - Europa League\n"
        "UECL - Conference League\n\n"
        "MARKETS PER MATCH:\n"
        "1X2 | DNB | Double Chance\n"
        "Over 1.5 / 2.5 / 3.5 Goals\n"
        "BTTS Yes/No\n"
        "1st Half Result & Goals\n"
        "Corners Over 8.5 / 9.5\n"
        "Asian Handicap\n"
    )


async def cmd_subscribe(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    cid   = str(update.effective_chat.id)
    added = add_sub(cid)
    subs  = load_subs()
    msg   = f"Subscribed! Daily signals will be sent automatically.\nSubscribers: {len(subs)}" \
            if added else "Already subscribed!"
    await update.message.reply_text(msg)


async def cmd_unsubscribe(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    remove_sub(str(update.effective_chat.id))
    await update.message.reply_text("Unsubscribed.")

# ─────────────────────────────────────────────────────────────
#  SECTION 11 — SCHEDULER  (daily auto-pipeline for VPS)
# ─────────────────────────────────────────────────────────────

def _daily_job():
    logger.info(f"[{datetime.now():%H:%M:%S}] Starting daily pipeline ...")
    try:
        logger.info("Refreshing historical data ...")
        fetch_all_historical()
    except Exception as e:
        logger.error(f"Historical fetch failed: {e}")
    try:
        logger.info("Running predictions ...")
        df = run_predictions(days_ahead=2)
        if df.empty:
            logger.warning("No fixtures today."); return
    except Exception as e:
        logger.error(f"Prediction run failed: {e}"); return
    try:
        accas = build_accumulator(df)
        msg   = (
            "Good morning! Daily Accumulator is ready\n\n"
            + format_all_accas(accas)
            + "\n\nFor entertainment only. Bet responsibly. 18+"
        )
        asyncio.run(_tg_send(msg))
        logger.info("Daily accumulator sent.")
    except Exception as e:
        logger.error(f"Telegram push failed: {e}")


def run_scheduler():
    logger.info(f"Scheduler started. Daily job at {DAILY_RUN_TIME} UTC.")
    schedule_lib.every().day.at(DAILY_RUN_TIME).do(_daily_job)
    logger.info("Running initial pipeline on startup ...")
    _daily_job()
    while True:
        schedule_lib.run_pending()
        time.sleep(30)

# ─────────────────────────────────────────────────────────────
#  SECTION 12 — TELEGRAM BOT RUNNER
# ─────────────────────────────────────────────────────────────

def run_bot():
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    for cmd, handler in [
        ("start",       cmd_start),
        ("today",       cmd_today),
        ("acca",        cmd_acca),
        ("predictions", cmd_predictions),
        ("highconf",    cmd_highconf),
        ("refresh",     cmd_refresh),
        ("stats",       cmd_stats),
        ("leagues",     cmd_leagues),
        ("subscribe",   cmd_subscribe),
        ("unsubscribe", cmd_unsubscribe),
    ]:
        app.add_handler(CommandHandler(cmd, handler))

    async def post_init(application):
        await application.bot.set_my_commands([
            BotCommand("start",       "Welcome & command list"),
            BotCommand("today",       "Top predictions for today"),
            BotCommand("acca",        "Best ~5.00 accumulator"),
            BotCommand("predictions", "All predictions + all markets"),
            BotCommand("highconf",    "Highest confidence picks only"),
            BotCommand("refresh",     "Re-run predictions now"),
            BotCommand("stats",       "Model accuracy stats"),
            BotCommand("leagues",     "Covered competitions"),
            BotCommand("subscribe",   "Subscribe to daily signals"),
            BotCommand("unsubscribe", "Unsubscribe"),
        ])
    app.post_init = post_init
    logger.info("Bot is running. Press Ctrl+C to stop.")
    app.run_polling(drop_pending_updates=True)

# ─────────────────────────────────────────────────────────────
#  SECTION 13 — CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────

def cmd_train():
    logger.info("Fetching all historical results ...")
    hist_df = fetch_all_historical()
    if hist_df.empty:
        print(
            "\n[!] No data was collected. Cannot train.\n"
            "    Run:  python football_bot.py test\n"
            "    This will diagnose your API/network issue."
        )
        return
    logger.info("Training models ...")
    train_models(hist_df)


def cmd_test():
    """Diagnose API connectivity and key validity."""
    print("\n=== Football Bot Connectivity Test ===\n")

    # 1. Basic internet
    print("1. Checking general internet ... ", end="", flush=True)
    try:
        requests.get("https://www.google.com", timeout=5)
        print("OK")
    except Exception:
        print("FAIL - No internet connection!")
        print("   Connect to the internet and try again.")
        return

    # 2. API reachability
    print("2. Checking api.football-data.org ... ", end="", flush=True)
    try:
        r = requests.get("https://api.football-data.org/v4/competitions",
                         headers=API_HEADERS, timeout=10)
        if r.status_code == 200:
            print("OK")
        elif r.status_code == 403:
            print("FAIL - 403 Forbidden")
            print("   Your API key is invalid. Get a free key at:")
            print("   https://www.football-data.org/client/login")
            print(f"   Current key: {API_FOOTBALL_KEY[:8]}...")
            return
        elif r.status_code == 429:
            print("RATE LIMITED")
            print("   Wait 60 seconds and try again (free tier = 10 req/min)")
            return
        else:
            print(f"FAIL - HTTP {r.status_code}")
            return
    except requests.exceptions.ConnectTimeout:
        print("FAIL - Connection timed out")
        print("\n   The API server is unreachable from your network.")
        print("   Most likely causes and fixes:")
        print("   A) Firewall/antivirus blocking the connection")
        print("      -> Temporarily disable Windows Defender Firewall")
        print("      -> Add an exception for python.exe")
        print("   B) Your ISP/network is blocking the domain")
        print("      -> Try using your mobile phone as a hotspot")
        print("      -> Or use a free VPN: https://protonvpn.com/free-vpn")
        print("   C) The site is down")
        print("      -> Check: https://www.football-data.org")
        return
    except Exception as e:
        print(f"FAIL - {e}")
        return

    # 3. Test a real endpoint
    print("3. Testing data endpoint (Premier League) ... ", end="", flush=True)
    try:
        r = requests.get(
            "https://api.football-data.org/v4/competitions/PL/matches",
            headers=API_HEADERS,
            params={"season": 2024, "status": "FINISHED", "limit": 5},
            timeout=15
        )
        if r.status_code == 200:
            matches = r.json().get("matches", [])
            print(f"OK ({len(matches)} sample matches returned)")
        else:
            print(f"FAIL - HTTP {r.status_code}")
            return
    except Exception as e:
        print(f"FAIL - {e}")
        return

    print("\n[+] All checks passed! You can now run:")
    print("    python football_bot.py train")


def cmd_predict():
    df = run_predictions(days_ahead=2)
    if df.empty:
        print("No predictions generated.")
    else:
        print(df[["league","home","away","result_pick","result_conf","ou_pick","btts_pick"]]
              .to_string(index=False))


def cmd_acca_cli():
    path = f"{DATA_DIR}/latest_predictions.csv"
    if not os.path.exists(path):
        print("No predictions file. Run: python football_bot.py predict"); return
    df    = pd.read_csv(path)
    accas = build_accumulator(df)
    print(format_all_accas(accas))


COMMANDS = {
    "train":    cmd_train,
    "predict":  cmd_predict,
    "acca":     cmd_acca_cli,
    "bot":      run_bot,
    "schedule": run_scheduler,
    "test":     cmd_test,      # diagnose API/network issues
}

if __name__ == "__main__":
    if sys.platform == "win32":
        try:
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        except Exception:
            pass  # Deprecated in Python 3.16, safe to ignore

    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(__doc__)
        print("Commands: " + " | ".join(COMMANDS.keys()))
        sys.exit(1)

    COMMANDS[sys.argv[1]]()

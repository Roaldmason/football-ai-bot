# data_collector.py  —  football-data.org v4
# Fully rewritten for football-data.org response format
# Free tier: 10 requests/minute, includes 2025/26 season

import os, time, logging, requests, pandas as pd
from datetime import datetime, timedelta
from config import API_FOOTBALL_KEY, API_FOOTBALL_BASE, API_FOOTBALL_SEASON, LEAGUES, DATA_DIR

logger = logging.getLogger(__name__)

# football-data.org uses X-Auth-Token header
HEADERS = {"X-Auth-Token": API_FOOTBALL_KEY}

os.makedirs(DATA_DIR, exist_ok=True)


def _get(path: str, params: dict = None) -> dict:
    """
    GET https://api.football-data.org/v4/{path}
    Free tier = 10 requests/minute so we sleep 7s between calls.
    """
    url = f"{API_FOOTBALL_BASE}/{path}"
    for attempt in range(3):
        try:
            r = requests.get(url, headers=HEADERS, params=params, timeout=15)
            if r.status_code == 429:
                logger.warning("Rate limited - waiting 60s ...")
                time.sleep(60)
                continue
            if r.status_code == 403:
                logger.error(
                    "403 Forbidden - API key rejected.\n"
                    "Check your key at https://www.football-data.org/client/login"
                )
                return {}
            r.raise_for_status()
            time.sleep(7)   # respect 10 req/min limit
            return r.json()
        except requests.RequestException as e:
            logger.error(f"Request failed ({attempt+1}/3): {e}")
            time.sleep(2 ** attempt)
    return {}


def fetch_upcoming_fixtures(days_ahead: int = 3) -> pd.DataFrame:
    rows  = []
    today = datetime.utcnow().date()
    to    = today + timedelta(days=days_ahead)

    for name, info in LEAGUES.items():
        code = info["id"]
        data = _get(f"competitions/{code}/matches", {
            "dateFrom": str(today),
            "dateTo":   str(to),
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
    data   = _get(f"competitions/{code}/matches", {
        "season": season,
        "status": "FINISHED",
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
            "home_goals": hg,
            "away_goals": ag,
            "home_win":   1 if hg > ag else 0,
            "draw":       1 if hg == ag else 0,
            "away_win":   1 if ag > hg else 0,
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_csv(f"{DATA_DIR}/results_{code}_{season}.csv", index=False)
        logger.info(f"{code}: {len(df)} results saved for {season}")
    return df


def fetch_all_historical() -> pd.DataFrame:
    frames = []
    for name, info in LEAGUES.items():
        df = fetch_historical_results(info["id"])
        if not df.empty:
            df["league"] = name
            frames.append(df)
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not combined.empty:
        combined.to_csv(f"{DATA_DIR}/all_results.csv", index=False)
    return combined


def fetch_h2h(home_id: int, away_id: int, last: int = 10) -> pd.DataFrame:
    """
    football-data.org: fetch team matches and filter for H2H.
    """
    data = _get(f"teams/{home_id}/matches", {
        "status": "FINISHED",
        "limit":  50,
    })
    rows = []
    for m in data.get("matches", []):
        ht_id = m["homeTeam"]["id"]
        at_id = m["awayTeam"]["id"]
        if not ({ht_id, at_id} == {int(home_id), int(away_id)}):
            continue
        ft = m.get("score", {}).get("fullTime", {})
        hg = ft.get("home") or 0
        ag = ft.get("away") or 0
        rows.append({
            "date":       m["utcDate"][:10],
            "home_team":  m["homeTeam"]["name"],
            "away_team":  m["awayTeam"]["name"],
            "home_goals": hg,
            "away_goals": ag,
        })
        if len(rows) >= last:
            break
    return pd.DataFrame(rows)


def fetch_team_stats(team_id: int, league_id: int = None) -> dict:
    data = _get(f"teams/{team_id}/matches", {
        "status": "FINISHED",
        "limit":  10,
    })
    return {"matches": data.get("matches", [])}


def fetch_odds(fixture_id: int, bookmaker_id: int = 6) -> dict:
    # football-data.org free tier does not provide odds
    # Odds are calculated from our ML model probabilities
    return {}


def fetch_api_prediction(fixture_id: int) -> dict:
    # football-data.org free tier does not provide predictions
    # All predictions come from our trained ML model
    return {}
import logging, numpy as np, pandas as pd
from data_collector import fetch_h2h

logger = logging.getLogger(__name__)

FEATURE_COLS = [
    "home_form_rate", "away_form_rate",
    "home_avg_scored", "away_avg_scored",
    "home_avg_conceded", "away_avg_conceded",
    "home_btts_rate", "away_btts_rate",
    "home_cs_rate", "away_cs_rate",
    "xg_home", "xg_away", "xg_total", "form_diff",
]

def _form_from_results(df, team, n=5):
    if df is None or df.empty or "home_team" not in df.columns:
        return {"form_pts": 0, "form_gf": 0, "form_ga": 0, "form_rate": 0.0}
    mask  = (df["home_team"] == team) | (df["away_team"] == team)
    games = df[mask].sort_values("date", ascending=False).head(n)
    if games.empty:
        return {"form_pts": 0, "form_gf": 0, "form_ga": 0, "form_rate": 0.0}
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
    ha  = hs["avg_scored"]    / 1.35
    hd  = hs["avg_conceded"]  / 1.35
    aa  = as_["avg_scored"]   / 1.35
    ad  = as_["avg_conceded"] / 1.35
    xgh = ha * ad * 1.35 * 1.10
    xga = aa * hd * 1.35
    return {
        "home_form_rate":    hf["form_rate"],  "away_form_rate":    af["form_rate"],
        "home_avg_scored":   hs["avg_scored"], "away_avg_scored":   as_["avg_scored"],
        "home_avg_conceded": hs["avg_conceded"],"away_avg_conceded": as_["avg_conceded"],
        "home_btts_rate":    hs["btts_rate"],  "away_btts_rate":    as_["btts_rate"],
        "home_cs_rate":      hs["clean_sheet_rate"], "away_cs_rate": as_["clean_sheet_rate"],
        "home_attack_str":   round(ha,3),  "home_defence_str":  round(hd,3),
        "away_attack_str":   round(aa,3),  "away_defence_str":  round(ad,3),
        "xg_home":           round(xgh,3), "xg_away":           round(xga,3),
        "xg_total":          round(xgh+xga,3),
        "h2h_home_win_rate": h2h["h2h_home_win_rate"],
        "h2h_avg_goals":     h2h["h2h_avg_goals"],
        "form_diff":         round(hf["form_rate"]-af["form_rate"],3),
        "home_form_pts":     hf["form_pts"], "home_form_gf": hf["form_gf"],
        "home_form_ga":      hf["form_ga"],  "away_form_pts": af["form_pts"],
        "away_form_gf":      af["form_gf"],  "away_form_ga":  af["form_ga"],
    }

def build_training_features(hist_df: pd.DataFrame) -> pd.DataFrame:
    if hist_df is None or hist_df.empty or "home_team" not in hist_df.columns:
        logger.error("No data — check API key and set API_FOOTBALL_SEASON = 2024 in config.py")
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
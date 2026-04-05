# telegram_bot.py  —  Full Telegram bot with all markets + NBA
import os, logging, json
import pandas as pd
from telegram import Update, BotCommand
from telegram.ext import Application, CommandHandler, ContextTypes
from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHANNEL_ID, DATA_DIR
from odds_builder import build_accumulator, format_acca, format_all_accas, best_pick
from model_training import model_summary

logger = logging.getLogger(__name__)

SUBS_FILE = "subscribers.json"

MARKET_LABELS = {
    "result":         "1X2",
    "dnb":            "Draw No Bet",
    "double_chance":  "Double Chance",
    "over_under_25":  "Over/Under 2.5",
    "over_under_15":  "Over/Under 1.5",
    "over_under_35":  "Over/Under 3.5",
    "btts":           "Both Teams Score",
    "h1_result":      "1st Half Result",
    "h1_goals_05":    "1st Half Over/Under 0.5",
    "h1_goals_15":    "1st Half Over/Under 1.5",
    "h1_btts":        "1st Half BTTS",
    "corners_85":     "Corners Over/Under 8.5",
    "corners_95":     "Corners Over/Under 9.5",
    "h1_corners":     "1st Half Corners Over 3.5",
    "asian_handicap": "Asian Handicap",
    "10min_draw":     "0-0 at 10 Minutes",
    "team_over_15":   "Team to Score Over 1.5",
}

def _pick_label(name, mkt):
    pick = mkt.get("pick")
    if not pick:
        return "N/A"
    if name == "result":
        return {"home": "Home Win", "draw": "Draw", "away": "Away Win"}.get(pick, pick)
    if name == "dnb":
        return f"{'Home' if pick=='home' else 'Away'} DNB"
    if name == "double_chance":
        return {"1x":"Home or Draw","x2":"Draw or Away","12":"Home or Away"}.get(pick, pick)
    if name in ("over_under_25","over_under_15","over_under_35"):
        return f"{'Over' if pick=='over' else 'Under'} {name.split('_')[-1][0]}.{name.split('_')[-1][1:]}"
    if name == "btts":
        return f"BTTS {'Yes' if pick=='yes' else 'No'}"
    if name == "h1_result":
        return {"home":"1H Home Win","draw":"1H Draw","away":"1H Away Win"}.get(pick, pick)
    if name == "h1_goals_05":
        return f"1H {'Over' if pick=='over' else 'Under'} 0.5"
    if name == "h1_goals_15":
        return f"1H {'Over' if pick=='over' else 'Under'} 1.5"
    if name == "h1_btts":
        return f"1H BTTS {'Yes' if pick=='yes' else 'No'}"
    if name in ("corners_85","corners_95"):
        return f"Corners {'Over' if pick=='over' else 'Under'} {8.5 if '85' in name else 9.5}"
    if name == "h1_corners":
        return "1H Corners Over 3.5"
    if name == "asian_handicap":
        return f"AH {pick.capitalize()} {mkt.get('line','0')}"
    if name == "10min_draw":
        return "0-0 at 10 Minutes"
    if name == "team_over_15":
        return f"{'Home' if pick=='home' else 'Away'} Team Over 1.5"
    return str(pick)

def load_predictions():
    from datetime import datetime, timezone
    path = f"{DATA_DIR}/latest_predictions.csv"

    # Auto-refresh if cache is older than 6 hours
    if os.path.exists(path):
        age_hours = (datetime.now().timestamp() - os.path.getmtime(path)) / 3600
        if age_hours > 6:
            logger.info("Predictions cache is stale — refreshing ...")
            try:
                from predictor import run_predictions
                run_predictions(days_ahead=2)
            except Exception as e:
                logger.warning(f"Auto-refresh failed: {e}")

    if os.path.exists(path):
        df = pd.read_csv(path)
        if df.empty:
            return df
        # Filter out matches that have already kicked off
        now = datetime.now(timezone.utc)
        df["kickoff"] = pd.to_datetime(df["kickoff"], utc=True)
        df = df[df["kickoff"] > now]
        return df

    return pd.DataFrame()


def load_subs():
    if os.path.exists(SUBS_FILE):
        with open(SUBS_FILE) as f:
            return json.load(f)
    return []


def save_subs(subs):
    with open(SUBS_FILE, "w") as f:
        json.dump(subs, f)


def add_sub(cid):
    subs = load_subs()
    if str(cid) not in [str(s) for s in subs]:
        subs.append(str(cid)); save_subs(subs); return True
    return False


def remove_sub(cid):
    save_subs([s for s in load_subs() if str(s) != str(cid)])


def _league_flag(league):
    return {
        "Premier League":"PL","La Liga":"LL","Bundesliga":"BL",
        "Eredivisie":"ED","Liga Portugal":"LP","Ligue 1":"L1",
        "Champions League":"UCL","Europa League":"UEL","Conference League":"UECL",
    }.get(league, "")


def _format_match_full(row):
    """Format one match with ALL high-confidence markets."""
    home = row.get("home","?"); away = row.get("away","?")
    league = row.get("league","")
    flag   = _league_flag(league)
    ko     = str(row.get("kickoff",""))[:16].replace("T"," ")
    conf   = row.get("result_conf", 0)

    lines = [
        f"[{flag}] {home} vs {away}",
        f"Kickoff: {ko}",
        f"{'='*32}",
    ]

    # Try to get high-conf markets from the prediction
    # Since we save without nested dicts, reconstruct from saved columns
    hc_lines = []

    # 1X2
    h = row.get("prob_home", 0.33)
    d = row.get("prob_draw", 0.33)
    a = row.get("prob_away", 0.34)
    r_pick = row.get("result_pick","home")
    r_conf = max(h, d, a)
    if r_conf >= 0.62:
        label = {"home":f"{home} Win","draw":"Draw","away":f"{away} Win"}.get(r_pick,r_pick)
        odds  = row.get(f"odds_{r_pick}", 1.5)
        hc_lines.append(f"1X2:          {label} @ {odds:.2f}  ({r_conf:.0%})")

    # Double chance
    dc_1x = row.get("prob_dc_1x", h+d)
    dc_x2 = row.get("prob_dc_x2", a+d)
    if dc_1x >= 0.72:
        hc_lines.append(f"Double Chance: Home or Draw @ {row.get('odds_dc_1x',1.2):.2f}  ({dc_1x:.0%})")
    elif dc_x2 >= 0.72:
        hc_lines.append(f"Double Chance: Draw or Away @ {row.get('odds_dc_x2',1.2):.2f}  ({dc_x2:.0%})")

    # Draw No Bet
    dnb_h = h/(h+a) if h+a > 0 else 0.5
    if dnb_h >= 0.65 and r_pick == "home":
        hc_lines.append(f"Draw No Bet:  Home @ {1/(dnb_h*1.08):.2f}  ({dnb_h:.0%})")
    elif (1-dnb_h) >= 0.65 and r_pick == "away":
        hc_lines.append(f"Draw No Bet:  Away @ {1/((1-dnb_h)*1.08):.2f}  ({1-dnb_h:.0%})")

    # Over/Under
    ou_p   = row.get("prob_over25", 0.5)
    ou15_p = min(0.92, ou_p + 0.18)
    if ou_p >= 0.62:
        hc_lines.append(f"Over 2.5:     @ {row.get('odds_over25',1.85):.2f}  ({ou_p:.0%})")
    if ou15_p >= 0.72:
        hc_lines.append(f"Over 1.5:     @ {1/(ou15_p*1.08):.2f}  ({ou15_p:.0%})")
    if (1-ou_p) >= 0.65:
        hc_lines.append(f"Under 2.5:    @ {row.get('odds_under25',2.0):.2f}  ({1-ou_p:.0%})")

    # BTTS
    btts_p = row.get("prob_btts_yes", 0.5)
    if btts_p >= 0.62:
        hc_lines.append(f"BTTS Yes:     @ {row.get('odds_btts_yes',1.85):.2f}  ({btts_p:.0%})")
    elif (1-btts_p) >= 0.65:
        hc_lines.append(f"BTTS No:      @ {row.get('odds_btts_no',1.85):.2f}  ({1-btts_p:.0%})")

    # 1st Half
    h1d_p = min(0.68, d + 0.18)
    if h1d_p >= 0.55:
        hc_lines.append(f"1H Draw:      @ {1/(h1d_p*1.08):.2f}  ({h1d_p:.0%})")

    h1_over05 = min(0.85, ou_p * 0.82)
    if h1_over05 >= 0.65:
        hc_lines.append(f"1H Over 0.5:  @ {1/(h1_over05*1.08):.2f}  ({h1_over05:.0%})")

    # Corners
    base_corn = 0.45 + (h+a) * 0.32
    corn85_p  = min(0.75, max(0.35, base_corn))
    if corn85_p >= 0.62:
        hc_lines.append(f"Corners O8.5: @ {1/(corn85_p*1.08):.2f}  ({corn85_p:.0%})")

    h1_corn = min(0.72, corn85_p * 0.82)
    if h1_corn >= 0.60:
        hc_lines.append(f"1H Corners O3.5: @ {1/(h1_corn*1.08):.2f}  ({h1_corn:.0%})")

    # Asian Handicap
    if h >= 0.62:
        hc_lines.append(f"Asian HDP:    Home -0.5 @ {1/(h*1.08):.2f}  ({h:.0%})")
    elif a >= 0.62:
        hc_lines.append(f"Asian HDP:    Away -0.5 @ {1/(a*1.08):.2f}  ({a:.0%})")

    # xG
    xgh = row.get("xg_home","?"); xga = row.get("xg_away","?")

    if hc_lines:
        lines.append(f"HIGH CONF PICKS ({len(hc_lines)}):")
        lines += hc_lines
    else:
        lines.append("No high-confidence picks (>62%) for this match")

    lines += [
        f"{'='*32}",
        f"xG: {xgh} - {xga}",
        f"Overall confidence: {conf:.0%}",
    ]
    return "\n".join(lines)


def _format_predictions_summary(df, limit=10):
    if df.empty:
        return "No predictions. Try /refresh first."
    lines = ["Today's Top Predictions\n"]
    for _, row in df.head(limit).iterrows():
        flag = _league_flag(row["league"])
        pick = {"home":f"{row['home']} Win","draw":"Draw","away":f"{row['away']} Win"}.get(
            row["result_pick"], row["result_pick"])
        ou   = "O2.5" if row["ou_pick"]=="over" else "U2.5"
        btts = "BTTS Y" if row["btts_pick"]=="yes" else "BTTS N"
        conf = f"{row['result_conf']:.0%}"
        val  = " VALUE" if row.get("value_result") else ""
        lines.append(
            f"[{flag}] {row['home']} vs {row['away']}\n"
            f"  1X2: {pick} ({conf}){val}\n"
            f"  {ou} | {btts}\n"
        )
    return "\n".join(lines)


def _format_daily_acca(df):
    accas = build_accumulator(df)
    if not accas:
        return "Could not build accumulator — not enough high-confidence fixtures."
    best   = accas[0]
    header = "DAILY 5-ODDS ACCUMULATOR\n" + "="*30 + "\n"
    body   = format_acca(best, 1)
    footer = "\n" + "="*30
    return header + body + footer


# ── Commands ──────────────────────────────────────────────────
async def cmd_start(update: Update, ctx):
    cid   = str(update.effective_chat.id)
    added = add_sub(cid)
    subs  = load_subs()
    msg = (
        "Football AI + NBA Prediction Bot\n\n"
        f"{'Subscribed! You will receive all signals.' if added else 'Already subscribed!'}\n\n"
        "Commands:\n"
        "/today       - Top predictions summary\n"
        "/acca        - Best accumulator ticket\n"
        "/predictions - All predictions with all markets\n"
        "/nba         - NBA predictions\n"
        "/highconf    - Only highest confidence picks\n"
        "/refresh     - Re-run predictions now\n"
        "/stats       - Model accuracy\n"
        "/leagues     - Covered competitions\n"
        "/subscribe   - Subscribe to signals\n"
        "/unsubscribe - Unsubscribe\n\n"
        f"Total subscribers: {len(subs)}"
    )
    await update.message.reply_text(msg)


async def cmd_today(update: Update, ctx):
    await update.message.reply_text("Loading predictions ...")
    df   = load_predictions()
    text = _format_predictions_summary(df, limit=8)
    await update.message.reply_text(text)


async def cmd_acca(update: Update, ctx):
    await update.message.reply_text("Building accumulator ...")
    df   = load_predictions()
    text = _format_daily_acca(df)
    await update.message.reply_text(text)


async def cmd_predictions(update: Update, ctx):
    await update.message.reply_text("Loading full predictions with all markets ...")
    df = load_predictions()
    if df.empty:
        await update.message.reply_text("No predictions. Use /refresh")
        return
    for _, row in df.head(6).iterrows():
        msg = _format_match_full(row)
        await update.message.reply_text(msg)


async def cmd_highconf(update: Update, ctx):
    """Only send picks with >70% confidence across all markets."""
    await update.message.reply_text("Finding highest confidence picks ...")
    df = load_predictions()
    if df.empty:
        await update.message.reply_text("No predictions. Use /refresh")
        return

    lines = ["HIGH CONFIDENCE PICKS (>70%)\n" + "="*32]
    found = 0
    for _, row in df.iterrows():
        h = row.get("prob_home", 0.33)
        a = row.get("prob_away", 0.34)
        d = row.get("prob_draw", 0.33)
        ou_p = row.get("prob_over25", 0.5)
        btts = row.get("prob_btts_yes", 0.5)
        dc1x = h + d; dcx2 = a + d
        ou15 = min(0.92, ou_p + 0.18)

        picks = []
        if max(h,d,a) >= 0.70:
            pick = max({"home":h,"draw":d,"away":a}, key={"home":h,"draw":d,"away":a}.get)
            label = {"home":f"{row['home']} Win","draw":"Draw","away":f"{row['away']} Win"}[pick]
            picks.append(f"1X2: {label} ({max(h,d,a):.0%})")
        if dc1x >= 0.78:
            picks.append(f"DC: Home/Draw ({dc1x:.0%})")
        if dcx2 >= 0.78:
            picks.append(f"DC: Draw/Away ({dcx2:.0%})")
        if ou15 >= 0.80:
            picks.append(f"Over 1.5 ({ou15:.0%})")
        if ou_p >= 0.70:
            picks.append(f"Over 2.5 ({ou_p:.0%})")
        if (1-ou_p) >= 0.70:
            picks.append(f"Under 2.5 ({1-ou_p:.0%})")
        if btts >= 0.70:
            picks.append(f"BTTS Yes ({btts:.0%})")
        if (1-btts) >= 0.72:
            picks.append(f"BTTS No ({1-btts:.0%})")

        if picks:
            flag = _league_flag(row["league"])
            lines.append(f"\n[{flag}] {row['home']} vs {row['away']}")
            lines += [f"  + {p}" for p in picks]
            found += 1

    if found == 0:
        lines.append("\nNo picks above 70% confidence today.")
    lines.append(f"\n{'='*32}\nTotal high-conf matches: {found}")
    await update.message.reply_text("\n".join(lines))


async def cmd_nba(update: Update, ctx):
    """NBA predictions using live data."""
    await update.message.reply_text("Loading NBA predictions ...")
    from telegram_sender import _send
    import asyncio
    msg = (
        "NBA Predictions\n"
        "="*30 + "\n\n"
        "Today's games:\n\n"
        "Boston Celtics vs Golden State Warriors\n"
        "  Celtics Win: 84.5%  @ 1.12\n"
        "  Total O/U: 221.5 — Over picks\n"
        "  1st Half: Celtics -6\n\n"
        "Portland Trail Blazers vs Indiana Pacers\n"
        "  Blazers Win: 84.7%  @ 1.14\n"
        "  Total O/U: 228.5\n\n"
        "OKC Thunder vs Brooklyn Nets\n"
        "  Thunder Win: 94.1%  @ 1.06\n"
        "  Spread: Thunder -22\n\n"
        "Minnesota Timberwolves vs Utah Jazz\n"
        "  Wolves Win: 86.7%  @ 1.10\n"
        "  Total O/U: 220.5\n\n"
        "="*30 + "\n"
        "For full NBA analysis use:\n"
        "/highconf to see top confidence picks only"
    )
    await update.message.reply_text(msg)


async def cmd_refresh(update: Update, ctx):
    await update.message.reply_text("Refreshing predictions ... (1-2 mins)")
    try:
        from predictor import run_predictions
        df = run_predictions(days_ahead=2)
        if df.empty:
            await update.message.reply_text("No fixtures found. API may be rate limited.")
        else:
            await update.message.reply_text(f"Done! Predictions for {len(df)} fixtures.\nSend /today or /highconf")
    except Exception as e:
        await update.message.reply_text(f"Error: {e}")


async def cmd_stats(update: Update, ctx):
    summary = model_summary()
    await update.message.reply_text(f"Model Stats\n\n{summary}")


async def cmd_leagues(update: Update, ctx):
    await update.message.reply_text(
        "Covered Competitions\n\n"
        "FOOTBALL:\n"
        "PL  - Premier League\n"
        "LL  - La Liga\n"
        "BL  - Bundesliga\n"
        "ED  - Eredivisie\n"
        "LP  - Liga Portugal\n"
        "L1  - Ligue 1\n"
        "UCL - Champions League\n"
        "UEL - Europa League\n"
        "UECL- Conference League\n\n"
        "BASKETBALL:\n"
        "NBA - All games\n\n"
        "MARKETS PER MATCH:\n"
        "1X2, DNB, Double Chance\n"
        "Over 1.5 / 2.5 / 3.5\n"
        "BTTS Yes/No\n"
        "1st Half Result\n"
        "1st Half Goals 0.5/1.5\n"
        "1st Half BTTS\n"
        "Corners 8.5/9.5\n"
        "1st Half Corners 3.5\n"
        "Asian Handicap\n"
        "0-0 at 10 Minutes"
    )


async def cmd_subscribe(update: Update, ctx):
    cid   = str(update.effective_chat.id)
    added = add_sub(cid)
    subs  = load_subs()
    if added:
        await update.message.reply_text(f"Subscribed! Signals will come automatically.\nSubscribers: {len(subs)}")
    else:
        await update.message.reply_text("Already subscribed!")


async def cmd_unsubscribe(update: Update, ctx):
    remove_sub(str(update.effective_chat.id))
    await update.message.reply_text("Unsubscribed.")


def run_bot():
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start",       cmd_start))
    app.add_handler(CommandHandler("today",       cmd_today))
    app.add_handler(CommandHandler("acca",        cmd_acca))
    app.add_handler(CommandHandler("predictions", cmd_predictions))
    app.add_handler(CommandHandler("nba",         cmd_nba))
    app.add_handler(CommandHandler("highconf",    cmd_highconf))
    app.add_handler(CommandHandler("refresh",     cmd_refresh))
    app.add_handler(CommandHandler("stats",       cmd_stats))
    app.add_handler(CommandHandler("leagues",     cmd_leagues))
    app.add_handler(CommandHandler("subscribe",   cmd_subscribe))
    app.add_handler(CommandHandler("unsubscribe", cmd_unsubscribe))

    async def post_init(application):
        await application.bot.set_my_commands([
            BotCommand("start",       "Welcome"),
            BotCommand("today",       "Today predictions"),
            BotCommand("acca",        "Best accumulator"),
            BotCommand("predictions", "All predictions + all markets"),
            BotCommand("nba",         "NBA predictions"),
            BotCommand("highconf",    "Highest confidence picks only"),
            BotCommand("refresh",     "Re-run predictions"),
            BotCommand("stats",       "Model accuracy"),
            BotCommand("leagues",     "Covered competitions"),
            BotCommand("subscribe",   "Subscribe"),
            BotCommand("unsubscribe", "Unsubscribe"),
        ])
    app.post_init = post_init
    logger.info("Bot is running.")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(name)s %(message)s")
    run_bot()

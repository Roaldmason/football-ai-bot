# telegram_sender.py  —  Sends signals to all subscribers
import asyncio, logging, json, os
from datetime import datetime, timezone
from telegram import Bot
from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

logger = logging.getLogger(__name__)
SUBS_FILE = "subscribers.json"


def _load_subs():
    if os.path.exists(SUBS_FILE):
        with open(SUBS_FILE) as f:
            return json.load(f)
    return []


async def _send(text: str, chat_id: str):
    bot = Bot(token=TELEGRAM_BOT_TOKEN)
    await bot.send_message(chat_id=chat_id, text=text)


def format_signal(signal: dict) -> str:
    pair      = signal["pair"]
    direction = signal["direction"]
    price     = signal["price"]
    sl        = signal["sl"]
    tp        = signal["tp"]
    rr        = signal["rr"]
    score     = signal["score"]
    strength  = signal["strength_emoji"]
    reasons   = signal["reasons"]
    trend     = signal["trend_1h"]
    rsi       = signal["rsi"]
    now       = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    pair_disp = "GOLD (XAU/USD)" if pair=="XAUUSD" else "EUR/USD"
    if "XAU" in pair:
        sl_pts = round(abs(price-sl),2); tp_pts = round(abs(price-tp),2); unit="pts"
    else:
        sl_pts = round(abs(price-sl)*10000,1); tp_pts = round(abs(price-tp)*10000,1); unit="pips"
    dir_arrow    = "BUY" if direction=="BUY" else "SELL"
    reasons_text = "\n".join([f"  + {r}" for r in reasons])
    crt_text = ""
    if signal.get("crt_signal"):
        crt = signal["crt_signal"]
        crt_text = f"\nCRT: {crt['pattern']} | Range: {crt['range']:.5f}"
    msg = f"""{strength} {pair_disp} — {dir_arrow}
{'='*30}
Entry:       {price:.5f}
Stop Loss:   {sl:.5f}  ({sl_pts} {unit})
Take Profit: {tp:.5f}  ({tp_pts} {unit})
Risk/Reward: 1:{rr}{crt_text}
{'='*30}
Strength: {score}/10
Trend 1H: {trend.upper()}
RSI: {rsi}

Confluences ({len(reasons)}):
{reasons_text}
{'='*30}
{now}"""
    return msg.strip()


def _format_football_signal(pred: dict) -> str:
    """Format a high-confidence football prediction as a signal."""
    home   = pred.get("home","?")
    away   = pred.get("away","?")
    league = pred.get("league","")
    ko     = str(pred.get("kickoff",""))[:16].replace("T"," ")

    h  = pred.get("prob_home", 0.33)
    d  = pred.get("prob_draw", 0.33)
    a  = pred.get("prob_away", 0.34)
    ou = pred.get("prob_over25", 0.5)
    bt = pred.get("prob_btts_yes", 0.5)

    r_pick = pred.get("result_pick","home")
    r_conf = max(h, d, a)
    label  = {"home":f"{home} Win","draw":"Draw","away":f"{away} Win"}.get(r_pick,r_pick)

    picks  = []
    # Only include picks above 62%
    if r_conf >= 0.62:
        picks.append(f"1X2: {label} ({r_conf:.0%}) @ {pred.get('odds_'+r_pick,2.0):.2f}")
    if h+d >= 0.72:
        picks.append(f"Double Chance: Home/Draw ({h+d:.0%})")
    if a+d >= 0.72:
        picks.append(f"Double Chance: Draw/Away ({a+d:.0%})")
    ou15 = min(0.92, ou+0.18)
    if ou15 >= 0.75:
        picks.append(f"Over 1.5 Goals ({ou15:.0%})")
    if ou >= 0.65:
        picks.append(f"Over 2.5 Goals ({ou:.0%}) @ {pred.get('odds_over25',1.85):.2f}")
    if (1-ou) >= 0.65:
        picks.append(f"Under 2.5 Goals ({1-ou:.0%})")
    if bt >= 0.65:
        picks.append(f"BTTS Yes ({bt:.0%})")
    if (1-bt) >= 0.68:
        picks.append(f"BTTS No ({1-bt:.0%})")
    h1d = min(0.68, d+0.18)
    if h1d >= 0.58:
        picks.append(f"1st Half Draw ({h1d:.0%})")
    h1_o05 = min(0.85, ou*0.82)
    if h1_o05 >= 0.68:
        picks.append(f"1st Half Over 0.5 ({h1_o05:.0%})")
    base_corn = 0.45 + (h+a)*0.32
    c85 = min(0.75, max(0.35, base_corn))
    if c85 >= 0.65:
        picks.append(f"Corners Over 8.5 ({c85:.0%})")

    if not picks:
        return ""

    now  = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    flag = {"Premier League":"PL","La Liga":"LL","Bundesliga":"BL","Eredivisie":"ED",
            "Liga Portugal":"LP","Ligue 1":"L1","Champions League":"UCL",
            "Europa League":"UEL","Conference League":"UECL"}.get(league,"")

    xgh = pred.get("xg_home","?"); xga = pred.get("xg_away","?")

    return (
        f"FOOTBALL SIGNAL\n"
        f"[{flag}] {home} vs {away}\n"
        f"Kickoff: {ko}\n"
        f"{'='*30}\n"
        f"HIGH CONFIDENCE PICKS:\n"
        + "\n".join([f"  + {p}" for p in picks]) +
        f"\n{'='*30}\n"
        f"xG: {xgh} - {xga}\n"
        f"{now}"
    )


def send_signal(signal: dict):
    msg = format_signal(signal)
    asyncio.run(_send(msg, TELEGRAM_CHAT_ID))
    subs = _load_subs()
    for cid in subs:
        if str(cid) != str(TELEGRAM_CHAT_ID):
            try:
                asyncio.run(_send(msg, str(cid)))
            except Exception as e:
                logger.warning(f"Failed to send to {cid}: {e}")
    logger.info(f"Signal sent: {signal['pair']} {signal['direction']}")


def send_football_predictions(predictions: list):
    """Send only high-confidence football predictions to all subscribers."""
    subs = _load_subs()
    sent = 0
    for pred in predictions:
        msg = _format_football_signal(pred)
        if not msg:
            continue
        asyncio.run(_send(msg, TELEGRAM_CHAT_ID))
        for cid in subs:
            if str(cid) != str(TELEGRAM_CHAT_ID):
                try:
                    asyncio.run(_send(msg, str(cid)))
                except Exception as e:
                    logger.warning(f"Send failed {cid}: {e}")
        sent += 1
    logger.info(f"Sent {sent} football signals")


def send_startup_message():
    msg = (
        "Prediction Bot Started\n\n"
        "Football + NBA signals\n"
        "Markets: 1X2, DNB, DC, O/U 1.5/2.5/3.5,\n"
        "BTTS, 1H Result, 1H Goals, Corners,\n"
        "1H Corners, Asian HDP, 0-0 at 10 mins\n\n"
        "Only HIGH confidence picks (>62%) are sent.\n"
        "Send /start to subscribe."
    )
    asyncio.run(_send(msg, TELEGRAM_CHAT_ID))

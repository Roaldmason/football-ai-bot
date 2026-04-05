# main_signal_bot.py  —  Main loop with football + NBA signals
import time, json, os, logging, schedule, threading
from datetime import datetime, timezone
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
from signal_engine import analyze_pair
from telegram_sender import send_signal, send_startup_message, send_football_predictions
from telegram_bot import (cmd_start, cmd_today, cmd_acca, cmd_predictions,
                           cmd_nba, cmd_highconf, cmd_refresh, cmd_stats,
                           cmd_leagues, cmd_subscribe, cmd_unsubscribe, add_sub)
from config import PAIRS, CHECK_INTERVAL_MIN, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("signals.log")]
)
logger = logging.getLogger(__name__)
last_signals = {}


def should_send(pair, direction):
    key = f"{pair}_{direction}"
    if key in last_signals:
        if (datetime.now(timezone.utc) - last_signals[key]).seconds < 14400:
            return False
    last_signals[key] = datetime.now(timezone.utc)
    return True


def scan_pairs():
    logger.info(f"Scanning {len(PAIRS)} pairs ...")
    for pair in PAIRS:
        try:
            signal = analyze_pair(pair)
            if signal and should_send(pair, signal["direction"]):
                send_signal(signal)
                crt_tag = " +CRT!" if signal.get("crt_signal") else ""
                logger.info(f"SIGNAL: {pair} {signal['direction']} Score:{signal['score']}{crt_tag}")
            else:
                logger.info(f"{pair}: No signal")
        except Exception as e:
            logger.error(f"Error {pair}: {e}")


def scan_football_predictions():
    """Run football predictions and send only high-confidence picks."""
    try:
        from predictor import run_predictions
        df = run_predictions(days_ahead=2)
        if df.empty:
            logger.info("No football predictions generated")
            return
        # Only high confidence rows
        high = df[df["result_conf"] >= 0.62]
        if not high.empty:
            preds = high.to_dict("records")
            send_football_predictions(preds)
            logger.info(f"Sent {len(preds)} high-conf football predictions")
        else:
            logger.info("No high-confidence football predictions today")
    except Exception as e:
        logger.error(f"Football predictions error: {e}")


def run():
    logger.info("Bot starting ...")
    send_startup_message()
    add_sub(TELEGRAM_CHAT_ID)

    # Initial scans
    scan_pairs()
    scan_football_predictions()

    # Schedule
    schedule.every(CHECK_INTERVAL_MIN).minutes.do(scan_pairs)
    schedule.every(4).hours.do(scan_football_predictions)

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

    def run_schedule():
        while True:
            schedule.run_pending()
            time.sleep(30)

    threading.Thread(target=run_schedule, daemon=True).start()
    logger.info(f"Bot running — forex every {CHECK_INTERVAL_MIN} mins, football every 4 hrs")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    run()

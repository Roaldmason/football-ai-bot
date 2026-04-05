# ============================================================
#  scheduler.py  –  Daily auto-run: collect → predict → push
# ============================================================

import asyncio
import logging
import schedule
import time
from datetime import datetime

from config import DAILY_RUN_TIME, TELEGRAM_BOT_TOKEN, TELEGRAM_CHANNEL_ID
from data_collector import fetch_all_historical, fetch_upcoming_fixtures
from predictor import run_predictions
from odds_builder import build_accumulator, format_all_accas

logger = logging.getLogger(__name__)


async def _send_telegram(text: str):
    """Send a message to the channel without running the full bot."""
    from telegram import Bot
    bot = Bot(token=TELEGRAM_BOT_TOKEN)
    # Split long messages
    max_len = 4096
    for i in range(0, len(text), max_len):
        await bot.send_message(
            chat_id=TELEGRAM_CHANNEL_ID,
            text=text[i:i + max_len],
            parse_mode="Markdown"
        )


def daily_job():
    """The full daily pipeline."""
    logger.info(f"[{datetime.now():%H:%M:%S}] Starting daily pipeline …")

    # 1. Refresh historical data (once a week would be sufficient but daily is safe)
    try:
        logger.info("Collecting latest results …")
        fetch_all_historical()
    except Exception as e:
        logger.error(f"Historical fetch failed: {e}")

    # 2. Run ML predictions
    try:
        logger.info("Running predictions …")
        df = run_predictions(days_ahead=2)
        if df.empty:
            logger.warning("No fixtures to predict today.")
            return
    except Exception as e:
        logger.error(f"Prediction run failed: {e}")
        return

    # 3. Build accumulator
    try:
        accas = build_accumulator(df)
        msg   = (
            "🌅 *Good morning! Daily Acca is ready* ⚽\n\n"
            + format_all_accas(accas)
            + "\n\n⚠️ _For entertainment only. Bet responsibly. 18+_"
        )
        asyncio.run(_send_telegram(msg))
        logger.info("Daily accumulator sent to Telegram.")
    except Exception as e:
        logger.error(f"Telegram push failed: {e}")


def run_scheduler():
    """Block forever running daily_job at DAILY_RUN_TIME."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("bot.log"),
        ]
    )
    logger.info(f"Scheduler started. Daily job at {DAILY_RUN_TIME} UTC.")
    schedule.every().day.at(DAILY_RUN_TIME).do(daily_job)

    # Run once immediately on startup
    logger.info("Running initial pipeline on startup …")
    daily_job()

    while True:
        schedule.run_pending()
        time.sleep(30)


if __name__ == "__main__":
    run_scheduler()

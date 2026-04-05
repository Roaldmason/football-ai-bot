#!/usr/bin/env python3
# ============================================================
#  main.py  —  Single entry point. Choose run mode via CLI.
#
#  Usage:
#    python main.py train       # Collect data & train models
#    python main.py predict     # Run predictions for today
#    python main.py acca        # Print accumulator to console
#    python main.py bot         # Start Telegram bot (polling)
#    python main.py schedule    # Start auto-scheduler (VPS)
#    python main.py all         # Full pipeline + bot
# ============================================================

import sys
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("bot.log"),
    ]
)
logger = logging.getLogger("main")


def cmd_train():
    from data_collector import fetch_all_historical
    from model_training import train
    logger.info("Fetching all historical results ...")
    hist_df = fetch_all_historical()
    logger.info("Training models ...")
    train(hist_df)


def cmd_predict():
    from predictor import run_predictions
    df = run_predictions(days_ahead=2)
    if df.empty:
        print("No predictions generated.")
    else:
        print(df[["league", "home", "away", "result_pick", "result_conf",
                   "ou_pick", "btts_pick"]].to_string(index=False))


def cmd_acca():
    import os
    import pandas as pd
    from odds_builder import build_accumulator, format_all_accas
    from config import DATA_DIR
    path = f"{DATA_DIR}/latest_predictions.csv"
    if not os.path.exists(path):
        print("No predictions file found. Run: python main.py predict")
        return
    df    = pd.read_csv(path)
    accas = build_accumulator(df)
    print(format_all_accas(accas))


def cmd_bot():
    from telegram_bot import run_bot
    run_bot()


def cmd_schedule():
    from scheduler import run_scheduler
    run_scheduler()


def cmd_all():
    cmd_train()
    cmd_predict()
    cmd_acca()
    cmd_bot()


COMMANDS = {
    "train":    cmd_train,
    "predict":  cmd_predict,
    "acca":     cmd_acca,
    "bot":      cmd_bot,
    "schedule": cmd_schedule,
    "all":      cmd_all,
}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print("Usage: python main.py [train|predict|acca|bot|schedule|all]")
        sys.exit(1)
    COMMANDS[sys.argv[1]]()
import os
TELEGRAM_CHANNEL_ID = os.getenv("TELEGRAM_CHANNEL_ID", "")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHANNEL_ID = os.getenv("TELEGRAM_CHANNEL_ID", "")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
API_KEY = os.getenv("API_KEY")
# football-data.org competition codes
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
DAILY_RUN_TIME   = "08:00"
DATA_DIR         = "data"
MODEL_DIR        = "models"
LOG_FILE         = "bot.log"
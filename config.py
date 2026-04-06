import os
 import os

TELEGRAM_BOT_TOKEN  = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHANNEL_ID = os.getenv("TELEGRAM_CHANNEL_ID", "")
API_KEY             = os.getenv("API_KEY")
API_FOOTBALL_KEY    = os.getenv("API_FOOTBALL_KEY", "")
API_FOOTBALL_BASE   = os.getenv("API_FOOTBALL_BASE", "https://api-football-v3.p.rapidapi.com")
API_FOOTBALL_SEASON = os.getenv("API_FOOTBALL_SEASON", "2024")
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
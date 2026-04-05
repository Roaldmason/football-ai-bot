# config.py  —  football-data.org v4
API_FOOTBALL_KEY    = "e0afc3e5ec334d44bc3d207811fbc764"
API_FOOTBALL_BASE   = "https://api.football-data.org/v4"
API_FOOTBALL_SEASON = 2025   # 2025 = 2025/26 season on football-data.org

TELEGRAM_BOT_TOKEN  = "8573812227:AAGyn4ab2tkWxhG8Kn5W7DbhYQoLGaDIlnI"
TELEGRAM_CHANNEL_ID = " 8573812227"

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
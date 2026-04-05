# ⚽ Football AI Prediction Bot

AI-powered football match predictor with Telegram delivery.
Covers: Premier League · La Liga · Bundesliga · Eredivisie · Liga Portugal · Ligue 1

---

## 🗂 Project Structure

```
football_ai_bot/
├── config.py               ← API keys & settings (edit this first)
├── data_collector.py       ← Pulls fixtures, results, odds from API-Football
├── feature_engineering.py  ← Builds ML features (form, xG, H2H, etc.)
├── model_training.py       ← Trains XGBoost / RF models
├── predictor.py            ← Runs per-fixture predictions
├── odds_builder.py         ← Builds ~5.00 accumulator tickets
├── telegram_bot.py         ← Telegram bot with commands
├── scheduler.py            ← Daily auto-run pipeline
├── main.py                 ← CLI entry point
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure API keys — open `config.py` and fill in:

```python
API_FOOTBALL_KEY    = "e0afc3e5ec334d44bc3d207811fbc764"  # already set
TELEGRAM_BOT_TOKEN  = "YOUR_BOT_TOKEN"     # from @BotFather on Telegram
TELEGRAM_CHANNEL_ID = "YOUR_CHANNEL_ID"    # e.g. -1001234567890
```

#### Get your Telegram Bot Token:
1. Open Telegram → search @BotFather
2. Send `/newbot` and follow the prompts
3. Copy the token into config.py

#### Get your Channel ID:
1. Add your bot as admin to your channel
2. Send a message to the channel
3. Visit: `https://api.telegram.org/bot<TOKEN>/getUpdates`
4. Copy the `chat.id` value (e.g. `-1001234567890`)

---

## 🚀 Running the Bot

### Step 1 — Train the AI models (first time only)
```bash
python main.py train
```
This fetches historical match data for all 6 leagues and trains three models:
- **1X2 Result** (XGBoost)
- **Over/Under 2.5 Goals** (XGBoost)
- **Both Teams to Score** (XGBoost)

### Step 2 — Run predictions
```bash
python main.py predict
```

### Step 3 — See the accumulator
```bash
python main.py acca
```

### Step 4 — Start the Telegram bot
```bash
python main.py bot
```

### Step 5 — Run everything on autopilot (VPS)
```bash
python main.py schedule
```

---

## 📱 Telegram Commands

| Command | Description |
|---|---|
| `/start` | Welcome message |
| `/today` | Top predictions for today |
| `/acca` | Best ~5.00 accumulator |
| `/predictions` | All predictions per league |
| `/refresh` | Re-run predictions now |
| `/stats` | Model accuracy stats |
| `/leagues` | Covered leagues |

---

## 🖥️ Running on a VPS (Ubuntu)

```bash
# 1. Upload project to your VPS
scp -r football_ai_bot/ user@yourserver:~/

# 2. SSH in and install Python deps
ssh user@yourserver
cd football_ai_bot
pip install -r requirements.txt

# 3. Train models once
python main.py train

# 4. Run scheduler as a background service
nohup python main.py schedule > bot.log 2>&1 &

# Or use systemd (recommended):
sudo nano /etc/systemd/system/football-bot.service
```

**Systemd service file:**
```ini
[Unit]
Description=Football AI Prediction Bot
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/football_ai_bot
ExecStart=/usr/bin/python3 main.py schedule
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable football-bot
sudo systemctl start football-bot
sudo systemctl status football-bot
```

---

## 📊 Example Telegram Output

```
🏆 DAILY 5-ODDS ACCUMULATOR
━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🔹 Leg 1:  Arsenal vs Everton
   📋 Market:     1X2
   ✅ Pick:       Arsenal Win ⭐ VALUE
   📊 Confidence: 78%
   💰 Odds:       1.55
   ⚽ xG:         2.1 – 0.7

🔹 Leg 2:  Bayern Munich vs Augsburg
   📋 Market:     Over/Under 2.5
   ✅ Pick:       Over 2.5 Goals
   📊 Confidence: 72%
   💰 Odds:       1.65

🔹 Leg 3:  Barcelona vs Sevilla
   📋 Market:     1X2
   ✅ Pick:       Barcelona Win
   📊 Confidence: 75%
   💰 Odds:       1.48

──────────────────────────────
📈 TOTAL ODDS:  5.04
🧠 AVG CONF:   75%
🏷  LEGS:       3

⚠️ Predictions are for entertainment only.
Bet responsibly. 18+
```

---

## ⚠️ API Rate Limits

The free API-Football plan allows **100 requests/day**.
Each full pipeline run uses approximately 30–60 requests depending on fixtures available.

To increase limits, upgrade at: https://api-sports.io/

---

## 📜 Disclaimer

This bot is for **educational and entertainment purposes only**.
Never bet more than you can afford to lose. Gambling can be addictive.
If you or someone you know has a gambling problem, seek help.

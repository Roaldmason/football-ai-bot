import os
import time
import ccxt

API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")

exchange = ccxt.binance({
    'apiKey': API_KEY,
    'secret': API_SECRET,
})

while True:
    try:
        ticker = exchange.fetch_ticker('BTC/USDT')
        price = ticker['last']
        print(f"BTC Price: ${price}")
        time.sleep(60)
    except Exception as e:
        print(f"Error: {e}")
        time.sleep(30)
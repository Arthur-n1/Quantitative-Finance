import datetime
import os
import time
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import requests
import yfinance as yf
import alpaca_trade_api as tradeapi

# =====================================================================
# CONFIGURATION ALPACA (RÉCUPÉRATION SÉCURISÉE VIA GITHUB SECRETS)
# =====================================================================
API_KEY = os.getenv("APCA_API_KEY_ID")
SECRET_KEY = os.getenv("APCA_API_SECRET_KEY")
BASE_URL = "https://paper-api.alpaca.markets"

# Initialisation de la connexion API
api = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL, api_version="v2")

# =====================================================================
# 1. CHARGEMENT DE L'UNIVERS (901 ACTIONS)
# =====================================================================
headers = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML,"
        " like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}


def get_wiki_tickers(url):
  try:
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")
    table = soup.find("table", {"id": "constituents"})
    if not table:
      table = soup.find("table", {"class": "wikitable"})
    df = pd.read_html(str(table))[0]
    col = "Symbol" if "Symbol" in df.columns else "Ticker"
    return (
        df[col]
        .astype(str)
        .str.replace(".", "-", regex=False)
        .str.strip()
        .tolist()
    )
  except Exception:
    return []


print("Récupération de l'univers des constituants...")
sp500_tickers = get_wiki_tickers(
    "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
)
sp400_tickers = get_wiki_tickers(
    "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies"
)
try:
  response = requests.get(
      "https://en.wikipedia.org/wiki/Nasdaq-100", headers=headers
  )
  nasdaq_tickers = (
      pd.read_html(response.text)[4]["Ticker"]
      .str.replace(".", "-", regex=False)
      .tolist()
  )
except Exception:
  nasdaq_tickers = []

universe_tickers = list(set(sp500_tickers + sp400_tickers + nasdaq_tickers))

start_date = "2020-01-01"
end_date = datetime.datetime.now().strftime("%Y-%m-%d")

print("Téléchargement des données historiques globales...")
all_tickers_to_download = universe_tickers + ["SPY"]
raw_data = yf.download(
    all_tickers_to_download, start=start_date, end=end_date, progress=False
)["Close"]
raw_data = raw_data.ffill().bfill()
returns_all = raw_data.pct_change().fillna(0)
sector_daily_return = returns_all["SPY"]

# =====================================================================
# 2. CALCUL DU TOP 5 ACTUEL (Logique de sélection)
# =====================================================================
ENTRY_Z = -1.5
EXIT_Z = 0.0
STOP_LOSS_Z = -3.0
lookback = 30

train_results = []
print("\nAnalyse de l'univers pour identifier le Top 5...")

for ticker in universe_tickers:
  if ticker not in returns_all.columns:
    continue
  sub_returns = returns_all[ticker]
  spread = sub_returns - sector_daily_return
  spread_cum = (1 + spread).cumprod()
  rolling_mean = spread_cum.rolling(window=lookback).mean()
  rolling_std = spread_cum.rolling(window=lookback).std().replace(0, np.nan)
  z_score = ((spread_cum - rolling_mean) / rolling_std).fillna(0)

  position, trades = 0, 0
  for z in z_score:
    if position == 0 and z <= ENTRY_Z:
      position = 1
      trades += 1
    elif position == 1 and (z >= EXIT_Z or z <= STOP_LOSS_Z):
      position = 0
      trades += 1

  if trades >= 5:
    bh_ret = (1 + sub_returns).prod() - 1
    train_results.append({"Ticker": ticker, "Score": bh_ret})

train_df = pd.DataFrame(train_results)
best_tickers = (
    train_df.sort_values(by="Score", ascending=False).head(5)["Ticker"].tolist()
)

print(f"\n🎯 Top 5 actions sélectionnées pour le trading : {best_tickers}")

# =====================================================================
# 3. EXÉCUTION DES ORDRES SUR ALPACA (PAPER TRADING)
# =====================================================================
print("\nVérification des signaux et passage des ordres sur Alpaca...")

for ticker in best_tickers:
  try:
    sub_returns = returns_all[ticker]
    spread = sub_returns - sector_daily_return
    spread_cum = (1 + spread).cumprod()
    rolling_mean = spread_cum.rolling(window=lookback).mean()
    rolling_std = spread_cum.rolling(window=lookback).std().replace(0, np.nan)
    z_score = ((spread_cum - rolling_mean) / rolling_std).fillna(0)

    current_z = z_score.iloc[-1]
    print(f"-> {ticker} : Z-Score actuel = {current_z:.2f}")

    # Si le Z-score franchit le seuil d'achat, on envoie l'ordre
    if current_z <= ENTRY_Z:
      print(f"⚡ Signal d'achat validé pour {ticker} !")
      # Décommente la ligne ci-dessous pour exécuter l'ordre réel sur Alpaca :
      # api.submit_order(symbol=ticker, qty=1, side='buy', type='market', time_in_force='gtc')
    else:
      print(f"Pas de signal d'achat pour {ticker} (Z-score trop haut).")

  except Exception as e:
    print(f"Erreur lors du traitement pour {ticker}: {e}")

print("\nFin du script d'évaluation.")
"""
05_alpaca_paper_trading.py
============================
Schritt 5 (MVP): Live-/Paper-Trading-Test über die Alpaca API (EINMALIGE AUSFÜHRUNG)

Dieses Script führt exakt EINEN Markt-Check aus:
1. Holt die letzten 400 Stunden BTC/USD über die Alpaca Crypto Data API.
2. Berechnet daraus dieselben STATIONÄREN Features wie in Schritt 3 für die
   letzte VOLLSTÄNDIG ABGESCHLOSSENE Stunde (iloc[-2]).
3. Lädt das in Schritt 4 trainierte Random-Forest-Modell (3% / 1% Strategie).
4. Sagt das aktuelle Regime (0-4) vorher.
5. Platziert bei einem Signal (1, 3) eine schützende BRACKET ORDER im Alpaca Account
   mit exakt +3.0% Take-Profit und -1.0% Stop-Loss.
6. Beendet sich nach der Ausführung sofort wieder.
"""

# ── IMPORTS ──────────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime, timedelta, timezone

from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import TakeProfitRequest, StopLossRequest

# ── KONFIGURATION ─────────────────────────────────────────────────────────────
# ⚠️ Hier eigene Alpaca PAPER-Trading API-Keys eintragen!
API_KEY    = "PKAMJ4ZDYEKTP34NX44GNWXCJF"
API_SECRET = "B43Rr36ZV9NSpQSCwKqABkAMvDTJRV8wgjcyoFzYYULA"

SYMBOL = "BTC/USD"
LOOKBACK_HOURS = 400
TRADE_QTY = 0.001

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "artifacts", "models", "regime_rf_model.pkl")

REGIME_LABELS = {
    0: "Low Vol -> HOLD",
    1: "Medium Vol Up -> BUY",
    2: "Medium Vol Down -> SELL",
    3: "High Vol Up -> STRONG BUY",
    4: "High Vol Down -> STRONG SELL",
}

# ── LIVE-DATEN LADEN ────────────────────────────────────────────────────────────
def fetch_latest_bars(client: CryptoHistoricalDataClient) -> pd.DataFrame:
    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=LOOKBACK_HOURS)

    print(f"[INFO] Lade Live-Daten ({SYMBOL}) von {start} bis {end}...")

    request = CryptoBarsRequest(
        symbol_or_symbols=SYMBOL,
        timeframe=TimeFrame.Hour,
        start=start,
        end=end,
    )
    bars = client.get_crypto_bars(request)
    df = bars.df

    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index(level=0, drop=True)

    df = df.sort_index()
    print(f"[INFO] {len(df)} Bars geladen (letzte: {df.index[-1]})")
    return df

# ── FEATURE ENGINEERING (1:1 ABSTIMMUNG MIT SCHRITT 3 & 4) ──────────────────────
def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Returns & Volatilität
    df["returns"] = df["close"].pct_change()
    df["volatility_24"] = df["returns"].rolling(24).std()
    df["direction_6"] = df["close"].pct_change(6).shift(-6)

    # Preis-Lags als prozentualer, stationärer Abstand
    df["lag_1"]   = (df["close"].shift(1) - df["close"]) / df["close"]
    df["lag_6"]   = (df["close"].shift(6) - df["close"]) / df["close"]
    df["lag_24"]  = (df["close"].shift(24) - df["close"]) / df["close"]
    df["lag_168"] = (df["close"].shift(168) - df["close"]) / df["close"]

    df["return_1"]   = df["close"].pct_change(1)
    df["return_6"]   = df["close"].pct_change(6)
    df["return_24"]  = df["close"].pct_change(24)
    df["return_168"] = df["close"].pct_change(168)

    df["volatility_6"]   = df["returns"].rolling(6).std()
    df["volatility_24"]  = df["returns"].rolling(24).std()
    df["volatility_168"] = df["returns"].rolling(168).std()

    # EMA Ribbon (Stationär)
    for period in [6, 12, 18, 24, 50, 100]:
        abs_ema = df["close"].ewm(span=period, adjust=False).mean()
        df[f"EMA_{period}"] = (abs_ema - df["close"]) / df["close"]

    df["ribbon_spread"]      = df["EMA_6"] - df["EMA_50"]
    df["ribbon_spread_norm"] = df["ribbon_spread"]

    df["ribbon_bull"] = ((df["EMA_6"] > df["EMA_12"]) & (df["EMA_12"] > df["EMA_18"])).astype(int)
    df["ribbon_bear"] = ((df["EMA_6"] < df["EMA_12"]) & (df["EMA_12"] < df["EMA_18"])).astype(int)

    # SMAs (Stationär)
    df["SMA_12"] = (df["close"].rolling(12).mean() - df["close"]) / df["close"]
    df["SMA_24"] = (df["close"].rolling(24).mean() - df["close"]) / df["close"]
    df["SMA_50"] = (df["close"].rolling(50).mean() - df["close"]) / df["close"]
    df["price_vs_SMA50"] = df["SMA_50"]

    # MACD
    ema_12 = df["close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["close"].ewm(span=26, adjust=False).mean()
    df["MACD"]        = ema_12 - ema_26
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_hist"]   = df["MACD"] - df["MACD_signal"]
    df["MACD_hist_change"] = df["MACD_hist"] - df["MACD_hist"].shift(1)

    df["macd_cross_up"] = ((df["MACD"].shift(1) < df["MACD_signal"].shift(1)) & (df["MACD"] > df["MACD_signal"])).astype(int)
    df["macd_cross_down"] = ((df["MACD"].shift(1) > df["MACD_signal"].shift(1)) & (df["MACD"] < df["MACD_signal"])).astype(int)

    # Echter RSI mit Wilder's Glättung (EWMA)
    delta = df["close"].diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    df["RSI"] = 100 - (100 / (1 + avg_gain / avg_loss))

    df["rsi_oversold"]   = (df["RSI"] < 30).astype(int)
    df["rsi_overbought"] = (df["RSI"] > 70).astype(int)
    df["rsi_neutral"]    = ((df["RSI"] >= 30) & (df["RSI"] <= 70)).astype(int)

    # Bollinger Bands
    df["BB_middle"] = df["close"].rolling(20).mean()
    df["BB_std"]    = df["close"].rolling(20).std()
    df["BB_upper"]  = df["BB_middle"] + (2 * df["BB_std"])
    df["BB_lower"]  = df["BB_middle"] - (2 * df["BB_std"])
    df["BB_width"]  = (df["BB_upper"] - df["BB_lower"]) / df["BB_middle"]
    df["BB_pos"]    = (df["close"] - df["BB_lower"]) / (df["BB_upper"] - df["BB_lower"])
    df["BB_squeeze"] = (df["BB_width"] < df["BB_width"].rolling(24).mean()).astype(int)

    # Volumen
    df["volume_ma_24"] = df["volume"].rolling(24).mean()
    df["volume_ratio"] = df["volume"] / df["volume_ma_24"]
    df["volume_surge"] = (df["volume_ratio"] > 2.0).astype(int)

    # S&R (Stationär)
    df["rolling_high_24"]  = df["high"].rolling(24).max().shift(1)
    df["rolling_low_24"]   = df["low"].rolling(24).min().shift(1)
    df["rolling_high_168"] = df["high"].rolling(168).max().shift(1)
    df["rolling_low_168"]  = df["low"].rolling(168).min().shift(1)

    df["dist_to_high_24"]  = (df["rolling_high_24"]  - df["close"]) / df["close"]
    df["dist_to_low_24"]   = (df["close"] - df["rolling_low_24"])   / df["close"]
    df["dist_to_high_168"] = (df["rolling_high_168"] - df["close"]) / df["close"]
    df["dist_to_low_168"]  = (df["close"] - df["rolling_low_168"])  / df["close"]

    pivot_abs = (df["high"].shift(1) + df["low"].shift(1) + df["close"].shift(1)) / 3
    r1_abs    = 2 * pivot_abs - df["low"].shift(1)
    s1_abs    = 2 * pivot_abs - df["high"].shift(1)

    df["pivot"]      = (pivot_abs - df["close"]) / df["close"]
    df["R1"]         = (r1_abs - df["close"]) / df["close"]
    df["S1"]         = (s1_abs - df["close"]) / df["close"]
    df["dist_pivot"] = df["pivot"]
    df["dist_R1"]    = df["R1"]
    df["dist_S1"]    = df["S1"]

    df["round_level"]   = (df["close"] // 5000) * 5000
    df["dist_to_round"] = abs(df["close"] - df["round_level"]) / df["close"]

    # Zeitliche Features
    df["hour"]  = df.index.hour
    df["dow"]   = df.index.dayofweek
    df["month"] = df.index.month
    df["hour_sin"]  = np.sin(2 * np.pi * df["hour"]  / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * df["hour"]  / 24)
    df["dow_sin"]   = np.sin(2 * np.pi * df["dow"]   / 7)
    df["dow_cos"]   = np.cos(2 * np.pi * df["dow"]   / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # ATR (Wird als Fallback-Spalte mitgeführt, falls vom Export verlangt)
    high_low   = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift(1)).abs()
    low_close  = (df["low"]  - df["close"].shift(1)).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["ATR"] = true_range.rolling(14).mean()

    return df

# ── REGIME VORHERSAGE ─────────────────────────────────────────────────────────
def predict_current_regime(df_features: pd.DataFrame, model, feature_cols: list):
    df_valid = df_features.dropna(subset=feature_cols)

    if df_valid.empty:
        raise ValueError("Zu wenig Historie geladen -> LOOKBACK_HOURS erhöhen.")

    # Greife auf iloc[-2] zu (letzte vollendete Stunde)
    latest_row = df_valid.iloc[[-2]]
    X_live = latest_row[feature_cols]

    regime_pred = model.predict(X_live)[0]
    proba = model.predict_proba(X_live)[0]

    print(f"\n[INFO] Vorhersage basiert auf Bar von: {latest_row.index[0]}")
    print(f"[INFO] Letzter geschlossener Schlusskurs: {latest_row['close'].values[0]:.2f} USD")
    print(f"\n[VORHERSAGE-WAHRSCHEINLICHKEITEN]")
    for cls, p in zip(model.classes_, proba):
        print(f"  Regime {int(cls)} ({REGIME_LABELS[int(cls)]:<28}): {p*100:5.1f}%")

    return int(regime_pred), latest_row

# ── ORDER-LOGIK (FESTE PROZENT-STRATEGIE FÜR ALPACA) ──────────────────────────
def execute_trade_signal(trading_client: TradingClient, regime: int, latest_row: pd.DataFrame):
    signal = REGIME_LABELS[regime]
    print(f"\n[SIGNAL] Regime {regime} -> {signal}")

    try:
        position = trading_client.get_open_position(SYMBOL.replace("/", ""))
        position_qty = float(position.qty)
        print(f"[INFO] Aktuelle Position: {position_qty} BTC")
    except Exception:
        position_qty = 0.0
        print("[INFO] Aktuelle Position: keine offene Position")

    current_price = float(latest_row['close'].values[0])

    # ── BUY-Signale (Regime 1, 3) mit FESTEM PROZENT-AUSSTIEG ──
    if regime in (1, 3):
        if position_qty > 0:
            print("[ACTION] Bereits long positioniert -> keine neue Order.")
            return

        # Synchronisation mit Backtest: +3% Take-Profit / -1% Stop-Loss
        tp_price = current_price * 1.03  # +3% Gewinnziel
        sl_price = current_price * 0.99  # -1% Reißleine

        order = MarketOrderRequest(
            symbol=SYMBOL,
            qty=TRADE_QTY,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.GTC,
            take_profit=TakeProfitRequest(limit_price=round(tp_price, 2)),
            stop_loss=StopLossRequest(stop_price=round(sl_price, 2))
        )
        result = trading_client.submit_order(order)
        print(f"[ACTION] ✅ Bracket BUY Order platziert! Target (+3%): ${tp_price:.2f}, Stop (-1%): ${sl_price:.2f}")

    # ── SELL-Signale (Regime 2, 4) ──
    elif regime in (2, 4):
        if position_qty <= 0:
            print("[ACTION] Keine open Long-Position -> keine SELL-Order.")
            return

        order = MarketOrderRequest(
            symbol=SYMBOL,
            qty=position_qty,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.GTC,
        )
        result = trading_client.submit_order(order)
        print(f"[ACTION] ✅ Market SELL Order platziert: Position manuell per Signal geschlossen.")

    else:
        print("[ACTION] HOLD -> keine Order platziert.")

def print_account_status(trading_client: TradingClient):
    account = trading_client.get_account()
    print("\n" + "=" * 60 + "\n  ALPACA PAPER ACCOUNT\n" + "=" * 60)
    print(f"  Konto-Status     : {account.status}")
    print(f"  Kapital (Cash)   : {float(account.cash):,.2f} USD")
    print(f"  Portfolio-Wert   : {float(account.portfolio_value):,.2f} USD\n" + "=" * 60)

# ── HAUPTPROGRAMM (EINMALIGE LIVE-AUSFÜHRUNG) ──────────────────────────────────
def main():
    print(f"[INFO] Lade Modell aus {MODEL_PATH}...")
    saved = joblib.load(MODEL_PATH)
    model = saved["model"]
    feature_cols = saved["feature_cols"]

    data_client = CryptoHistoricalDataClient(API_KEY, API_SECRET)
    trading_client = TradingClient(API_KEY, API_SECRET, paper=True)

    print_account_status(trading_client)

    # Führt den Check exakt einmal aus
    df_raw = fetch_latest_bars(data_client)
    df_features = compute_features(df_raw)

    regime, latest_row = predict_current_regime(df_features, model, feature_cols)
    execute_trade_signal(trading_client, regime, latest_row)

    print("\n✅ Schritt 5 (Paper-Trading-Test) erfolgreich ausgeführt!")

if __name__ == "__main__":
    main()
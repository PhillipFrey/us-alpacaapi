"""
05_alpaca_paper_trading.py
============================
Schritt 5 (MVP): Rough Live-/Paper-Trading-Test über die Alpaca API

Dieses Script ist der erste End-to-End-Test des Modells aus Schritt 4 mit
ECHTZEIT-Daten:

1. Holt die letzten Stunden BTC/USD über die Alpaca Crypto Data API
2. Berechnet daraus dieselben Features wie in Schritt 3 (Pre-Split Prep)
   für die AKTUELLSTE abgeschlossene Stunde
3. Lädt das in Schritt 4 trainierte Random-Forest-Modell
   (artifacts/models/regime_rf_model.pkl)
4. Sagt das aktuelle Regime (0-4) vorher
5. Leitet daraus ein Signal ab (HOLD / BUY / SELL / STRONG BUY / STRONG SELL)
6. Platziert – NUR im Paper-Trading-Account – eine grobe Test-Order:
       - Regime 1/3 -> kleine Long-Position (Market BUY)
       - Regime 2/4 -> bestehende Long-Position schließen (Market SELL),
                       falls vorhanden
       - Regime 0   -> keine Aktion (HOLD)

WICHTIG:
    - Dieses Script verwendet AUSSCHLIESSLICH den Alpaca PAPER-Trading
      Endpunkt (paper=True). Es wird kein echtes Geld eingesetzt.
    - Die Order-Größe ist bewusst sehr klein gehalten (TRADE_QTY) und dient
      nur dem groben funktionalen Test der Pipeline (Daten -> Features ->
      Modell -> Order).
    - Dieses Script führt EINEN Durchlauf aus ("ein Tick"). Für echtes
      kontinuierliches Paper-Trading müsste es z.B. stündlich per Cronjob /
      Scheduler ausgeführt werden (siehe Hinweis am Skriptende).
    - Feature Selection, Hyperparameter-Tuning und ein robustes Risiko-
      management folgen erst NACH einem erfolgreichen MVP-Test.

Voraussetzungen:
    - artifacts/models/regime_rf_model.pkl muss existieren (Schritt 4)
    - Alpaca Paper-Trading API Keys (siehe API_KEY / API_SECRET unten)

Ausführung:
    python scripts/05_alpaca_paper_trading.py
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


# ── KONFIGURATION ─────────────────────────────────────────────────────────────
# ⚠️ Hier eigene Alpaca PAPER-Trading API-Keys eintragen
#    (Settings -> API Keys im Alpaca Paper-Account, NICHT Live-Account!)
API_KEY    = "DEIN_PAPER_API_KEY"
API_SECRET = "DEIN_PAPER_API_SECRET"

SYMBOL = "BTC/USD"

# Wie viele Stunden Historie laden wir, um alle rollenden Features
# (max. Lookback z.B. 50 für SMA50 / EMA50) berechnen zu können?
LOOKBACK_HOURS = 200

# Größe der Test-Order (BTC). Bewusst sehr klein für den groben MVP-Test.
TRADE_QTY = 0.001

# Pfade
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
    """
    Holt die letzten LOOKBACK_HOURS Stunden BTC/USD OHLCV-Daten über die
    Alpaca Crypto Data API.

    Args:
        client: Initialisierter CryptoHistoricalDataClient.

    Returns:
        pd.DataFrame: OHLCV-DataFrame mit datetime Index, sortiert aufsteigend.
    """
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

    # MultiIndex (symbol, timestamp) -> nur timestamp als Index behalten
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index(level=0, drop=True)

    df = df.sort_index()
    print(f"[INFO] {len(df)} Bars geladen "
          f"(letzte: {df.index[-1]})")

    return df


# ── FEATURE ENGINEERING (analog Schritt 3 / 03_pre_split_prep.py) ───────────────
def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Berechnet GENAU die gleichen Feature-Gruppen wie in Schritt 3
    (03_pre_split_prep.py), angewendet auf die aktuellen Live-Daten.
    Die Spaltennamen und Berechnungslogik sind 1:1 identisch zu
    03_pre_split_prep.py, damit das in Schritt 4 trainierte Modell
    (model.feature_cols) exakt passende Spalten erhält.

    WICHTIG (No-Leakage): Alle rollenden/verzögerten Features verwenden nur
    Informationen, die zum Zeitpunkt jeder Zeile bereits bekannt waren
    (.shift(1) wo nötig) – exakt wie im Trainings-Pipeline-Schritt.

    Feature-Gruppen (siehe 03_pre_split_prep.py):
        0. Returns & Volatilität (returns, volatility_24, direction_6)
        1. Price Lags (lag_1, lag_6, lag_24, lag_168)
        2. Rolling Returns (return_1, return_6, return_24, return_168)
        3. Volatility (volatility_6, volatility_24, volatility_168)
        4. EMA Ribbon (EMA_6/12/18/24/50/100, ribbon_spread*, ribbon_bull/bear)
        5. SMA (SMA_12/24/50, price_vs_SMA50)
        6. MACD (MACD, MACD_signal, MACD_hist, MACD_hist_change, macd_cross_up/down)
        7. RSI (RSI, rsi_oversold/overbought/neutral)
        8. Bollinger Bands (BB_middle/std/upper/lower/width/pos/squeeze)
        9. Volumen (volume_ma_24, volume_ratio, volume_surge)
        10. Support/Resistance (rolling_high/low_24/168, dist_to_*, pivot, R1, S1, ...)
        11. Zeitliche Features (hour/dow/month + sin/cos)
        12. ATR (zusätzlich aus Schritt 4, für Triple-Barrier)

    Args:
        df: Roher OHLCV-DataFrame (datetime Index).

    Returns:
        pd.DataFrame: DataFrame mit allen berechneten Feature-Spalten.
                      Die letzte Zeile enthält die aktuellsten Werte für die
                      Live-Vorhersage.
    """
    df = df.copy()

    # ── 0. Returns & Volatilität (analog add_target_variable, ohne Regime) ──
    df["returns"] = df["close"].pct_change()
    df["volatility_24"] = df["returns"].rolling(24).std()
    df["direction_6"] = df["close"].pct_change(6)

    # ── 1. Price Lags ────────────────────────────────────────────────────────
    df["lag_1"]   = df["close"].shift(1)
    df["lag_6"]   = df["close"].shift(6)
    df["lag_24"]  = df["close"].shift(24)
    df["lag_168"] = df["close"].shift(168)

    # ── 2. Rolling Returns ───────────────────────────────────────────────────
    df["return_1"]   = df["close"].pct_change(1)
    df["return_6"]   = df["close"].pct_change(6)
    df["return_24"]  = df["close"].pct_change(24)
    df["return_168"] = df["close"].pct_change(168)

    # ── 3. Volatility (mehrere Zeitskalen) ──────────────────────────────────
    df["volatility_6"]   = df["returns"].rolling(6).std()
    df["volatility_24"]  = df["returns"].rolling(24).std()
    df["volatility_168"] = df["returns"].rolling(168).std()

    # ── 4. EMA Ribbon ────────────────────────────────────────────────────────
    for period in [6, 12, 18, 24, 50, 100]:
        df[f"EMA_{period}"] = df["close"].ewm(span=period, adjust=False).mean()

    df["ribbon_spread"]      = df["EMA_6"] - df["EMA_50"]
    df["ribbon_spread_norm"] = df["ribbon_spread"] / df["close"]

    df["ribbon_bull"] = (
        (df["EMA_6"]  > df["EMA_12"]) &
        (df["EMA_12"] > df["EMA_18"]) &
        (df["EMA_18"] > df["EMA_24"])
    ).astype(int)

    df["ribbon_bear"] = (
        (df["EMA_6"]  < df["EMA_12"]) &
        (df["EMA_12"] < df["EMA_18"]) &
        (df["EMA_18"] < df["EMA_24"])
    ).astype(int)

    # ── 5. SMA ───────────────────────────────────────────────────────────────
    df["SMA_12"] = df["close"].rolling(12).mean()
    df["SMA_24"] = df["close"].rolling(24).mean()
    df["SMA_50"] = df["close"].rolling(50).mean()
    df["price_vs_SMA50"] = df["close"] / df["SMA_50"] - 1

    # ── 6. MACD ──────────────────────────────────────────────────────────────
    ema_12 = df["close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["close"].ewm(span=26, adjust=False).mean()

    df["MACD"]        = ema_12 - ema_26
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_hist"]   = df["MACD"] - df["MACD_signal"]
    df["MACD_hist_change"] = df["MACD_hist"] - df["MACD_hist"].shift(1)

    df["macd_cross_up"] = (
        (df["MACD"].shift(1) < df["MACD_signal"].shift(1)) &
        (df["MACD"]          > df["MACD_signal"])
    ).astype(int)

    df["macd_cross_down"] = (
        (df["MACD"].shift(1) > df["MACD_signal"].shift(1)) &
        (df["MACD"]          < df["MACD_signal"])
    ).astype(int)

    # ── 7. RSI (14) ──────────────────────────────────────────────────────────
    delta = df["close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()

    df["RSI"] = 100 - (100 / (1 + gain / loss))

    df["rsi_oversold"]   = (df["RSI"] < 30).astype(int)
    df["rsi_overbought"] = (df["RSI"] > 70).astype(int)
    df["rsi_neutral"]    = ((df["RSI"] >= 30) & (df["RSI"] <= 70)).astype(int)

    # ── 8. Bollinger Bands (20, 2 std) ──────────────────────────────────────
    df["BB_middle"] = df["close"].rolling(20).mean()
    df["BB_std"]    = df["close"].rolling(20).std()
    df["BB_upper"]  = df["BB_middle"] + (2 * df["BB_std"])
    df["BB_lower"]  = df["BB_middle"] - (2 * df["BB_std"])
    df["BB_width"]  = (df["BB_upper"] - df["BB_lower"]) / df["BB_middle"]
    df["BB_pos"]    = (df["close"] - df["BB_lower"]) / (df["BB_upper"] - df["BB_lower"])
    df["BB_squeeze"] = (df["BB_width"] < df["BB_width"].rolling(24).mean()).astype(int)

    # ── 9. Volumen ───────────────────────────────────────────────────────────
    df["volume_ma_24"] = df["volume"].rolling(24).mean()
    df["volume_ratio"] = df["volume"] / df["volume_ma_24"]
    df["volume_surge"] = (df["volume_ratio"] > 2.0).astype(int)

    # ── 10. Support & Resistance ─────────────────────────────────────────────
    df["rolling_high_24"]  = df["high"].rolling(24).max().shift(1)
    df["rolling_low_24"]   = df["low"].rolling(24).min().shift(1)
    df["rolling_high_168"] = df["high"].rolling(168).max().shift(1)
    df["rolling_low_168"]  = df["low"].rolling(168).min().shift(1)

    df["dist_to_high_24"]  = (df["rolling_high_24"]  - df["close"]) / df["close"]
    df["dist_to_low_24"]   = (df["close"] - df["rolling_low_24"])   / df["close"]
    df["dist_to_high_168"] = (df["rolling_high_168"] - df["close"]) / df["close"]
    df["dist_to_low_168"]  = (df["close"] - df["rolling_low_168"])  / df["close"]

    df["pivot"]      = (df["high"].shift(1) + df["low"].shift(1) + df["close"].shift(1)) / 3
    df["R1"]         = 2 * df["pivot"] - df["low"].shift(1)
    df["S1"]         = 2 * df["pivot"] - df["high"].shift(1)
    df["dist_pivot"] = (df["close"] - df["pivot"]) / df["close"]
    df["dist_R1"]    = (df["close"] - df["R1"])    / df["close"]
    df["dist_S1"]    = (df["close"] - df["S1"])    / df["close"]

    df["round_level"]   = (df["close"] // 5000) * 5000
    df["dist_to_round"] = abs(df["close"] - df["round_level"]) / df["close"]

    # ── 11. Zeitliche Features ───────────────────────────────────────────────
    df["hour"]  = df.index.hour
    df["dow"]   = df.index.dayofweek
    df["month"] = df.index.month

    df["hour_sin"]  = np.sin(2 * np.pi * df["hour"]  / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * df["hour"]  / 24)
    df["dow_sin"]   = np.sin(2 * np.pi * df["dow"]   / 7)
    df["dow_cos"]   = np.cos(2 * np.pi * df["dow"]   / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # ── 12. ATR (zusätzlich aus Schritt 4, für Triple-Barrier) ──────────────
    high_low   = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift(1)).abs()
    low_close  = (df["low"]  - df["close"].shift(1)).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["ATR"] = true_range.rolling(14).mean()

    return df



# ── REGIME VORHERSAGE ─────────────────────────────────────────────────────────
def predict_current_regime(df_features: pd.DataFrame, model, feature_cols: list):
    """
    Sagt das Regime (0-4) für die aktuellste vollständige Zeile vorher.

    Args:
        df_features: DataFrame mit allen berechneten Features.
        model: Trainiertes Random-Forest-Modell.
        feature_cols: Liste der vom Modell erwarteten Feature-Spalten
                       (aus dem .pkl gespeichert).

    Returns:
        tuple: (regime (int), timestamp der zugrunde liegenden Zeile)
    """
    # Letzte Zeile mit vollständigen (nicht-NaN) Feature-Werten verwenden
    df_valid = df_features.dropna(subset=feature_cols)

    if df_valid.empty:
        raise ValueError(
            "Keine vollständige Zeile für die Vorhersage verfügbar "
            "(zu wenig Historie geladen -> LOOKBACK_HOURS erhöhen)."
        )

    latest_row = df_valid.iloc[[-1]]
    X_live = latest_row[feature_cols]

    regime_pred = model.predict(X_live)[0]
    proba = model.predict_proba(X_live)[0]

    print(f"\n[INFO] Vorhersage basiert auf Bar von: {latest_row.index[0]}")
    print(f"[INFO] Letzter Schlusskurs: {latest_row['close'].values[0]:.2f} USD")
    print(f"\n[VORHERSAGE-WAHRSCHEINLICHKEITEN]")
    for cls, p in zip(model.classes_, proba):
        print(f"  Regime {int(cls)} ({REGIME_LABELS[int(cls)]:<28}): {p*100:5.1f}%")

    return int(regime_pred), latest_row.index[0]


# ── ORDER-LOGIK ───────────────────────────────────────────────────────────────
def execute_trade_signal(trading_client: TradingClient, regime: int):
    """
    Leitet aus dem vorhergesagten Regime ein Trading-Signal ab und platziert
    eine grobe Market-Order im Paper-Trading-Account.

    Logik (MVP, bewusst simpel):
        - Regime 1 oder 3 (Up-Signal)   -> Market BUY  (TRADE_QTY), wenn
                                            aktuell keine Position offen ist
        - Regime 2 oder 4 (Down-Signal) -> Market SELL der gesamten
                                            bestehenden Position (falls vorhanden)
        - Regime 0 (HOLD)               -> keine Aktion

    Args:
        trading_client: Initialisierter TradingClient (paper=True).
        regime: Vorhergesagtes Regime (0-4).
    """
    signal = REGIME_LABELS[regime]
    print(f"\n[SIGNAL] Regime {regime} -> {signal}")

    # Aktuelle Position prüfen
    try:
        position = trading_client.get_open_position(SYMBOL.replace("/", ""))
        position_qty = float(position.qty)
        print(f"[INFO] Aktuelle Position: {position_qty} BTC")
    except Exception:
        position_qty = 0.0
        print("[INFO] Aktuelle Position: keine offene Position")

    # ── BUY-Signale (Regime 1, 3) ────────────────────────────────────────────
    if regime in (1, 3):
        if position_qty > 0:
            print("[ACTION] Bereits long positioniert -> keine neue Order.")
            return

        order = MarketOrderRequest(
            symbol=SYMBOL,
            qty=TRADE_QTY,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.GTC,
        )
        result = trading_client.submit_order(order)
        print(f"[ACTION] ✅ Market BUY Order platziert: "
              f"{TRADE_QTY} {SYMBOL} (Order-ID: {result.id})")

    # ── SELL-Signale (Regime 2, 4) ───────────────────────────────────────────
    elif regime in (2, 4):
        if position_qty <= 0:
            print("[ACTION] Keine offene Long-Position -> keine SELL-Order.")
            return

        order = MarketOrderRequest(
            symbol=SYMBOL,
            qty=position_qty,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.GTC,
        )
        result = trading_client.submit_order(order)
        print(f"[ACTION] ✅ Market SELL Order platziert: "
              f"{position_qty} {SYMBOL} (Order-ID: {result.id})")

    # ── HOLD (Regime 0) ───────────────────────────────────────────────────────
    else:
        print("[ACTION] HOLD -> keine Order platziert.")


# ── ACCOUNT-INFO ──────────────────────────────────────────────────────────────
def print_account_status(trading_client: TradingClient):
    """Gibt einen kurzen Überblick über den Paper-Account aus."""
    account = trading_client.get_account()
    print("\n" + "=" * 60)
    print("  ALPACA PAPER ACCOUNT")
    print("=" * 60)
    print(f"  Konto-Status     : {account.status}")
    print(f"  Kapital (Cash)   : {float(account.cash):,.2f} USD")
    print(f"  Portfolio-Wert   : {float(account.portfolio_value):,.2f} USD")
    print("=" * 60)


# ── HAUPTPROGRAMM ─────────────────────────────────────────────────────────────
def main():
    """
    Hauptfunktion: Live-Daten holen, Features berechnen, Regime vorhersagen,
    grobe Paper-Order platzieren.
    """
    # Modell laden
    print(f"[INFO] Lade Modell aus {MODEL_PATH}...")
    saved = joblib.load(MODEL_PATH)
    model = saved["model"]
    feature_cols = saved["feature_cols"]
    print(f"[INFO] Modell geladen ({len(feature_cols)} Features erwartet)")

    # Clients initialisieren
    data_client = CryptoHistoricalDataClient(API_KEY, API_SECRET)
    trading_client = TradingClient(API_KEY, API_SECRET, paper=True)

    # Account-Status anzeigen
    print_account_status(trading_client)

    # Live-Daten laden & Features berechnen
    df_raw = fetch_latest_bars(data_client)
    df_features = compute_features(df_raw)

    # Regime vorhersagen
    regime, ts = predict_current_regime(df_features, model, feature_cols)

    # Trade-Signal ausführen (Paper-Order)
    execute_trade_signal(trading_client, regime)

    print("\n✅ Schritt 5 (Rough Paper-Trading-Test) abgeschlossen")
    print("\n[HINWEIS] Dieses Script führt EINEN Tick aus. Für laufendes")
    print("          Paper-Trading z.B. stündlich per Cronjob/Scheduler")
    print("          ausführen, z.B.:")
    print("          0 * * * * cd /pfad/zum/projekt && "
          ".venv/bin/python scripts/05_alpaca_paper_trading.py")


if __name__ == "__main__":
    main()

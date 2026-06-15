"""
04_modeling_and_backtest.py
============================
Schritt 4: KI-Modellierung (Random Forest) & Historischer Backtest

Dieses Script führt den Vergangenheits-Check durch:
1. Lädt die vorbereitete 'data/features.csv' aus Schritt 3.
2. Führt einen chronologischen Split durch (80% Training, 20% Testset).
3. Trainiert einen Random Forest Classifier auf den stationären Features.
4. Berechnet den historischen Backtest basierend auf FESTEN PROZENTGRENZEN:
   - Take-Profit (TP): +3.0%
   - Stop-Loss (SL)  : -1.0%
   - Max. Haltedauer : 24 Stunden (Timeout)
5. Berechnet die historische Win-Rate, die Equity-Kurve und speichert das Modell.
"""

# ── IMPORTS ──────────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib

# ── KONFIGURATION ─────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH = os.path.join(BASE_DIR, "data", "features.csv")
MODEL_DIR  = os.path.join(BASE_DIR, "artifacts", "models")
IMAGE_DIR  = os.path.join(BASE_DIR, "artifacts", "images")

MAX_HOLDING = 24  # Maximale Haltedauer eines Trades: 24 Stunden

# Verzeichnisse erstellen, falls sie fehlen
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)

# ── FESTE PROZENT-STRATEGIE (TRIPLE-BARRIER) ──────────────────────────────────
def triple_barrier_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Berechnet für JEDE Stunde das Triple-Barrier-Ergebnis in der Vergangenheit
    basierend auf festen Prozentgrenzen: Take-Profit = +3%, Stop-Loss = -1%.
    """
    closes  = df["close"].to_numpy()
    highs   = df["high"].to_numpy()
    lows    = df["low"].to_numpy()
    regimes = df["regime"].to_numpy()
    n = len(df)

    long_outcome  = np.full(n, np.nan)
    long_return   = np.full(n, np.nan)
    short_outcome = np.full(n, np.nan)
    short_return  = np.full(n, np.nan)

    print(f"[INFO] Berechne feste Prozent-Barrieren (TP=+3%, SL=-1%, Limit={MAX_HOLDING}h)...")

    for i in range(n - MAX_HOLDING):
        entry = closes[i]

        # ── LONG SIMULATION (+3% / -1%) ──────────────────────────────────────
        tp_long = entry * 1.03  # Gewinn bei +3%
        sl_long = entry * 0.99  # Reißleine bei -1%

        for j in range(i + 1, i + 1 + MAX_HOLDING):
            if highs[j] >= tp_long:
                long_outcome[i] = 1
                long_return[i]  = 0.03  # +3% Gewinn buchen
                break
            if lows[j] <= sl_long:
                long_outcome[i] = 0
                long_return[i]  = -0.01  # -1% Verlust buchen
                break
        else:
            # Weder TP noch SL erreicht -> Ausstieg nach 24 Stunden zum Endkurs
            long_outcome[i] = -1
            long_return[i]  = (closes[i + MAX_HOLDING] - entry) / entry

        # ── SHORT SIMULATION (+3% / -1%) ─────────────────────────────────────
        tp_short = entry * 0.97  # Gewinn bei Sturz um -3%
        sl_short = entry * 1.01  # Verlust bei Anstieg um +1%

        for j in range(i + 1, i + 1 + MAX_HOLDING):
            if lows[j] <= tp_short:
                short_outcome[i] = 1
                short_return[i]  = 0.03
                break
            if highs[j] >= sl_short:
                short_outcome[i] = 0
                short_return[i]  = -0.01
                break
        else:
            short_outcome[i] = -1
            short_return[i]  = (entry - closes[i + MAX_HOLDING]) / entry

    df["tb_long_outcome"]  = long_outcome
    df["tb_long_return"]   = long_return
    df["tb_short_outcome"] = short_outcome
    df["tb_short_return"]  = short_return

    # Die Signale des Modells (regime) mit den historischen Ergebnissen verknüpfen
    trade_outcome = np.full(n, np.nan)
    trade_return  = np.full(n, np.nan)

    is_long  = np.isin(regimes, [1, 3])
    is_short = np.isin(regimes, [2, 4])

    trade_outcome[is_long] = long_outcome[is_long]
    trade_return[is_long]  = long_return[is_long]

    trade_outcome[is_short] = short_outcome[is_short]
    trade_return[is_short]  = short_return[is_short]

    df["trade_outcome"] = trade_outcome
    df["trade_return"]  = trade_return

    return df

# ── HAUPTPROGRAMM ─────────────────────────────────────────────────────────────
def main():
    print(f"[INFO] Lade Features aus {INPUT_PATH}...")
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError("features.csv fehlt! Bitte zuerst Skript 03 ausführen.")

    df = pd.read_csv(INPUT_PATH, index_col=0, parse_dates=True)
    df = df.sort_index()

    # Berechne die Prozent-Ausstiege für den Backtest
    df = triple_barrier_labels(df)

    # Features definieren (Alle Spalten außer den rohen Preisen und Labels)
    exclude_cols = [
        "open", "high", "low", "close", "volume", "returns", "direction_6", "regime",
        "tb_long_outcome", "tb_long_return", "tb_short_outcome", "tb_short_return",
        "trade_outcome", "trade_return", "round_level"
    ]
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    # NaN-Zeilen durch die rollenden Fenster sauber löschen
    df_clean = df.dropna(subset=feature_cols + ["regime"])

    # Chronologischer Split (80% Training / 20% Testset für ehrlichen Backtest)
    split_idx = int(len(df_clean) * 0.80)
    df_train = df_clean.iloc[:split_idx]
    df_test  = df_clean.iloc[split_idx:]

    X_train = df_train[feature_cols]
    y_train = df_train["regime"]
    X_test  = df_test[feature_cols]
    y_test  = df_test["regime"]

    print(f"[INFO] Daten aufgeteilt:")
    print(f"       Training: {X_train.shape[0]} Stunden (Zeilen)")
    print(f"       Testset : {X_test.shape[0]} Stunden (Zeilen)")

    # KI-Modell initialisieren und trainieren
    print(f"[INFO] Trainiere Random Forest Modell auf {len(feature_cols)} Features...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Vorhersagen auf dem ungesehenen Testset
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("\n" + "=" * 60 + "\n  MODELL-METRIKEN AUF DEM TESTSET (EVALUIERUNG)\n" + "=" * 60)
    print(f"  Gesamt-Accuracy: {acc*100:.2f}%")
    print("\n[Klassifikations-Report]")
    print(classification_report(y_test, y_pred, zero_division=0))

    # ── HISTORISCHER BACKTEST-REPORT ──────────────────────────────────────────
    # Wir isolieren die Zeilen des Testsets, in denen die KI aktiv getradet hat
    df_backtest = df_test.copy()
    df_backtest["pred_regime"] = y_pred

    # Ein Trade findet statt, wenn das Modell Regime 1, 2, 3 oder 4 vorhersagt
    trades = df_backtest[df_backtest["pred_regime"].isin([1, 2, 3, 4]) & df_backtest["trade_return"].notna()]

    print("\n" + "=" * 60 + "\n  HISTORISCHER VERGANGENHEITS-CHECK (+3% / -1% STRATEGIE)\n" + "=" * 60)
    if trades.empty:
        print("  [WARNUNG] Das Modell hat im Testzeitraum keine Trades ausgelöst.")
        return

    total_trades = len(trades)
    winning_trades = len(trades[trades["trade_outcome"] == 1])
    win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0

    print(f"  Anzahl simulierter Trades : {total_trades}")
    print(f"  Gewonnene Trades (+3.0%)  : {winning_trades}")
    print(f"  Verlorene Trades (-1.0%)  : {len(trades[trades['trade_outcome'] == 0])}")
    print(f"  Zeit-Timeouts (nach 24h)  : {len(trades[trades['trade_outcome'] == -1])}")
    print(f"  Effektive Win-Rate        : {win_rate:.2f}%")

    # Equity Curve berechnen (Zinseszins-Effekt simulieren)
    # Startkapital = 1.0 (entspricht 100%)
    returns_vector = trades["trade_return"].to_numpy()
    equity_curve = np.cumprod(1 + returns_vector)
    final_return = (equity_curve[-1] - 1) * 100 if len(equity_curve) > 0 else 0

    print(f"\n  FINANZIELLES ERGEBNIS:")
    print(f"  Gesamtrendite im Testset  : {final_return:+.2f}%")
    print(f"  Endkapital (Faktor)       : {equity_curve[-1]:.4f}x vom Startkapital")
    print("=" * 60)

    # Equity Curve als Chart visualisieren und abspeichern
    plt.figure(figsize=(12, 6))
    plt.plot(trades.index, equity_curve, label="Unsere 3% / 1% Strategie", color="gold", linewidth=2)
    plt.axhline(1.0, color="red", linestyle="--", alpha=0.5, label="Startkapital")
    plt.title("Historische Kapitalentwicklung (Backtest auf ungesehenem Testset)", fontsize=14, fontweight="bold")
    plt.xlabel("Datum", fontsize=12)
    plt.ylabel("Kapital-Faktor (Start = 1.0)", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)

    chart_path = os.path.join(IMAGE_DIR, "step4_equity_curve.png")
    plt.savefig(chart_path, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Performance-Chart gespeichert unter: {chart_path}")

    # Das trainierte Modell für den Live-Bot (Skript 5) exportieren
    model_path = os.path.join(MODEL_DIR, "regime_rf_model.pkl")
    joblib.dump({"model": model, "feature_cols": feature_cols}, model_path)
    print(f"[INFO] KI-Modell erfolgreich exportiert nach: {model_path}\n")

if __name__ == "__main__":
    main()
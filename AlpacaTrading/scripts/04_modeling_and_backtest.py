"""
04_modeling_and_backtest.py
============================
Schritt 4: KI-Modellierung (Random Forest) & Historischer Backtest + 5 Erweiterte Plots

Dieses Script führt den Vergangenheits-Check durch:
1. Lädt die vorbereitete 'data/features.csv' aus Schritt 3.
2. Führt einen chronologischen Split durch (80% Training, 20% Testset).
3. Trainiert einen Random Forest Classifier auf den stationären Features.
4. Berechnet den historischen Backtest basierend auf FESTEN PROZENTGRENZEN:
   - Take-Profit (TP): +3.0% | - Stop-Loss (SL): -1.0% | - Limit: 24h
5. Nutzt ein realistisches 2% Fixed-Fractional Risk Management (Start: 100.000 USD).
6. Generiert 5 professionelle Quant-Handelsdiagramme im Ordner artifacts/images/.
"""

# ── IMPORTS ──────────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
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

REGIME_LABELS = {
    0: "Low Vol -> HOLD",
    1: "Medium Vol Up -> BUY",
    2: "Medium Vol Down -> SELL",
    3: "High Vol Up -> STRONG BUY",
    4: "High Vol Down -> STRONG SELL",
}

# ── FESTE PROZENT-STRATEGIE (TRIPLE-BARRIER) ──────────────────────────────────
def triple_barrier_labels(df: pd.DataFrame) -> pd.DataFrame:
    closes  = df["close"].to_numpy()
    highs   = df["high"].to_numpy()
    lows    = df["low"].to_numpy()
    regimes = df["regime"].to_numpy()
    n = len(df)

    long_outcome   = np.full(n, np.nan)
    long_return    = np.full(n, np.nan)
    long_duration  = np.full(n, np.nan)
    short_outcome  = np.full(n, np.nan)
    short_return   = np.full(n, np.nan)
    short_duration = np.full(n, np.nan)

    print(f"[INFO] Berechne feste Prozent-Barrieren (TP=+3%, SL=-1%, Limit={MAX_HOLDING}h)...")

    for i in range(n - MAX_HOLDING):
        entry = closes[i]

        # ── LONG SIMULATION (+3% / -1%) ──
        tp_long = entry * 1.03
        sl_long = entry * 0.99

        for j in range(i + 1, i + 1 + MAX_HOLDING):
            if highs[j] >= tp_long:
                long_outcome[i]  = 1
                long_return[i]   = 0.03
                long_duration[i] = j - i
                break
            if lows[j] <= sl_long:
                long_outcome[i]  = 0
                long_return[i]   = -0.01
                long_duration[i] = j - i
                break
        else:
            long_outcome[i]  = -1
            long_return[i]   = (closes[i + MAX_HOLDING] - entry) / entry
            long_duration[i] = MAX_HOLDING

        # ── SHORT SIMULATION (+3% / -1%) ──
        tp_short = entry * 0.97
        sl_short = entry * 1.01

        for j in range(i + 1, i + 1 + MAX_HOLDING):
            if lows[j] <= tp_short:
                short_outcome[i]  = 1
                short_return[i]   = 0.03
                short_duration[i] = j - i
                break
            if highs[j] >= sl_short:
                short_outcome[i]  = 0
                short_return[i]   = -0.01
                short_duration[i] = j - i
                break
        else:
            short_outcome[i]  = -1
            short_return[i]   = (entry - closes[i + MAX_HOLDING]) / entry
            short_duration[i] = MAX_HOLDING

    df["tb_long_outcome"]  = long_outcome
    df["tb_long_return"]   = long_return
    df["tb_short_outcome"] = short_outcome
    df["tb_short_return"]  = short_return

    # Mappen auf das Signal
    trade_outcome  = np.full(n, np.nan)
    trade_return   = np.full(n, np.nan)
    trade_duration = np.full(n, np.nan)

    is_long  = np.isin(regimes, [1, 3])
    is_short = np.isin(regimes, [2, 4])

    trade_outcome[is_long]  = long_outcome[is_long]
    trade_return[is_long]   = long_return[is_long]
    trade_duration[is_long] = long_duration[is_long]

    trade_outcome[is_short]  = short_outcome[is_short]
    trade_return[is_short]   = short_return[is_short]
    trade_duration[is_short] = short_duration[is_short]

    df["trade_outcome"]  = trade_outcome
    df["trade_return"]   = trade_return
    df["trade_duration"] = trade_duration

    return df

# ── HAUPTPROGRAMM ─────────────────────────────────────────────────────────────
def main():
    print(f"[INFO] Lade Features aus {INPUT_PATH}...")
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError("features.csv fehlt! Bitte zuerst Skript 03 ausführen.")

    df = pd.read_csv(INPUT_PATH, index_col=0, parse_dates=True).sort_index()
    df = triple_barrier_labels(df)

    exclude_cols = [
        "open", "high", "low", "close", "volume", "returns", "direction_6", "regime",
        "tb_long_outcome", "tb_long_return", "tb_short_outcome", "tb_short_return",
        "trade_outcome", "trade_return", "trade_duration", "round_level"
    ]
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    df_clean = df.dropna(subset=feature_cols + ["regime"])

    split_idx = int(len(df_clean) * 0.80)
    df_train = df_clean.iloc[:split_idx]
    df_test  = df_clean.iloc[split_idx:]

    X_train, y_train = df_train[feature_cols], df_train["regime"]
    X_test, y_test   = df_test[feature_cols], df_test["regime"]

    print(f"[INFO] Trainiere Random Forest Modell auf {len(feature_cols)} Features...")
    model = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("\n" + "=" * 60 + "\n  MODELL-METRIKEN AUF DEM TESTSET (EVALUIERUNG)\n" + "=" * 60)
    print(f"  Gesamt-Accuracy: {acc*100:.2f}%")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Backtest DataFrame isolieren
    df_backtest = df_test.copy()
    df_backtest["pred_regime"] = y_pred
    trades = df_backtest[df_backtest["pred_regime"].isin([1, 2, 3, 4]) & df_backtest["trade_return"].notna()].copy()

    if trades.empty:
        print("[WARNUNG] Keine Trades ausgelöst."); return

    total_trades = len(trades)
    winning_trades = len(trades[trades["trade_outcome"] == 1])
    win_rate = (winning_trades / total_trades) * 100

    print("\n" + "=" * 60 + "\n  HISTORISCHER VERGANGENHEITS-CHECK\n" + "=" * 60)
    print(f"  Anzahl Trades             : {total_trades}")
    print(f"  Gewonnene Trades (+3.0%)  : {winning_trades}")
    print(f"  Verlorene Trades (-1.0%)  : {len(trades[trades['trade_outcome'] == 0])}")
    print(f"  Timeouts (nach 24h)       : {len(trades[trades['trade_outcome'] == -1])}")
    print(f"  Effektive Win-Rate        : {win_rate:.2f}%")

    # ── REALISTISCHE PORTFOLIO-BERECHNUNG (2% FIXED RISK MANAGEMENT) ──
    START_KAPITAL = 100000.0  # Startkapital: 100k USD
    RISK_PER_TRADE = 0.02     # Wir riskieren max. 2% des Portfolios pro Trade

    portfolio_wert = START_KAPITAL
    equity_curve = []

    for r in trades["trade_return"].to_numpy():
        trade_effekt = portfolio_wert * RISK_PER_TRADE * r
        portfolio_wert += trade_effekt
        equity_curve.append(portfolio_wert)

    equity_curve = np.array(equity_curve)
    final_return = ((portfolio_wert - START_KAPITAL) / START_KAPITAL) * 100

    print(f"\n  FINANZIELLES ERGEBNIS BEI EINEM REAlISTISCHEN PORTFOLIO:")
    print(f"  Virtuelles Startkapital   : {START_KAPITAL:,.2f} USD")
    print(f"  Endgültiger Portfolio-Wert: {portfolio_wert:,.2f} USD")
    print(f"  Reale Gesamtrendite       : {final_return:+.2f}%")
    print("=" * 60)

    # 📈 COMPILING PLOTS (Aktualisiert mit neuem Kurs-Vektor)
    # PLOT 1: Equity Curve (In echten Dollar)
    plt.figure(figsize=(10, 5))
    plt.plot(trades.index, equity_curve, color="gold", linewidth=2, label="3% / 1% Strategie (2% Risk)")
    plt.axhline(START_KAPITAL, color="red", linestyle="--", alpha=0.5, label="Startkapital")
    plt.title("Reale Portfolioentwicklung (100k USD Startkapital)", fontsize=12, fontweight="bold")
    plt.ylabel("Depotwert (USD)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(IMAGE_DIR, "step4_equity_curve.png"), bbox_inches="tight")
    plt.close()

    # PLOT 2: Max Drawdown Plot
    running_max = np.maximum.accumulate(equity_curve)
    drawdowns = (equity_curve - running_max) / running_max * 100
    plt.figure(figsize=(10, 4))
    plt.fill_between(trades.index, drawdowns, color="crimson", alpha=0.4, label="Drawdown %")
    plt.axhline(0, color="black", linestyle="-", alpha=0.3)
    plt.title("Historischer Drawdown (Risiko-Analyse)", fontsize=12, fontweight="bold")
    plt.ylabel("Verlust vom Peak (%)")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(IMAGE_DIR, "step4_drawdown.png"), bbox_inches="tight")
    plt.close()

    # PLOT 3: Hit-Rate per Regime (FutureWarning gefixt via hue-Zuweisung)
    regime_stats = trades.groupby("pred_regime")["trade_outcome"].apply(lambda x: (x == 1).sum() / len(x) * 100)
    plt.figure(figsize=(8, 4))
    regime_x = [f"Regime {int(i)}" for i in regime_stats.index]
    sns.barplot(x=regime_x, y=regime_stats.values, hue=regime_x, palette="viridis", legend=False)
    plt.axhline(25.0, color="red", linestyle="--", label="Gewinnschwelle (25% bei CRV 3)")
    plt.title("Win-Rate aufgeteilt nach vorhergesagten Marktregimes", fontsize=12, fontweight="bold")
    plt.ylabel("Win-Rate (%)")
    plt.legend()
    plt.savefig(os.path.join(IMAGE_DIR, "step4_hit_rate_per_regime.png"), bbox_inches="tight")
    plt.close()

    # PLOT 4: Trade-Duration Histogram
    plt.figure(figsize=(8, 4))
    plt.hist(trades["trade_duration"], bins=int(MAX_HOLDING), color="royalblue", edgecolor="black", alpha=0.7)
    plt.title("Verteilung der Trade-Dauer (Wann erfolgt der Ausstieg?)", fontsize=12, fontweight="bold")
    plt.xlabel("Haltedauer in Stunden")
    plt.ylabel("Anzahl Trades")
    plt.grid(True, alpha=0.2)
    plt.savefig(os.path.join(IMAGE_DIR, "step4_trade_duration_histogram.png"), bbox_inches="tight")
    plt.close()

    # PLOT 5: Confusion Matrix Heatmap
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=[f"P:{i}" for i in range(5)],
                yticklabels=[f"T:{i}" for i in range(5)])
    plt.title("Confusion Matrix (KI-Fehleranalyse)\nT=True, P=Predicted", fontsize=12, fontweight="bold")
    plt.savefig(os.path.join(IMAGE_DIR, "step4_confusion_matrix.png"), bbox_inches="tight")
    plt.close()

    print(f"[INFO] ✅ Alle 5 Analyse-Plots erfolgreich unter '{IMAGE_DIR}' abgelegt!")

    # Export des Modells
    joblib.dump({"model": model, "feature_cols": feature_cols}, os.path.join(MODEL_DIR, "regime_rf_model.pkl"))
    print(f"[INFO] KI-Modell exportiert.\n")

if __name__ == "__main__":
    main()
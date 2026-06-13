"""
04_modeling_and_backtest.py
============================
Schritt 4 (MVP): Y-Variablen, Train/Test-Split, Quick-Modell & Rough Backtest

Dieses Script ist der erste End-to-End-Prototyp:
1. Lädt features.csv
2. Berechnet ATR (Average True Range) als zusätzliches Feature
3. Definiert die Y-Variablen:
       - regime        (0-4, bereits vorhanden – primäres Klassifikationsziel)
       - trade_outcome (Triple-Barrier-Label: 1=Win, 0=Loss, -1=Timeout)
4. Chronologischer Train/Test-Split (80/20)
5. Trainiert einen Random Forest Classifier (Standardparameter, kein Tuning)
   auf 'regime'
6. Führt einen groben Backtest durch: Wenn das Modell Regime 1/3 (Long) oder
   2/4 (Short) vorhersagt, wird die Triple-Barrier-Outcome der jeweiligen
   Richtung verwendet um den Trade-Ausgang zu bestimmen.

WICHTIG – TRIPLE BARRIER METHODE:
    Für jede Stunde wird simuliert: "Was wäre wenn ich hier eine Position
    eröffne?"
        Take-Profit = Einstieg ± 1.5 * ATR
        Stop-Loss   = Einstieg ∓ 1.0 * ATR
        Zeitlimit   = 24 Stunden
    Welche Barriere wird zuerst erreicht?
        TP zuerst   -> 1 (Win)
        SL zuerst   -> 0 (Loss)
        Zeitlimit   -> -1 (Timeout, Exit zum dann aktuellen Preis)

Dies ist ein ROUGH/MVP Test. Feature Selection, Hyperparameter-Tuning und
ein sauberer Backtest (Sharpe, Drawdown) folgen in späteren Scripts erst
wenn die Grundidee hier vielversprechend ist.

Ausführung:
    python scripts/04_modeling_and_backtest.py

Input:
    data/features.csv

Output:
    data/train.csv, data/test.csv
    artifacts/models/regime_rf_model.pkl
    artifacts/images/step4_confusion_matrix.png
    artifacts/images/step4_feature_importance.png
    artifacts/images/step4_equity_curve.png
"""

# ── IMPORTS ──────────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ── KONFIGURATION ─────────────────────────────────────────────────────────────
COLOR_PRICE = "#185FA5"
COLOR_BUY   = "#1D9E75"
COLOR_SELL  = "#E24B4A"

# Triple-Barrier Parameter
ATR_PERIOD     = 14    # Perioden für ATR-Berechnung
TP_MULTIPLIER  = 1.5   # Take-Profit = Einstieg ± 1.5 * ATR
SL_MULTIPLIER  = 1.0   # Stop-Loss   = Einstieg ∓ 1.0 * ATR
MAX_HOLDING    = 24    # Max. Haltedauer in Stunden (Zeitlimit)

# Train/Test Split
TRAIN_RATIO = 0.8

# Pfade
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(BASE_DIR, "data")
IMAGES_DIR  = os.path.join(BASE_DIR, "artifacts", "images")
MODELS_DIR  = os.path.join(BASE_DIR, "artifacts", "models")
INPUT_CSV   = os.path.join(DATA_DIR, "features.csv")

REGIME_LABELS = {
    0: "Low Vol (HOLD)",
    1: "Med Vol Up (BUY)",
    2: "Med Vol Down (SELL)",
    3: "High Vol Up (STRONG BUY)",
    4: "High Vol Down (STRONG SELL)",
}

# Spalten die NICHT als Modell-Input (X) verwendet werden
EXCLUDE_FROM_X = [
    "open", "high", "low", "close", "volume", "trade_count", "vwap",
    "returns", "direction_6", "regime", "trade_outcome",
    "tb_long_outcome", "tb_long_return", "tb_short_outcome", "tb_short_return",
]


# ── VERZEICHNISSE ─────────────────────────────────────────────────────────────
def ensure_directories():
    """Erstellt benötigte Verzeichnisse falls nicht vorhanden."""
    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)


# ── DATEN LADEN ───────────────────────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    """
    Lädt die Feature-Daten aus features.csv.

    Returns:
        pd.DataFrame: Feature-DataFrame mit datetime Index.
    """
    print(f"[INFO] Lade Daten aus {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV, index_col=0, parse_dates=True)
    print(f"[INFO] Geladen: {df.shape[0]:,} Zeilen, {df.shape[1]} Spalten")
    return df


# ── ATR BERECHNEN ──────────────────────────────────────────────────────────────
def add_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.DataFrame:
    """
    Berechnet den Average True Range (ATR) – ein Standardmaß für Volatilität
    in absoluten Preiseinheiten (USD). Wird sowohl als Feature als auch für
    die Triple-Barrier-Berechnung benötigt.

    True Range = max(
        high - low,
        |high - close_vorher|,
        |low - close_vorher|
    )
    ATR = rollender Durchschnitt der True Range über `period` Perioden.

    Args:
        df: DataFrame mit OHLC-Daten.
        period: Anzahl Perioden für den rollenden Durchschnitt.

    Returns:
        pd.DataFrame: DataFrame mit zusätzlicher 'ATR' Spalte.
    """
    high_low   = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift(1)).abs()
    low_close  = (df["low"]  - df["close"].shift(1)).abs()

    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["ATR"] = true_range.rolling(period).mean()

    return df


# ── TRIPLE BARRIER LABELING ────────────────────────────────────────────────────
def triple_barrier_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Berechnet für JEDE Stunde zwei unabhängige Triple-Barrier-Outcomes:
    einmal für eine LONG-Position und einmal für eine SHORT-Position.
    Beide werden unabhängig von 'regime' berechnet, da im Backtest die vom
    MODELL vorhergesagte Richtung über Long/Short entscheidet – nicht das
    historische Regime-Label.

    LONG:  TP = Einstieg + TP_MULTIPLIER * ATR
           SL = Einstieg - SL_MULTIPLIER * ATR
           Win wenn 'high' innerhalb MAX_HOLDING Stunden zuerst TP erreicht.

    SHORT: TP = Einstieg - TP_MULTIPLIER * ATR
           SL = Einstieg + SL_MULTIPLIER * ATR
           Win wenn 'low' innerhalb MAX_HOLDING Stunden zuerst TP erreicht.

    Outcome-Codierung:
        1  = Take-Profit zuerst erreicht (Win)
        0  = Stop-Loss zuerst erreicht (Loss)
        -1 = Zeitlimit erreicht, Exit zum Preis nach MAX_HOLDING Stunden

    Zusätzlich wird 'trade_outcome' (Y-Variable) gesetzt:
        - Für regime 1/3 (Long-Signal): tb_long_outcome
        - Für regime 2/4 (Short-Signal): tb_short_outcome
        - Für regime 0: NaN (kein Trade)

    Args:
        df: DataFrame mit close, high, low, ATR, regime.

    Returns:
        pd.DataFrame: DataFrame mit zusätzlichen Spalten:
            tb_long_outcome, tb_long_return,
            tb_short_outcome, tb_short_return,
            trade_outcome
    """
    closes  = df["close"].to_numpy()
    highs   = df["high"].to_numpy()
    lows    = df["low"].to_numpy()
    atr     = df["ATR"].to_numpy()
    regimes = df["regime"].to_numpy()
    n = len(df)

    long_outcome  = np.full(n, np.nan)
    long_return   = np.full(n, np.nan)
    short_outcome = np.full(n, np.nan)
    short_return  = np.full(n, np.nan)

    print(f"[INFO] Berechne Triple-Barrier Labels (TP={TP_MULTIPLIER}x ATR, "
          f"SL={SL_MULTIPLIER}x ATR, Zeitlimit={MAX_HOLDING}h)...")

    for i in range(n - MAX_HOLDING):
        a = atr[i]
        entry = closes[i]

        if np.isnan(a) or a == 0:
            continue

        # ── LONG Simulation ──────────────────────────────────────────────
        tp_long = entry + TP_MULTIPLIER * a
        sl_long = entry - SL_MULTIPLIER * a

        for j in range(i + 1, i + 1 + MAX_HOLDING):
            if highs[j] >= tp_long:
                long_outcome[i] = 1
                long_return[i]  = (tp_long - entry) / entry
                break
            if lows[j] <= sl_long:
                long_outcome[i] = 0
                long_return[i]  = (sl_long - entry) / entry
                break
        else:
            long_outcome[i] = -1
            long_return[i]  = (closes[i + MAX_HOLDING] - entry) / entry

        # ── SHORT Simulation ─────────────────────────────────────────────
        tp_short = entry - TP_MULTIPLIER * a
        sl_short = entry + SL_MULTIPLIER * a

        for j in range(i + 1, i + 1 + MAX_HOLDING):
            if lows[j] <= tp_short:
                short_outcome[i] = 1
                short_return[i]  = (entry - tp_short) / entry
                break
            if highs[j] >= sl_short:
                short_outcome[i] = 0
                short_return[i]  = (entry - sl_short) / entry
                break
        else:
            short_outcome[i] = -1
            short_return[i]  = (entry - closes[i + MAX_HOLDING]) / entry

    df["tb_long_outcome"]  = long_outcome
    df["tb_long_return"]   = long_return
    df["tb_short_outcome"] = short_outcome
    df["tb_short_return"]  = short_return

    # trade_outcome: Y-Variable basierend auf dem HISTORISCHEN Regime-Signal
    # Long-Regimes: 1 (Med Vol Up), 3 (High Vol Up)
    # Short-Regimes: 2 (Med Vol Down), 4 (High Vol Down)
    # Regime 0: kein Trade -> NaN
    trade_outcome = np.full(n, np.nan)
    is_long  = np.isin(regimes, [1, 3])
    is_short = np.isin(regimes, [2, 4])
    trade_outcome[is_long]  = long_outcome[is_long]
    trade_outcome[is_short] = short_outcome[is_short]
    df["trade_outcome"] = trade_outcome

    return df


# ── TRAIN/TEST SPLIT ────────────────────────────────────────────────────────────
def chronological_split(df: pd.DataFrame):
    """
    Teilt den DataFrame chronologisch in Train (80%) und Test (20%) auf.
    Kein zufälliger Split, da Zeitreihendaten zeitliche Abhängigkeiten haben.

    Die letzten MAX_HOLDING Zeilen werden entfernt, da für sie keine
    Triple-Barrier-Outcomes berechnet werden konnten (NaN).

    Args:
        df: Vollständiger Feature-DataFrame.

    Returns:
        tuple: (df_train, df_test)
    """
    # Zeilen ohne gültige Triple-Barrier-Outcomes entfernen (letzte MAX_HOLDING Zeilen)
    df = df.iloc[:-MAX_HOLDING].copy()

    split_idx = int(len(df) * TRAIN_RATIO)
    df_train = df.iloc[:split_idx].copy()
    df_test  = df.iloc[split_idx:].copy()

    print(f"\n[INFO] Train/Test Split (chronologisch, {TRAIN_RATIO:.0%}/{1-TRAIN_RATIO:.0%}):")
    print(f"  Train: {df_train.index[0]} bis {df_train.index[-1]}  ({len(df_train):,} Zeilen)")
    print(f"  Test:  {df_test.index[0]} bis {df_test.index[-1]}  ({len(df_test):,} Zeilen)")

    return df_train, df_test


# ── MODELL TRAINING ────────────────────────────────────────────────────────────
def train_model(df_train: pd.DataFrame):
    """
    Trainiert einen Random Forest Classifier auf 'regime' mit Standardparametern
    (kein Hyperparameter-Tuning – das folgt erst nach erfolgreichem MVP-Test).

    class_weight='balanced' kompensiert die ungleiche Verteilung der Regime-Klassen
    (Regime 0 ist deutlich häufiger als 3/4).

    Args:
        df_train: Trainings-DataFrame.

    Returns:
        tuple: (model, feature_columns)
    """
    feature_cols = [c for c in df_train.columns if c not in EXCLUDE_FROM_X and c != "ATR"]
    feature_cols = feature_cols + ["ATR"]   # ATR explizit als Feature mit aufnehmen

    X_train = df_train[feature_cols]
    y_train = df_train["regime"]

    print(f"\n[INFO] Trainiere Random Forest Classifier auf 'regime'...")
    print(f"  Anzahl Features: {len(feature_cols)}")
    print(f"  Trainingsbeispiele: {len(X_train):,}")

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    return model, feature_cols


# ── MODELL EVALUATION ───────────────────────────────────────────────────────────
def evaluate_model(model, df_test: pd.DataFrame, feature_cols: list):
    """
    Evaluiert das Modell auf dem Testset: Accuracy, Classification Report,
    Confusion Matrix.

    Args:
        model: Trainiertes Modell.
        df_test: Test-DataFrame.
        feature_cols: Liste der Feature-Spaltennamen.

    Returns:
        np.ndarray: Vorhergesagte Regime-Klassen für das Testset.
    """
    X_test = df_test[feature_cols]
    y_test = df_test["regime"]

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    print("\n" + "=" * 60)
    print("  MODELL EVALUATION (Testset)")
    print("=" * 60)
    print(f"\n[ACCURACY] {acc:.4f} ({acc * 100:.2f}%)")

    print(f"\n[CLASSIFICATION REPORT]")
    target_names = [REGIME_LABELS[i] for i in sorted(y_test.unique())]
    print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))

    # Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_pred, labels=sorted(y_test.unique()))
    fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=[f"Regime {i}" for i in sorted(y_test.unique())],
                yticklabels=[f"Regime {i}" for i in sorted(y_test.unique())],
                ax=ax, cbar=True)
    ax.set_xlabel("Vorhergesagt", fontsize=10)
    ax.set_ylabel("Tatsächlich", fontsize=10)
    ax.set_title("Confusion Matrix – Regime Klassifikation (Testset)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(IMAGES_DIR, "step4_confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\n[INFO] Gespeichert: {path}")

    return y_pred


# ── FEATURE IMPORTANCE ──────────────────────────────────────────────────────────
def plot_feature_importance(model, feature_cols: list, top_n: int = 20):
    """
    Plottet die Top-N wichtigsten Features laut Random Forest Feature Importance.

    Hilft bei der späteren Feature Selection: Welche Features tragen am
    meisten zur Regime-Klassifikation bei?

    Args:
        model: Trainiertes Modell.
        feature_cols: Liste der Feature-Spaltennamen.
        top_n: Anzahl der Top-Features die angezeigt werden.
    """
    importances = pd.Series(model.feature_importances_, index=feature_cols)
    importances = importances.sort_values(ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    importances.sort_values().plot(kind="barh", color=COLOR_PRICE, ax=ax)
    ax.set_title(f"Top {top_n} Feature Importance – Random Forest",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Importance", fontsize=10)
    ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(IMAGES_DIR, "step4_feature_importance.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[INFO] Gespeichert: {path}")

    print(f"\n[TOP {top_n} FEATURES]")
    for i, (feat, imp) in enumerate(importances.sort_values(ascending=False).items(), 1):
        print(f"  {i:>2}. {feat:<25} {imp:.4f}")


# ── ROUGH BACKTEST ───────────────────────────────────────────────────────────────
def run_backtest(df_test: pd.DataFrame, y_pred: np.ndarray):
    """
    Grober Backtest auf dem Testset:

    Für jede Stunde wo das Modell ein Long-Signal (Regime 1/3) oder
    Short-Signal (Regime 2/4) vorhersagt, wird die ENTSPRECHENDE
    Triple-Barrier-Outcome (tb_long_* bzw. tb_short_*) verwendet um zu
    bestimmen ob der Trade gewonnen oder verloren hätte.

    Bei Regime 0 (HOLD) wird kein Trade eröffnet.

    Berechnet:
        - Anzahl Trades
        - Win-Rate
        - Gesamtrendite (Summe der einzelnen Trade-Returns)
        - Equity-Kurve (kumulativ, Startkapital = 1.0)

    Args:
        df_test: Test-DataFrame mit tb_long_*/tb_short_* Spalten.
        y_pred: Vorhergesagte Regime-Klassen.

    Returns:
        pd.DataFrame: DataFrame mit Trade-Details für die Equity-Kurve.
    """
    df_bt = df_test.copy()
    df_bt["predicted_regime"] = y_pred

    trade_returns = []
    trade_outcomes = []
    trade_dates = []

    for idx, row in df_bt.iterrows():
        pred = row["predicted_regime"]

        if pred in (1, 3):     # Long-Signal
            ret = row["tb_long_return"]
            out = row["tb_long_outcome"]
        elif pred in (2, 4):   # Short-Signal
            ret = row["tb_short_return"]
            out = row["tb_short_outcome"]
        else:                  # Regime 0 -> kein Trade
            continue

        if pd.isna(ret):
            continue

        trade_returns.append(ret)
        trade_outcomes.append(out)
        trade_dates.append(idx)

    n_trades = len(trade_returns)

    if n_trades == 0:
        print("\n[WARNUNG] Keine Trades im Backtest – Modell sagt nur Regime 0 vorher.")
        return pd.DataFrame()

    trade_returns = np.array(trade_returns)
    trade_outcomes = np.array(trade_outcomes)

    n_wins   = np.sum(trade_outcomes == 1)
    n_losses = np.sum(trade_outcomes == 0)
    n_timeout = np.sum(trade_outcomes == -1)
    win_rate = n_wins / n_trades

    total_return_sum = trade_returns.sum()
    avg_return = trade_returns.mean()

    # Equity-Kurve: kumulatives Produkt (1 + return), Startkapital = 1.0
    equity = (1 + trade_returns).cumprod()
    final_equity = equity[-1]
    total_return_compound = (final_equity - 1) * 100

    print("\n" + "=" * 60)
    print("  ROUGH BACKTEST – TESTSET")
    print("=" * 60)
    print(f"\n[TRADES]")
    print(f"  Anzahl Trades   : {n_trades:,}")
    print(f"  Wins (TP)       : {n_wins:,} ({n_wins/n_trades*100:.1f}%)")
    print(f"  Losses (SL)     : {n_losses:,} ({n_losses/n_trades*100:.1f}%)")
    print(f"  Timeouts        : {n_timeout:,} ({n_timeout/n_trades*100:.1f}%)")
    print(f"  Win-Rate        : {win_rate*100:.2f}%")

    print(f"\n[RENDITE]")
    print(f"  Durchschnittliche Rendite pro Trade : {avg_return*100:.4f}%")
    print(f"  Summe aller Trade-Returns           : {total_return_sum*100:.2f}%")
    print(f"  Gesamtrendite (compound, equity)    : {total_return_compound:.2f}%")
    print(f"  Endkapital (Start = 1.0)            : {final_equity:.4f}")

    if total_return_compound > 0:
        print(f"\n  ✅ Strategie wäre PROFITABEL gewesen (+{total_return_compound:.2f}%)")
    else:
        print(f"\n  ❌ Strategie wäre VERLUSTREICH gewesen ({total_return_compound:.2f}%)")

    print("=" * 60)

    # Equity-Kurve Plot
    df_equity = pd.DataFrame({"equity": equity}, index=trade_dates)

    fig, ax = plt.subplots(figsize=(12, 5), dpi=150)
    ax.plot(df_equity.index, df_equity["equity"], color=COLOR_PRICE, linewidth=1.2)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1.0, label="Break-Even")
    ax.fill_between(df_equity.index, 1.0, df_equity["equity"],
                     where=df_equity["equity"] >= 1.0, alpha=0.2, color=COLOR_BUY)
    ax.fill_between(df_equity.index, 1.0, df_equity["equity"],
                     where=df_equity["equity"] < 1.0, alpha=0.2, color=COLOR_SELL)

    ax.set_title("Equity-Kurve – Rough Backtest (Testset)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Datum", fontsize=10)
    ax.set_ylabel("Kapital (Start = 1.0)", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(IMAGES_DIR, "step4_equity_curve.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\n[INFO] Gespeichert: {path}")

    return df_equity


# ── HAUPTPROGRAMM ─────────────────────────────────────────────────────────────
def main():
    """Hauptfunktion: Y-Variablen definieren, Modell trainieren, Backtest durchführen."""
    ensure_directories()

    # Daten laden
    df = load_data()

    # ATR berechnen
    print("\n[INFO] Berechne ATR...")
    df = add_atr(df)

    # Triple-Barrier Labels berechnen (Y-Variablen)
    df = triple_barrier_labels(df)

    # NaN-Zeilen entfernen (durch ATR-Rolling und letzte MAX_HOLDING Zeilen)
    df = df.dropna(subset=["ATR"])

    # Train/Test Split
    df_train, df_test = chronological_split(df)

    # Speichern
    df_train.to_csv(os.path.join(DATA_DIR, "train.csv"))
    df_test.to_csv(os.path.join(DATA_DIR, "test.csv"))
    print(f"\n[INFO] Gespeichert: data/train.csv, data/test.csv")

    # Trade-Outcome Verteilung anzeigen
    print(f"\n[TRADE_OUTCOME VERTEILUNG (gesamt)]")
    print(df["trade_outcome"].value_counts(dropna=False).sort_index().to_string())

    # Modell trainieren
    model, feature_cols = train_model(df_train)

    # Modell speichern
    model_path = os.path.join(MODELS_DIR, "regime_rf_model.pkl")
    joblib.dump({"model": model, "feature_cols": feature_cols}, model_path)
    print(f"\n[INFO] Modell gespeichert: {model_path}")

    # Evaluation
    y_pred = evaluate_model(model, df_test, feature_cols)

    # Feature Importance
    plot_feature_importance(model, feature_cols)

    # Rough Backtest
    run_backtest(df_test, y_pred)

    print("\n✅ Schritt 4 (MVP) abgeschlossen")


if __name__ == "__main__":
    main()

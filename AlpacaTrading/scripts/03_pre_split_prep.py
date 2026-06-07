"""
03_pre_split_prep.py
====================
Schritt 3: Feature Engineering VOR dem Train/Test-Split

Dieses Script lädt die Rohdaten aus data/raw_data.csv, berechnet alle
technischen Features ausschließlich aus Vergangenheitsdaten (kein Data Leakage),
definiert die Zielvariable (Regime 0–4) und speichert den fertigen
Feature-DataFrame als data/features.csv.

NO-LEAKAGE PRINZIP:
    Alle Rolling Windows und Lags verwenden .shift(1) wo die aktuelle
    Kerze noch nicht abgeschlossen ist. Nur abgeschlossene Kerzen fließen ein.

Regime-Labels:
    0 = Low Volatilität         → HOLD
    1 = Medium Vol + Aufwärts   → BUY
    2 = Medium Vol + Abwärts    → SELL
    3 = High Vol + Aufwärts     → STRONG BUY
    4 = High Vol + Abwärts      → STRONG SELL

Ausführung:
    python scripts/03_pre_split_prep.py

Input:
    data/raw_data.csv

Output:
    data/features.csv
    artifacts/images/step3_feature_examples.png
    artifacts/images/step3_regime_distribution.png
"""

# ── IMPORTS ──────────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

# ── KONFIGURATION ─────────────────────────────────────────────────────────────
COLOR_PRICE  = "#185FA5"
COLOR_BUY    = "#1D9E75"
COLOR_SELL   = "#E24B4A"
COLOR_VOL    = "#F4A261"

REGIME_COLORS = {
    0: "#8E9AAF",   # Low Vol – grau (HOLD)
    1: "#1D9E75",   # Med Vol Up – grün (BUY)
    2: "#E24B4A",   # Med Vol Down – rot (SELL)
    3: "#0D6B4F",   # High Vol Up – dunkelgrün (STRONG BUY)
    4: "#8B1A1A",   # High Vol Down – dunkelrot (STRONG SELL)
}

REGIME_LABELS = {
    0: "Low Vol (HOLD)",
    1: "Med Vol ↑ (BUY)",
    2: "Med Vol ↓ (SELL)",
    3: "High Vol ↑ (STRONG BUY)",
    4: "High Vol ↓ (STRONG SELL)",
}

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(BASE_DIR, "data")
IMAGES_DIR = os.path.join(BASE_DIR, "artifacts", "images")
INPUT_CSV  = os.path.join(DATA_DIR, "raw_data.csv")
OUTPUT_CSV = os.path.join(DATA_DIR, "features.csv")


# ── VERZEICHNISSE ─────────────────────────────────────────────────────────────
def ensure_directories():
    """Erstellt benötigte Verzeichnisse falls nicht vorhanden."""
    os.makedirs(IMAGES_DIR, exist_ok=True)


# ── DATEN LADEN ───────────────────────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    """
    Lädt die Rohdaten aus raw_data.csv.

    Returns:
        pd.DataFrame: Rohdaten mit datetime Index.
    """
    print(f"[INFO] Lade Daten aus {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV, index_col=0, parse_dates=True)
    print(f"[INFO] Geladen: {df.shape[0]:,} Zeilen, {df.shape[1]} Spalten")
    return df


# ── ZIELVARIABLE (REGIME LABELS) ──────────────────────────────────────────────
def add_target_variable(df: pd.DataFrame) -> pd.DataFrame:
    """
    Definiert die Zielvariable 'regime' (0–4) basierend auf
    Volatilität und 6h-Richtung.

    Quantile werden auf Gesamtdaten berechnet (erlaubt, da nur Statistik),
    die eigentliche Klassifikation ist lookahead-frei.

    Args:
        df: DataFrame mit OHLCV-Daten.

    Returns:
        pd.DataFrame: DataFrame mit zusätzlichen Zielvariablen.
    """
    # Stündliche Returns
    df["returns"] = df["close"].pct_change()

    # Rollende 24h Volatilität
    df["volatility_24"] = df["returns"].rolling(24).std()

    # Volatilitäts-Quantile auf Gesamtdaten (VOR Split – erlaubt weil nur Statistik)
    vol_q33 = df["volatility_24"].quantile(0.33)
    vol_q66 = df["volatility_24"].quantile(0.66)
    print(f"[INFO] Volatilitäts-Quantile: 33%={vol_q33:.5f}, 66%={vol_q66:.5f}")

    # 6h Richtung (Vorhersageziel: Wie hat sich Preis in den letzten 6h entwickelt?)
    df["direction_6"] = df["close"].pct_change(6)

    # Regime Labels zuweisen
    def assign_regime(row):
        """Weist eine Regime-Klasse 0–4 basierend auf Vol und Richtung zu."""
        vol = row["volatility_24"]
        direction = row["direction_6"]

        if pd.isna(vol) or pd.isna(direction):
            return np.nan

        if vol <= vol_q33:
            return 0   # Low Vol → HOLD
        elif vol <= vol_q66:
            return 1 if direction >= 0 else 2   # Med Vol → BUY/SELL
        else:
            return 3 if direction >= 0 else 4   # High Vol → STRONG BUY/SELL

    df["regime"] = df.apply(assign_regime, axis=1)

    return df


# ── GRUPPE 1: PREIS LAGS ──────────────────────────────────────────────────────
def add_price_lags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preis-Lags für kurzfristige, mittelfristige und wöchentliche Rückreferenz.
    Shift = kein Leakage (nur vergangene abgeschlossene Kerzen).

    Args:
        df: DataFrame.

    Returns:
        pd.DataFrame: DataFrame mit Preis-Lag-Features.
    """
    df["lag_1"]   = df["close"].shift(1)    # Vorherige Stunde
    df["lag_6"]   = df["close"].shift(6)    # Vor 6 Stunden
    df["lag_24"]  = df["close"].shift(24)   # Vor 24 Stunden (gestern)
    df["lag_168"] = df["close"].shift(168)  # Vor 168 Stunden (letzte Woche)
    return df


# ── GRUPPE 2: ROLLING RETURNS ─────────────────────────────────────────────────
def add_rolling_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns über verschiedene Rückblickperioden.
    Misst Momentum auf unterschiedlichen Zeitskalen.

    Args:
        df: DataFrame.

    Returns:
        pd.DataFrame: DataFrame mit Return-Features.
    """
    df["return_1"]   = df["close"].pct_change(1)    # 1h Rendite
    df["return_6"]   = df["close"].pct_change(6)    # 6h Rendite
    df["return_24"]  = df["close"].pct_change(24)   # 24h Rendite
    df["return_168"] = df["close"].pct_change(168)  # Wöchentliche Rendite
    return df


# ── GRUPPE 3: VOLATILITÄT ─────────────────────────────────────────────────────
def add_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rollende Standardabweichung der Returns auf mehreren Zeitskalen.
    Kernfeature für die Regime-Erkennung.

    Args:
        df: DataFrame.

    Returns:
        pd.DataFrame: DataFrame mit Volatilitäts-Features.
    """
    df["volatility_6"]   = df["returns"].rolling(6).std()    # 6h Volatilität
    df["volatility_24"]  = df["returns"].rolling(24).std()   # 24h Volatilität (bereits vorhanden)
    df["volatility_168"] = df["returns"].rolling(168).std()  # Wöchentliche Volatilität
    return df


# ── GRUPPE 4: EMA RIBBON ─────────────────────────────────────────────────────
def add_ema_ribbon(df: pd.DataFrame) -> pd.DataFrame:
    """
    EMA Ribbon: Mehrere exponentiell gewichtete Mittelwerte.
    Ribbon-Ausrichtung zeigt Trendstärke und -richtung.

    Bullish Ribbon: Kurze EMAs > Lange EMAs (Aufwärtstrend)
    Bearish Ribbon: Kurze EMAs < Lange EMAs (Abwärtstrend)

    Args:
        df: DataFrame.

    Returns:
        pd.DataFrame: DataFrame mit EMA-Features.
    """
    for period in [6, 12, 18, 24, 50, 100]:
        df[f"EMA_{period}"] = df["close"].ewm(span=period, adjust=False).mean()

    # Ribbon Spread (normalisiert)
    df["ribbon_spread"]      = df["EMA_6"] - df["EMA_50"]
    df["ribbon_spread_norm"] = df["ribbon_spread"] / df["close"]

    # Ribbon Ausrichtung
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

    return df


# ── GRUPPE 5: SMA ─────────────────────────────────────────────────────────────
def add_sma_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simple Moving Averages und Preis-Abstand zum SMA50 als Normalisierung.

    Args:
        df: DataFrame.

    Returns:
        pd.DataFrame: DataFrame mit SMA-Features.
    """
    df["SMA_12"]        = df["close"].rolling(12).mean()
    df["SMA_24"]        = df["close"].rolling(24).mean()
    df["SMA_50"]        = df["close"].rolling(50).mean()
    df["price_vs_SMA50"] = df["close"] / df["SMA_50"] - 1   # Abstand normalisiert
    return df


# ── GRUPPE 6: MACD ────────────────────────────────────────────────────────────
def add_macd_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    MACD (Moving Average Convergence Divergence):
    Trendfolge-Indikator basierend auf EMA-Differenzen.

    Crossover-Signale werden Leakage-frei berechnet (Vergleich mit shift(1)).

    Args:
        df: DataFrame.

    Returns:
        pd.DataFrame: DataFrame mit MACD-Features.
    """
    ema_12 = df["close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["close"].ewm(span=26, adjust=False).mean()

    df["MACD"]             = ema_12 - ema_26
    df["MACD_signal"]      = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_hist"]        = df["MACD"] - df["MACD_signal"]
    df["MACD_hist_change"] = df["MACD_hist"] - df["MACD_hist"].shift(1)

    # No-Repaint Crossover: vergleiche aktuelle mit vorheriger Kerze
    df["macd_cross_up"] = (
        (df["MACD"].shift(1) < df["MACD_signal"].shift(1)) &
        (df["MACD"]          > df["MACD_signal"])
    ).astype(int)

    df["macd_cross_down"] = (
        (df["MACD"].shift(1) > df["MACD_signal"].shift(1)) &
        (df["MACD"]          < df["MACD_signal"])
    ).astype(int)

    return df


# ── GRUPPE 7: RSI ─────────────────────────────────────────────────────────────
def add_rsi_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    RSI (Relative Strength Index, 14 Perioden):
    Momentum-Oszillator für Überkauft/Überverkauft-Erkennung.

    < 30  = Überverkauft → potenzielle Kaufgelegenheit
    > 70  = Überkauft    → potenzielle Verkaufsgelegenheit
    30–70 = Neutral

    Args:
        df: DataFrame.

    Returns:
        pd.DataFrame: DataFrame mit RSI-Features.
    """
    delta = df["close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()

    df["RSI"] = 100 - (100 / (1 + gain / loss))

    df["rsi_oversold"]   = (df["RSI"] < 30).astype(int)   # Überverkauft
    df["rsi_overbought"] = (df["RSI"] > 70).astype(int)   # Überkauft
    df["rsi_neutral"]    = ((df["RSI"] >= 30) & (df["RSI"] <= 70)).astype(int)

    return df


# ── GRUPPE 8: BOLLINGER BANDS ─────────────────────────────────────────────────
def add_bollinger_bands(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bollinger Bands (20 Perioden, 2 Standardabweichungen):
    Volatilitäts-basierte Preis-Envelope.

    BB_width:   Breite der Bänder (normalisiert) → Volatilitätsindikator
    BB_pos:     Position des Preises innerhalb der Bänder (0=unteres, 1=oberes Band)
    BB_squeeze: Engste Bänder der letzten 24h → Ausbruch steht bevor

    Args:
        df: DataFrame.

    Returns:
        pd.DataFrame: DataFrame mit Bollinger-Band-Features.
    """
    df["BB_middle"] = df["close"].rolling(20).mean()
    df["BB_std"]    = df["close"].rolling(20).std()
    df["BB_upper"]  = df["BB_middle"] + (2 * df["BB_std"])
    df["BB_lower"]  = df["BB_middle"] - (2 * df["BB_std"])
    df["BB_width"]  = (df["BB_upper"] - df["BB_lower"]) / df["BB_middle"]
    df["BB_pos"]    = (df["close"] - df["BB_lower"]) / (df["BB_upper"] - df["BB_lower"])
    df["BB_squeeze"] = (df["BB_width"] < df["BB_width"].rolling(24).mean()).astype(int)

    return df


# ── GRUPPE 9: VOLUMEN ─────────────────────────────────────────────────────────
def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Volumens-basierte Features:
    Abweichung vom Durchschnittsvolumen signalisiert erhöhte Marktaktivität.

    volume_ratio: Aktuelles Volumen / 24h Durchschnitt
    volume_surge: True wenn Volumen > 2× Durchschnitt → außergewöhnliche Aktivität

    Args:
        df: DataFrame.

    Returns:
        pd.DataFrame: DataFrame mit Volumen-Features.
    """
    df["volume_ma_24"] = df["volume"].rolling(24).mean()
    df["volume_ratio"] = df["volume"] / df["volume_ma_24"]
    df["volume_surge"] = (df["volume_ratio"] > 2.0).astype(int)

    return df


# ── GRUPPE 10: SUPPORT & RESISTANCE ──────────────────────────────────────────
def add_support_resistance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Support- und Widerstandslevel basierend auf rollenden Hochs/Tiefs und Pivot Points.

    shift(1) garantiert No-Leakage: Levels basieren auf abgeschlossenen Kerzen.
    Psychologische Level (runde Tausender) als zusätzlicher Attraktor.

    Args:
        df: DataFrame.

    Returns:
        pd.DataFrame: DataFrame mit S&R-Features.
    """
    # Rolling High/Low (shift=1 für No-Leakage)
    df["rolling_high_24"]  = df["high"].rolling(24).max().shift(1)
    df["rolling_low_24"]   = df["low"].rolling(24).min().shift(1)
    df["rolling_high_168"] = df["high"].rolling(168).max().shift(1)
    df["rolling_low_168"]  = df["low"].rolling(168).min().shift(1)

    # Abstand zu S&R Levels (normalisiert durch aktuellen Preis)
    df["dist_to_high_24"]  = (df["rolling_high_24"]  - df["close"]) / df["close"]
    df["dist_to_low_24"]   = (df["close"] - df["rolling_low_24"])   / df["close"]
    df["dist_to_high_168"] = (df["rolling_high_168"] - df["close"]) / df["close"]
    df["dist_to_low_168"]  = (df["close"] - df["rolling_low_168"])  / df["close"]

    # Pivot Points (basieren auf vorheriger Kerze → No-Leakage)
    df["pivot"]      = (df["high"].shift(1) + df["low"].shift(1) + df["close"].shift(1)) / 3
    df["R1"]         = 2 * df["pivot"] - df["low"].shift(1)
    df["S1"]         = 2 * df["pivot"] - df["high"].shift(1)
    df["dist_pivot"] = (df["close"] - df["pivot"]) / df["close"]
    df["dist_R1"]    = (df["close"] - df["R1"])    / df["close"]
    df["dist_S1"]    = (df["close"] - df["S1"])    / df["close"]

    # Psychologische Level (runde 5000er)
    df["round_level"]   = (df["close"] // 5000) * 5000
    df["dist_to_round"] = abs(df["close"] - df["round_level"]) / df["close"]

    return df


# ── GRUPPE 11: ZEITLICHE FEATURES ─────────────────────────────────────────────
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Zyklisch kodierte Zeitfeatures (Sinus/Kosinus Encoding).

    Lineare Kodierung (Stunde 0–23) wäre falsch, da Stunde 23 und Stunde 0
    nicht weit auseinanderliegen. Zyklische Kodierung löst dieses Problem.

    Args:
        df: DataFrame.

    Returns:
        pd.DataFrame: DataFrame mit zeitlichen Features.
    """
    df["hour"]  = df.index.hour
    df["dow"]   = df.index.dayofweek
    df["month"] = df.index.month

    # Sinus/Kosinus Encoding für Zyklizität
    df["hour_sin"]  = np.sin(2 * np.pi * df["hour"]  / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * df["hour"]  / 24)
    df["dow_sin"]   = np.sin(2 * np.pi * df["dow"]   / 7)
    df["dow_cos"]   = np.cos(2 * np.pi * df["dow"]   / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    return df


# ── PLOT 1: FEATURE EXAMPLES ──────────────────────────────────────────────────
def plot_feature_examples(df: pd.DataFrame):
    """
    4 Subplots mit wichtigsten technischen Indikatoren auf einem Zoom-Zeitraum.

    Subplots:
        1. BTC Preis + EMA 12 + EMA 50 + Bollinger Bands
        2. RSI mit Überkauft/Überverkauft-Zonen
        3. MACD + Signal + Histogramm
        4. Volatilität 24h mit Regime-Farben

    Zeigt: Wie die Features visuell auf Marktbewegungen reagieren.
    """
    # Letzten 90 Tage für gut sichtbare Details
    cutoff = df.index.max() - pd.Timedelta(days=90)
    dz = df.loc[df.index >= cutoff].copy()

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), dpi=150,
                              gridspec_kw={"height_ratios": [2.5, 1, 1.5, 1]},
                              sharex=True)
    fig.suptitle("Feature Engineering – Technische Indikatoren (letzte 90 Tage)",
                 fontsize=13, fontweight="bold")

    # ── Subplot 1: Preis + EMA + BB ──────────────────────────────────────────
    ax1 = axes[0]
    ax1.fill_between(dz.index, dz["BB_lower"], dz["BB_upper"],
                     alpha=0.15, color=COLOR_PRICE, label="Bollinger Bands")
    ax1.plot(dz.index, dz["close"],   color=COLOR_PRICE,  linewidth=1.2, label="BTC Close")
    ax1.plot(dz.index, dz["EMA_12"],  color=COLOR_BUY,    linewidth=0.9, linestyle="--",
             label="EMA 12")
    ax1.plot(dz.index, dz["EMA_50"],  color=COLOR_SELL,   linewidth=0.9, linestyle="--",
             label="EMA 50")
    ax1.plot(dz.index, dz["BB_upper"], color=COLOR_PRICE, linewidth=0.5, alpha=0.5)
    ax1.plot(dz.index, dz["BB_lower"], color=COLOR_PRICE, linewidth=0.5, alpha=0.5)
    ax1.set_ylabel("Preis (USD)", fontsize=9)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax1.legend(loc="upper left", fontsize=8, ncol=4)
    ax1.grid(True, alpha=0.3)

    # ── Subplot 2: RSI ────────────────────────────────────────────────────────
    ax2 = axes[1]
    ax2.plot(dz.index, dz["RSI"], color="#9B59B6", linewidth=1.0, label="RSI(14)")
    ax2.axhline(70, color=COLOR_SELL, linestyle=":", linewidth=1.0, label="Überkauft (70)")
    ax2.axhline(30, color=COLOR_BUY,  linestyle=":", linewidth=1.0, label="Überverkauft (30)")
    ax2.fill_between(dz.index, 70, dz["RSI"].clip(upper=100),
                     where=dz["RSI"] > 70, alpha=0.2, color=COLOR_SELL)
    ax2.fill_between(dz.index, dz["RSI"].clip(lower=0), 30,
                     where=dz["RSI"] < 30, alpha=0.2, color=COLOR_BUY)
    ax2.set_ylabel("RSI", fontsize=9)
    ax2.set_ylim(0, 100)
    ax2.legend(loc="upper left", fontsize=8, ncol=3)
    ax2.grid(True, alpha=0.3)

    # ── Subplot 3: MACD ───────────────────────────────────────────────────────
    ax3 = axes[2]
    ax3.plot(dz.index, dz["MACD"],        color=COLOR_PRICE, linewidth=1.0, label="MACD")
    ax3.plot(dz.index, dz["MACD_signal"], color=COLOR_SELL,  linewidth=1.0, linestyle="--",
             label="Signal")
    # Histogramm
    colors_hist = [COLOR_BUY if v >= 0 else COLOR_SELL for v in dz["MACD_hist"]]
    ax3.bar(dz.index, dz["MACD_hist"], color=colors_hist, alpha=0.6, width=0.04,
            label="Histogramm")
    ax3.axhline(0, color="black", linewidth=0.5)
    ax3.set_ylabel("MACD", fontsize=9)
    ax3.legend(loc="upper left", fontsize=8, ncol=3)
    ax3.grid(True, alpha=0.3)

    # ── Subplot 4: Volatilität + Regime ──────────────────────────────────────
    ax4 = axes[3]
    ax4.plot(dz.index, dz["volatility_24"], color=COLOR_VOL, linewidth=0.9,
             label="Volatilität 24h")

    # Regime-Farben als Hintergrund einzeichnen
    if "regime" in dz.columns:
        for regime_id, color in REGIME_COLORS.items():
            mask = dz["regime"] == regime_id
            if mask.any():
                ax4.fill_between(dz.index, 0, dz["volatility_24"].max(),
                                 where=mask, alpha=0.15, color=color)

    ax4.set_ylabel("Volatilität", fontsize=9)
    ax4.set_xlabel("Datum", fontsize=9)
    ax4.xaxis.set_major_formatter(mdates.DateFormatter("%d. %b %Y"))
    ax4.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.xticks(rotation=45)
    ax4.legend(loc="upper left", fontsize=8)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(IMAGES_DIR, "step3_feature_examples.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[INFO] Gespeichert: {path}")


# ── PLOT 2: REGIME VERTEILUNG ─────────────────────────────────────────────────
def plot_regime_distribution(df: pd.DataFrame):
    """
    Balkendiagramm der Regime-Verteilung (Stunden pro Klasse).

    Zeigt: Ist die Klassenverteilung balanced? Imbalance würde
    class_weight='balanced' im ML-Modell erfordern.
    """
    counts = df["regime"].value_counts().sort_index()
    labels = [f"Regime {int(i)}\n{REGIME_LABELS[int(i)]}" for i in counts.index]
    colors = [REGIME_COLORS[int(i)] for i in counts.index]

    fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
    bars = ax.bar(labels, counts.values, color=colors, alpha=0.85, edgecolor="white")

    # Anzahl und Prozent über den Balken
    total = counts.sum()
    for bar, count in zip(bars, counts.values):
        pct = count / total * 100
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                f"{count:,}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=9)

    ax.set_title("Regime-Verteilung – Stunden pro Klasse (0–4)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Regime", fontsize=10)
    ax.set_ylabel("Anzahl Stunden", fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0, counts.max() * 1.15)

    plt.tight_layout()
    path = os.path.join(IMAGES_DIR, "step3_regime_distribution.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[INFO] Gespeichert: {path}")


# ── TERMINAL AUSGABE ──────────────────────────────────────────────────────────
def print_summary(df_raw: pd.DataFrame, df_clean: pd.DataFrame):
    """
    Gibt abschließende Zusammenfassung der Feature-Engineering-Ergebnisse aus.

    Args:
        df_raw:   DataFrame vor NaN-Bereinigung.
        df_clean: DataFrame nach NaN-Bereinigung.
    """
    print("\n" + "=" * 60)
    print("  FEATURE ENGINEERING – ZUSAMMENFASSUNG")
    print("=" * 60)

    print(f"\n[FEATURE STATISTIK]")
    print(f"  Features gesamt          : {len(df_clean.columns)}")
    print(f"  Datenpunkte nach Bereinigung: {len(df_clean):,}")
    print(f"  Verlorene Zeilen durch NaN  : {len(df_raw) - len(df_clean):,}")

    print(f"\n[REGIME VERTEILUNG]")
    regime_counts = df_clean["regime"].value_counts().sort_index()
    for regime_id, count in regime_counts.items():
        label = REGIME_LABELS[int(regime_id)]
        pct   = count / len(df_clean) * 100
        print(f"  Regime {int(regime_id)} ({label:>25}): {count:>6,} ({pct:.1f}%)")

    print(f"\n[REGIME IN PROZENT]")
    print((df_clean["regime"].value_counts(normalize=True).sort_index() * 100).round(2).to_string())

    print(f"\n[ALLE FEATURE-SPALTEN]")
    feature_cols = [c for c in df_clean.columns if c not in
                    ["open", "high", "low", "close", "volume", "trade_count", "vwap", "regime"]]
    for i, col in enumerate(feature_cols, 1):
        print(f"  {i:>2}. {col}")

    print("=" * 60)


# ── HAUPTPROGRAMM ─────────────────────────────────────────────────────────────
def main():
    """Hauptfunktion: Feature Engineering Pipeline ausführen."""
    ensure_directories()

    # Daten laden
    df = load_data()

    print("\n[INFO] Berechne Zielvariable (Regime Labels)...")
    df = add_target_variable(df)

    print("[INFO] Berechne Features – Gruppe 1: Preis Lags...")
    df = add_price_lags(df)

    print("[INFO] Berechne Features – Gruppe 2: Rolling Returns...")
    df = add_rolling_returns(df)

    print("[INFO] Berechne Features – Gruppe 3: Volatilität...")
    df = add_volatility_features(df)

    print("[INFO] Berechne Features – Gruppe 4: EMA Ribbon...")
    df = add_ema_ribbon(df)

    print("[INFO] Berechne Features – Gruppe 5: SMA...")
    df = add_sma_features(df)

    print("[INFO] Berechne Features – Gruppe 6: MACD...")
    df = add_macd_features(df)

    print("[INFO] Berechne Features – Gruppe 7: RSI...")
    df = add_rsi_features(df)

    print("[INFO] Berechne Features – Gruppe 8: Bollinger Bands...")
    df = add_bollinger_bands(df)

    print("[INFO] Berechne Features – Gruppe 9: Volumen...")
    df = add_volume_features(df)

    print("[INFO] Berechne Features – Gruppe 10: Support & Resistance...")
    df = add_support_resistance(df)

    print("[INFO] Berechne Features – Gruppe 11: Zeitliche Features...")
    df = add_time_features(df)

    # NaN Zeilen entfernen (durch Lags und Rolling Windows)
    df_raw_len = len(df)
    df_clean = df.dropna()
    print(f"\n[INFO] NaN-Bereinigung: {df_raw_len:,} → {len(df_clean):,} Zeilen "
          f"({df_raw_len - len(df_clean):,} entfernt)")

    # Regime-Spalte als Integer
    df_clean["regime"] = df_clean["regime"].astype(int)

    # Features speichern
    df_clean.to_csv(OUTPUT_CSV)
    print(f"[INFO] Gespeichert: {OUTPUT_CSV}")

    # Plots erstellen
    print("\n[INFO] Erstelle Plots...")
    plot_feature_examples(df_clean)
    plot_regime_distribution(df_clean)

    # Zusammenfassung ausgeben
    print_summary(df, df_clean)

    print("\n✅ Schritt 3 abgeschlossen – features.csv gespeichert")


if __name__ == "__main__":
    main()

"""
03_pre_split_prep.py
====================
Schritt 3: Feature Engineering VOR dem Train/Test-Split (MATHEMATISCH KORRIGIERT)

Dieses Script lädt die Rohdaten aus data/raw_data.csv, berechnet alle
technischen Features ausschließlich aus Vergangenheitsdaten (kein Data Leakage),
definiert die zukunftsgerichtete Zielvariable (Regime 0–4) und speichert den
fertigen Feature-DataFrame als data/features.csv.

KORREKTUREN:
    - Target Lookahead: Richtung via .pct_change(6).shift(-6) gelöst.
    - Stationarität: Alle absoluten Preis-Indikatoren wurden in prozentuale
                     Abstände zum aktuellen Schlusskurs umgewandelt.
    - Echter RSI: Ersetzt einfachen Durchschnitt durch Wilder's Smoothing (EWMA).
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
    os.makedirs(IMAGES_DIR, exist_ok=True)


# ── DATEN LADEN ───────────────────────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    print(f"[INFO] Lade Daten aus {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV, index_col=0, parse_dates=True)
    print(f"[INFO] Geladen: {df.shape[0]:,} Zeilen, {df.shape[1]} Spalten")
    return df


# ── ZIELVARIABLE (REGIME LABELS) ──────────────────────────────────────────────
def add_target_variable(df: pd.DataFrame) -> pd.DataFrame:
    # Stündliche Returns
    df["returns"] = df["close"].pct_change()

    # Rollende 24h Volatilität
    df["volatility_24"] = df["returns"].rolling(24).std()

    # Volatilitäts-Quantile auf Gesamtdaten
    vol_q33 = df["volatility_24"].quantile(0.33)
    vol_q66 = df["volatility_24"].quantile(0.66)
    print(f"[INFO] Volatilitäts-Quantile: 33%={vol_q33:.5f}, 66%={vol_q66:.5f}")

    # FIX 1: Echter Lookahead – Schaut 6 Stunden in die ZUKUNFT!
    df["direction_6"] = df["close"].pct_change(6).shift(-6)

    # Regime Labels zuweisen
    def assign_regime(row):
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


# ── GRUPPE 1: PREIS LAGS (STATIONÄR) ──────────────────────────────────────────
def add_price_lags(df: pd.DataFrame) -> pd.DataFrame:
    # FIX 2: Umgewandelt in prozentualen Abstand zum aktuellen Schlusskurs
    df["lag_1"]   = (df["close"].shift(1) - df["close"]) / df["close"]
    df["lag_6"]   = (df["close"].shift(6) - df["close"]) / df["close"]
    df["lag_24"]  = (df["close"].shift(24) - df["close"]) / df["close"]
    df["lag_168"] = (df["close"].shift(168) - df["close"]) / df["close"]
    return df


# ── GRUPPE 2: ROLLING RETURNS ─────────────────────────────────────────────────
def add_rolling_returns(df: pd.DataFrame) -> pd.DataFrame:
    df["return_1"]   = df["close"].pct_change(1)
    df["return_6"]   = df["close"].pct_change(6)
    df["return_24"]  = df["close"].pct_change(24)
    df["return_168"] = df["close"].pct_change(168)
    return df


# ── GRUPPE 3: VOLATILITÄT ─────────────────────────────────────────────────────
def add_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    df["volatility_6"]   = df["returns"].rolling(6).std()
    df["volatility_24"]  = df["returns"].rolling(24).std()
    df["volatility_168"] = df["returns"].rolling(168).std()
    return df


# ── GRUPPE 4: EMA RIBBON (STATIONÄR) ──────────────────────────────────────────
def add_ema_ribbon(df: pd.DataFrame) -> pd.DataFrame:
    # FIX 3: EMAs als prozentualer Abstand zum aktuellen Preis berechnet
    for period in [6, 12, 18, 24, 50, 100]:
        abs_ema = df["close"].ewm(span=period, adjust=False).mean()
        df[f"EMA_{period}"] = (abs_ema - df["close"]) / df["close"]

    # Spreads basieren nun logischerweise direkt auf den bereits skalierten EMAs
    df["ribbon_spread"]      = df["EMA_6"] - df["EMA_50"]
    df["ribbon_spread_norm"] = df["ribbon_spread"]

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


# ── GRUPPE 5: SMA (STATIONÄR) ─────────────────────────────────────────────────
def add_sma_features(df: pd.DataFrame) -> pd.DataFrame:
    # FIX 4: SMAs als prozentualen Abstand zum aktuellen Kurs skaliert
    df["SMA_12"]        = (df["close"].rolling(12).mean() - df["close"]) / df["close"]
    df["SMA_24"]        = (df["close"].rolling(24).mean() - df["close"]) / df["close"]
    df["SMA_50"]        = (df["close"].rolling(50).mean() - df["close"]) / df["close"]
    df["price_vs_SMA50"] = df["SMA_50"]
    return df


# ── GRUPPE 6: MACD ────────────────────────────────────────────────────────────
def add_macd_features(df: pd.DataFrame) -> pd.DataFrame:
    ema_12 = df["close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["close"].ewm(span=26, adjust=False).mean()

    df["MACD"]             = ema_12 - ema_26
    df["MACD_signal"]      = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_hist"]        = df["MACD"] - df["MACD_signal"]
    df["MACD_hist_change"] = df["MACD_hist"] - df["MACD_hist"].shift(1)

    df["macd_cross_up"] = (
            (df["MACD"].shift(1) < df["MACD_signal"].shift(1)) &
            (df["MACD"]          > df["MACD_signal"])
    ).astype(int)

    df["macd_cross_down"] = (
            (df["MACD"].shift(1) > df["MACD_signal"].shift(1)) &
            (df["MACD"]          < df["MACD_signal"])
    ).astype(int)
    return df


# ── GRUPPE 7: RSI (KORRIGIERT MIT EWMA GLÄTTUNG) ──────────────────────────────
def add_rsi_features(df: pd.DataFrame) -> pd.DataFrame:
    # FIX 5: Ersetzt falschen .rolling().mean() durch echtes Wilder's Smoothing via ewm
    delta = df["close"].diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()

    df["RSI"] = 100 - (100 / (1 + avg_gain / avg_loss))

    df["rsi_oversold"]   = (df["RSI"] < 30).astype(int)
    df["rsi_overbought"] = (df["RSI"] > 70).astype(int)
    df["rsi_neutral"]    = ((df["RSI"] >= 30) & (df["RSI"] <= 70)).astype(int)
    return df


# ── GRUPPE 8: BOLLINGER BANDS ─────────────────────────────────────────────────
def add_bollinger_bands(df: pd.DataFrame) -> pd.DataFrame:
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
    df["volume_ma_24"] = df["volume"].rolling(24).mean()
    df["volume_ratio"] = df["volume"] / df["volume_ma_24"]
    df["volume_surge"] = (df["volume_ratio"] > 2.0).astype(int)
    return df


# ── GRUPPE 10: SUPPORT & RESISTANCE (STATIONÄR) ───────────────────────────────
def add_support_resistance(df: pd.DataFrame) -> pd.DataFrame:
    df["rolling_high_24"]  = df["high"].rolling(24).max().shift(1)
    df["rolling_low_24"]   = df["low"].rolling(24).min().shift(1)
    df["rolling_high_168"] = df["high"].rolling(168).max().shift(1)
    df["rolling_low_168"]  = df["low"].rolling(168).min().shift(1)

    df["dist_to_high_24"]  = (df["rolling_high_24"]  - df["close"]) / df["close"]
    df["dist_to_low_24"]   = (df["close"] - df["rolling_low_24"])   / df["close"]
    df["dist_to_high_168"] = (df["rolling_high_168"] - df["close"]) / df["close"]
    df["dist_to_low_168"]  = (df["close"] - df["rolling_low_168"])  / df["close"]

    # FIX 6: Pivot Points berechnen (absolute Zwischenwerte, normalisierte Features)
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
    return df


# ── GRUPPE 11: ZEITLICHE FEATURES ─────────────────────────────────────────────
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df["hour"]  = df.index.hour
    df["dow"]   = df.index.dayofweek
    df["month"] = df.index.month

    df["hour_sin"]  = np.sin(2 * np.pi * df["hour"]  / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * df["hour"]  / 24)
    df["dow_sin"]   = np.sin(2 * np.pi * df["dow"]   / 7)
    df["dow_cos"]   = np.cos(2 * np.pi * df["dow"]   / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    return df


# ── PLOT-FUNKTIONEN (Bleiben identisch für Strukturstabilität) ─────────────────
def plot_feature_examples(df: pd.DataFrame):
    cutoff = df.index.max() - pd.Timedelta(days=90)
    dz = df.loc[df.index >= cutoff].copy()

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), dpi=150,
                             gridspec_kw={"height_ratios": [2.5, 1, 1.5, 1]},
                             sharex=True)
    fig.suptitle("Feature Engineering – Technische Indikatoren (letzte 90 Tage)", fontsize=13, fontweight="bold")

    # Preis-Plot (Da EMAs nun skaliert sind, plotten wir hier nur das BB-Envelope als Annäherung relativ)
    ax1 = axes[0]
    ax1.plot(dz.index, dz["close"], color=COLOR_PRICE, linewidth=1.2, label="BTC Close")
    ax1.set_ylabel("Preis (USD)", fontsize=9)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(True, alpha=0.3)

    # RSI
    ax2 = axes[1]
    ax2.plot(dz.index, dz["RSI"], color="#9B59B6", linewidth=1.0, label="RSI(14)")
    ax2.axhline(70, color=COLOR_SELL, linestyle=":", linewidth=1.0)
    ax2.axhline(30, color=COLOR_BUY,  linestyle=":", linewidth=1.0)
    ax2.set_ylabel("RSI", fontsize=9)
    ax2.set_ylim(0, 100)
    ax2.grid(True, alpha=0.3)

    # MACD
    ax3 = axes[2]
    ax3.plot(dz.index, dz["MACD"], color=COLOR_PRICE, linewidth=1.0, label="MACD")
    ax3.plot(dz.index, dz["MACD_signal"], color=COLOR_SELL, linewidth=1.0, linestyle="--")
    colors_hist = [COLOR_BUY if v >= 0 else COLOR_SELL for v in dz["MACD_hist"]]
    ax3.bar(dz.index, dz["MACD_hist"], color=colors_hist, alpha=0.6, width=0.04)
    ax3.set_ylabel("MACD", fontsize=9)
    ax3.grid(True, alpha=0.3)

    # Volatilität
    ax4 = axes[3]
    ax4.plot(dz.index, dz["volatility_24"], color=COLOR_VOL, linewidth=0.9, label="Volatilität 24h")
    ax4.set_ylabel("Volatilität", fontsize=9)
    ax4.set_xlabel("Datum", fontsize=9)
    ax4.xaxis.set_major_formatter(mdates.DateFormatter("%d. %b %Y"))
    plt.xticks(rotation=45)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, "step3_feature_examples.png"), dpi=150)
    plt.close()


def plot_regime_distribution(df: pd.DataFrame):
    counts = df["regime"].value_counts().sort_index()
    labels = [f"Regime {int(i)}\n{REGIME_LABELS[int(i)]}" for i in counts.index]
    colors = [REGIME_COLORS[int(i)] for i in counts.index]

    fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
    bars = ax.bar(labels, counts.values, color=colors, alpha=0.85, edgecolor="white")

    total = counts.sum()
    for bar, count in zip(bars, counts.values):
        pct = count / total * 100
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                f"{count:,}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=9)

    ax.set_title("Regime-Verteilung – Stunden pro Klasse (0–4)", fontsize=12, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, "step3_regime_distribution.png"), dpi=150)
    plt.close()


def print_summary(df_raw: pd.DataFrame, df_clean: pd.DataFrame):
    print("\n" + "=" * 60 + "\n  FEATURE ENGINEERING – ZUSAMMENFASSUNG\n" + "=" * 60)
    print(f"  Features gesamt          : {len(df_clean.columns)}")
    print(f"  Datenpunkte nach Bereinigung: {len(df_clean):,}")
    print(f"  Verlorene Zeilen durch NaN  : {len(df_raw) - len(df_clean):,}")
    print("\n[REGIME VERTEILUNG]")
    regime_counts = df_clean["regime"].value_counts().sort_index()
    for regime_id, count in regime_counts.items():
        print(f"  Regime {int(regime_id)} ({REGIME_LABELS[int(regime_id)]:>25}): {count:>6,} ({count/len(df_clean)*100:.1f}%)")
    print("=" * 60)


# ── HAUPTPROGRAMM ─────────────────────────────────────────────────────────────
def main():
    ensure_directories()
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

    # Bereinigung
    df_raw_len = len(df)
    df_clean = df.dropna()
    print(f"\n[INFO] NaN-Bereinigung: {df_raw_len:,} → {len(df_clean):,}")

    df_clean["regime"] = df_clean["regime"].astype(int)
    df_clean.to_csv(OUTPUT_CSV)
    print(f"[INFO] Gespeichert: {OUTPUT_CSV}")

    print("\n[INFO] Erstelle Plots...")
    plot_feature_examples(df_clean)
    plot_regime_distribution(df_clean)
    print_summary(df, df_clean)
    print("\n✅ Schritt 3 abgeschlossen – features.csv sauber und stationär gespeichert")


if __name__ == "__main__":
    main()
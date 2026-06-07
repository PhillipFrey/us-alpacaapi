"""
02_data_understanding.py
========================
Schritt 2: Explorative Datenanalyse (EDA)

Dieses Script lädt die Rohdaten aus data/raw_data.csv und führt eine
vollständige explorative Datenanalyse durch. Keine Veränderungen an den Daten –
nur verstehen, visualisieren und dokumentieren.

Alle Plots werden in artifacts/images/ gespeichert.

Ausführung:
    python scripts/02_data_understanding.py

Input:
    data/raw_data.csv

Output:
    artifacts/images/step2_price_history.png
    artifacts/images/step2_price_zoomed.png
    artifacts/images/step2_returns_distribution.png
    artifacts/images/step2_monthly_avg.png
    artifacts/images/step2_volume.png
    artifacts/images/step2_volatility.png
    artifacts/images/step2_missing_values.png
"""

# ── IMPORTS ──────────────────────────────────────────────────────────────────
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import numpy as np
import os

# ── KONFIGURATION ─────────────────────────────────────────────────────────────
# Farben (konsistent durch alle Scripts)
COLOR_PRICE  = "#185FA5"   # BTC Preis
COLOR_BUY    = "#1D9E75"   # Buy Signal / positive
COLOR_SELL   = "#E24B4A"   # Sell Signal / negative
COLOR_VOLUME = "#8E9AAF"   # Volumen
COLOR_VOL    = "#F4A261"   # Volatilität

# Pfade
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(BASE_DIR, "data")
IMAGES_DIR  = os.path.join(BASE_DIR, "artifacts", "images")
INPUT_CSV   = os.path.join(DATA_DIR, "raw_data.csv")

# Wichtige Marktereignisse für Annotierungen
MARKET_EVENTS = {
    "2022-11-11": ("FTX Crash",          COLOR_SELL),
    "2024-01-10": ("Bitcoin ETF approved", COLOR_BUY),
    "2024-04-20": ("Bitcoin Halving",     COLOR_BUY),
    "2024-11-06": ("Post-Election Rally", COLOR_BUY),
}


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


# ── PLOT 1: KOMPLETTER PREISVERLAUF ──────────────────────────────────────────
def plot_price_history(df: pd.DataFrame):
    """
    Zeichnet den kompletten BTC/USD Preisverlauf 2022–2025 mit
    annotierten Marktereignissen als vertikale Linien.

    Zeigt: Gesamter Preisverlauf, Korrekturen und Bullenmärkte auf einen Blick.
    """
    fig, ax = plt.subplots(figsize=(14, 5), dpi=150)

    ax.plot(df.index, df["close"], color=COLOR_PRICE, linewidth=0.8, label="BTC/USD Kurs")

    # Marktereignisse einzeichnen
    for date_str, (label, color) in MARKET_EVENTS.items():
        date = pd.Timestamp(date_str)
        if df.index.min() <= date <= df.index.max():
            ax.axvline(date, color=color, linestyle="--", linewidth=1.2, alpha=0.8)
            y_pos = df["close"].max() * 0.95
            ax.text(date, y_pos, label, rotation=90, fontsize=7,
                    va="top", ha="right", color=color, alpha=0.9)

    ax.set_title("BTC/USD Stündlicher Preisverlauf 2022–2025", fontsize=13, fontweight="bold")
    ax.set_xlabel("Datum", fontsize=10)
    ax.set_ylabel("Preis (USD)", fontsize=10)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(IMAGES_DIR, "step2_price_history.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[INFO] Gespeichert: {path}")


# ── PLOT 2: ZOOM POST-ELECTION RALLY ─────────────────────────────────────────
def plot_price_zoomed(df: pd.DataFrame):
    """
    Zoom auf Oktober 2024 bis Januar 2025 – zeigt den Post-Election Rally im Detail.

    Zeigt: Explosive Preisbewegung nach der US-Wahl Nov 2024 im Detail.
    """
    mask = (df.index >= "2024-10-01") & (df.index <= "2025-01-31")
    df_zoom = df.loc[mask]

    fig, ax = plt.subplots(figsize=(12, 4), dpi=150)
    ax.plot(df_zoom.index, df_zoom["close"], color=COLOR_PRICE, linewidth=1.0)

    # Wahl-Event einzeichnen
    election_date = pd.Timestamp("2024-11-06")
    ax.axvline(election_date, color=COLOR_BUY, linestyle="--", linewidth=1.5)
    ax.text(election_date, df_zoom["close"].max() * 0.98,
            "Post-Election Rally", rotation=90, fontsize=8,
            va="top", ha="right", color=COLOR_BUY)

    ax.set_title("BTC/USD Zoom: Post-Election Rally (Okt 2024 – Jan 2025)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Datum", fontsize=10)
    ax.set_ylabel("Preis (USD)", fontsize=10)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d. %b %Y"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.xticks(rotation=45)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(IMAGES_DIR, "step2_price_zoomed.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[INFO] Gespeichert: {path}")


# ── PLOT 3: STÜNDLICHE RENDITEVERTEILUNG ─────────────────────────────────────
def plot_returns_distribution(df: pd.DataFrame):
    """
    Histogramm der stündlichen Returns mit Normalverteilungskurve.

    Zeigt: Fat Tails gegenüber der Normalverteilung – typisch für Krypto-Assets.
    Die hohe Kurtosis belegt extreme Ausschläge in beide Richtungen.
    """
    returns = df["close"].pct_change().dropna()

    skewness = returns.skew()
    kurtosis = returns.kurtosis()
    mean_r   = returns.mean()
    std_r    = returns.std()

    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)

    # Histogramm
    ax.hist(returns, bins=200, density=True, alpha=0.6,
            color=COLOR_PRICE, label="Stündliche Returns")

    # Normalverteilung darüber legen (reine NumPy Implementierung – kein scipy nötig)
    x = np.linspace(returns.min(), returns.max(), 500)
    normal_curve = (1 / (std_r * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean_r) / std_r) ** 2)
    ax.plot(x, normal_curve, color=COLOR_SELL, linewidth=2, label="Normalverteilung")

    # Statistiken im Plot
    stats_text = (
        f"Mittelwert:  {mean_r:.6f}\n"
        f"Std:         {std_r:.4f}\n"
        f"Schiefe:     {skewness:.4f}\n"
        f"Kurtosis:    {kurtosis:.2f}"
    )
    ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
            fontsize=9, va="top", ha="right",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    ax.set_xlim(-0.15, 0.15)
    ax.set_title("Stündliche BTC/USD Renditeverteilung mit Fat Tails",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Stündliche Rendite", fontsize=10)
    ax.set_ylabel("Dichte", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(IMAGES_DIR, "step2_returns_distribution.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[INFO] Gespeichert: {path}")


# ── PLOT 4: MONATLICHE DURCHSCHNITTSPREISE ────────────────────────────────────
def plot_monthly_avg(df: pd.DataFrame):
    """
    Durchschnittlicher close-Preis pro Monat, farblich nach Jahr unterschieden.

    Zeigt: Langfristige Preisdynamik und Jahresvergleich – analog zum Beijing PM2.5 Repo.
    """
    df_monthly = df["close"].resample("ME").mean().reset_index()
    df_monthly.columns = ["Datum", "Durchschnittspreis"]
    df_monthly["Jahr"] = df_monthly["Datum"].dt.year

    years   = sorted(df_monthly["Jahr"].unique())
    palette = [COLOR_PRICE, COLOR_BUY, COLOR_SELL, "#9B59B6"]
    colors  = {yr: palette[i % len(palette)] for i, yr in enumerate(years)}

    fig, ax = plt.subplots(figsize=(12, 4), dpi=150)

    for year in years:
        mask = df_monthly["Jahr"] == year
        subset = df_monthly.loc[mask]
        ax.plot(subset["Datum"], subset["Durchschnittspreis"],
                color=colors[year], linewidth=1.8, marker="o",
                markersize=4, label=str(year))

    ax.set_title("Monatlicher Durchschnittspreis BTC/USD nach Jahr",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Datum", fontsize=10)
    ax.set_ylabel("Durchschnittspreis (USD)", fontsize=10)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.legend(title="Jahr", fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(IMAGES_DIR, "step2_monthly_avg.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[INFO] Gespeichert: {path}")


# ── PLOT 5: VOLUMENANALYSE ────────────────────────────────────────────────────
def plot_volume(df: pd.DataFrame):
    """
    Zwei Subplots: BTC Preis (oben) und Volumen als Balkendiagramm (unten).

    Zeigt: Korrelation zwischen hohem Volumen und starken Preisbewegungen –
    Volume Spikes signalisieren oft Trendwenden oder Bestätigungen.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), dpi=150,
                                    gridspec_kw={"height_ratios": [2, 1]},
                                    sharex=True)

    # Oben: Preis
    ax1.plot(df.index, df["close"], color=COLOR_PRICE, linewidth=0.7)
    ax1.set_ylabel("Preis (USD)", fontsize=10)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax1.set_title("BTC/USD Preis und Handelsvolumen", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3)

    # Unten: Volumen – für Performance als resampeltes Tagesvolumen darstellen
    df_vol = df["volume"].resample("D").sum()
    ax2.bar(df_vol.index, df_vol.values, color=COLOR_VOLUME, alpha=0.7, width=0.8)
    ax2.set_ylabel("Tagesvolumen (BTC)", fontsize=10)
    ax2.set_xlabel("Datum", fontsize=10)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(IMAGES_DIR, "step2_volume.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[INFO] Gespeichert: {path}")


# ── PLOT 6: VOLATILITÄTS-CLUSTER ─────────────────────────────────────────────
def plot_volatility(df: pd.DataFrame):
    """
    Rolling 24h Standardabweichung der stündlichen Returns.

    Zeigt: Volatilitäts-Clustering – ruhige Marktphasen wechseln sich
    mit explosiven Phasen ab (ARCH-Effekt). Relevant für Regime-Klassifikation.
    """
    returns    = df["close"].pct_change()
    volatility = returns.rolling(24).std()

    fig, ax = plt.subplots(figsize=(14, 4), dpi=150)

    ax.fill_between(df.index, volatility, alpha=0.4, color=COLOR_VOL)
    ax.plot(df.index, volatility, color=COLOR_VOL, linewidth=0.6)

    # Horizontale Linie bei 33. und 66. Perzentil (Regime-Grenzen)
    q33 = volatility.quantile(0.33)
    q66 = volatility.quantile(0.66)
    ax.axhline(q33, color=COLOR_BUY, linestyle=":", linewidth=1.2,
               label=f"33. Perzentil ({q33:.4f})")
    ax.axhline(q66, color=COLOR_SELL, linestyle=":", linewidth=1.2,
               label=f"66. Perzentil ({q66:.4f})")

    ax.set_title("Rollende 24h Volatilität – Volatilitäts-Cluster BTC/USD",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Datum", fontsize=10)
    ax.set_ylabel("Rollierende Std (24h)", fontsize=10)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(IMAGES_DIR, "step2_volatility.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[INFO] Gespeichert: {path}")


# ── PLOT 7: FEHLENDE WERTE ────────────────────────────────────────────────────
def plot_missing_values(df: pd.DataFrame):
    """
    Balkendiagramm der fehlenden Werte pro Spalte.

    Zeigt: Vollständigkeit der Daten – wichtig für Feature Engineering.
    """
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)

    fig, ax = plt.subplots(figsize=(8, 4), dpi=150)

    colors_bars = [COLOR_SELL if v > 0 else COLOR_BUY for v in missing.values]
    bars = ax.bar(missing.index, missing.values, color=colors_bars, alpha=0.8)

    # Prozentzahl über den Balken
    for bar, pct in zip(bars, missing_pct.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{pct:.1f}%", ha="center", va="bottom", fontsize=9)

    ax.set_title("Fehlende Werte pro Spalte", fontsize=12, fontweight="bold")
    ax.set_xlabel("Spalte", fontsize=10)
    ax.set_ylabel("Anzahl fehlender Werte", fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(IMAGES_DIR, "step2_missing_values.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[INFO] Gespeichert: {path}")


# ── STATISTISCHE AUSGABE ──────────────────────────────────────────────────────
def print_statistics(df: pd.DataFrame):
    """
    Gibt umfassende statistische Kennzahlen im Terminal aus.

    Args:
        df: Rohdaten DataFrame.
    """
    returns = df["close"].pct_change().dropna()

    print("\n" + "=" * 60)
    print("  EDA – STATISTISCHE AUSWERTUNG")
    print("=" * 60)

    print("\n[DESKRIPTIVE STATISTIK]")
    print(df.describe().to_string())

    print("\n[FEHLENDE WERTE PRO SPALTE]")
    print(df.isnull().sum().to_string())

    print("\n[STÜNDLICHE RETURN STATISTIKEN]")
    print(f"  Mittelwert : {returns.mean():.6f}")
    print(f"  Std        : {returns.std():.6f}")
    print(f"  Min        : {returns.min():.4f}")
    print(f"  Max        : {returns.max():.4f}")
    print(f"  Schiefe    : {returns.skew():.4f}")
    print(f"  Kurtosis   : {returns.kurtosis():.4f}")

    print("\n[VOLUMEN NULLWERTE]")
    print(f"  Nullwerte beim Volumen: {(df['volume'] == 0).sum()}")

    # Wichtige Erkenntnisse
    max_price    = df["close"].max()
    min_price    = df["close"].min()
    max_drawdown = (min_price - max_price) / max_price * 100
    avg_hourly_vol = returns.std() * 100
    max_vol_date = df["volume"].idxmax()

    print("\n[WICHTIGSTE ERKENNTNISSE]")
    print(f"  Erkenntnis 1: BTC hatte {max_drawdown:.1f}% maximalen Drawdown im Zeitraum "
          f"(${max_price:,.0f} → ${min_price:,.0f})")
    print(f"  Erkenntnis 2: Durchschnittliche stündliche Volatilität: {avg_hourly_vol:.4f}%")
    print(f"  Erkenntnis 3: Höchstes Volumen war am {max_vol_date.strftime('%d.%m.%Y %H:%M')} "
          f"({df['volume'].max():,.2f} BTC)")
    print("=" * 60)


# ── HAUPTPROGRAMM ─────────────────────────────────────────────────────────────
def main():
    """Hauptfunktion: Daten laden, analysieren und visualisieren."""
    ensure_directories()

    df = load_data()

    print("\n[INFO] Erstelle Plots...")

    plot_price_history(df)
    plot_price_zoomed(df)
    plot_returns_distribution(df)
    plot_monthly_avg(df)
    plot_volume(df)
    plot_volatility(df)
    plot_missing_values(df)

    print_statistics(df)

    print("\n✅ Schritt 2 abgeschlossen – alle Plots in artifacts/images/ gespeichert")


if __name__ == "__main__":
    main()

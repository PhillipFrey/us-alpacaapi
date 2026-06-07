"""
01_data_acquisition.py
======================
Schritt 1: Rohdaten-Beschaffung via Alpaca API

Dieses Script verbindet sich mit der Alpaca Crypto Historical Data API,
ruft alle verfügbaren stündlichen BTC/USD Bars von Juli 2022 bis heute ab
und speichert die Rohdaten als CSV. Keine Bereinigung, keine Feature-Engineering
– nur sauberes Laden und Speichern der Rohdaten.

Ausführung:
    python scripts/01_data_acquisition.py

Output:
    data/raw_data.csv
"""

# ── IMPORTS ──────────────────────────────────────────────────────────────────
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime
import pandas as pd
import os

# ── KONFIGURATION ─────────────────────────────────────────────────────────────
API_KEY    = "PKAMJ4ZDYEKTP34NX44GNWXCJF"
API_SECRET = "B43Rr36ZV9NSpQSCwKqABkAMvDTJRV8wgjcyoFzYYULA"
SYMBOL     = "BTC/USD"
START_DATE = "2022-07-01"
END_DATE   = "2025-06-01"
TIMEFRAME  = TimeFrame.Hour

# Pfade relativ zum Projektroot (Unternehmenssoftware/)
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(BASE_DIR, "data")
OUTPUT_CSV = os.path.join(DATA_DIR, "raw_data.csv")


# ── VERZEICHNISSE PRÜFEN / ERSTELLEN ─────────────────────────────────────────
def ensure_directories():
    """Erstellt benötigte Verzeichnisse falls nicht vorhanden."""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "artifacts", "images"), exist_ok=True)
    print(f"[INFO] Verzeichnisse geprüft: {DATA_DIR}")


# ── DATEN LADEN ───────────────────────────────────────────────────────────────
def fetch_btc_data() -> pd.DataFrame:
    """
    Ruft stündliche BTC/USD Bars von der Alpaca API ab.

    Returns:
        pd.DataFrame: Rohdaten mit OHLCV-Spalten und datetime Index.
    """
    print(f"[INFO] Verbinde mit Alpaca API...")
    client = CryptoHistoricalDataClient(API_KEY, API_SECRET)

    request_params = CryptoBarsRequest(
        symbol_or_symbols=SYMBOL,
        timeframe=TIMEFRAME,
        start=datetime.strptime(START_DATE, "%Y-%m-%d"),
        end=datetime.strptime(END_DATE, "%Y-%m-%d"),
    )

    print(f"[INFO] Lade {SYMBOL} {TIMEFRAME} Bars von {START_DATE} bis {END_DATE}...")
    bars = client.get_crypto_bars(request_params)

    # Konvertiere zu DataFrame
    df = bars.df
    return df


# ── DATAFRAME BEREINIGEN ──────────────────────────────────────────────────────
def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bereinigt den rohen API-DataFrame:
    - MultiIndex auflösen (Symbol-Level entfernen)
    - Index als datetime setzen
    - Chronologisch sortieren
    - Nur relevante Spalten behalten

    Args:
        df: Roher DataFrame von der Alpaca API.

    Returns:
        pd.DataFrame: Bereinigter DataFrame.
    """
    # MultiIndex auflösen – Symbol-Level entfernen
    if isinstance(df.index, pd.MultiIndex):
        df = df.droplevel("symbol")

    # Index als datetime sicherstellen und timezone-aware → naive (UTC)
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    # Chronologisch sortieren
    df = df.sort_index()

    # Nur relevante Spalten behalten
    cols_to_keep = ["open", "high", "low", "close", "volume", "trade_count", "vwap"]
    available = [c for c in cols_to_keep if c in df.columns]
    df = df[available]

    return df


# ── STATISTIKEN AUSGEBEN ──────────────────────────────────────────────────────
def print_statistics(df: pd.DataFrame):
    """
    Gibt eine erste Inspektion der Rohdaten im Terminal aus.

    Args:
        df: Bereinigter DataFrame.
    """
    print("\n" + "=" * 60)
    print("  ROHDATEN – ERSTE INSPEKTION")
    print("=" * 60)

    print(f"\n[SHAPE]")
    print(f"  Zeilen: {df.shape[0]:,}  |  Spalten: {df.shape[1]}")

    print(f"\n[DATENTYPEN]")
    print(df.dtypes.to_string())

    print(f"\n[ERSTE 10 ZEILEN]")
    print(df.head(10).to_string())

    print(f"\n[DESKRIPTIVE STATISTIK]")
    print(df.describe().to_string())

    print(f"\n[FEHLENDE WERTE PRO SPALTE]")
    missing = df.isnull().sum()
    print(missing.to_string())

    print(f"\n[ZEITRAUM]")
    print(f"  Erster Timestamp : {df.index[0]}")
    print(f"  Letzter Timestamp: {df.index[-1]}")
    duration = df.index[-1] - df.index[0]
    print(f"  Zeitraum         : {duration.days} Tage ({duration.days / 365.25:.1f} Jahre)")

    print(f"\n[GESAMTANZAHL DATENPUNKTE]")
    print(f"  {len(df):,} stündliche Bars")

    print("=" * 60)


# ── HAUPTPROGRAMM ─────────────────────────────────────────────────────────────
def main():
    """Hauptfunktion: Daten laden, bereinigen, speichern und inspizieren."""
    ensure_directories()

    # Prüfen ob Datei bereits existiert
    if os.path.exists(OUTPUT_CSV):
        print(f"[INFO] raw_data.csv bereits vorhanden – überspringe API-Aufruf.")
        print(f"[INFO] Lösche {OUTPUT_CSV} manuell um Daten neu zu laden.")
        df = pd.read_csv(OUTPUT_CSV, index_col=0, parse_dates=True)
    else:
        # API aufrufen
        try:
            df_raw = fetch_btc_data()
            df = clean_dataframe(df_raw)
        except Exception as e:
            print(f"[FEHLER] API-Aufruf fehlgeschlagen: {e}")
            raise

        # Speichern
        df.to_csv(OUTPUT_CSV)
        print(f"\n[INFO] Gespeichert: {OUTPUT_CSV}")

    # Statistiken ausgeben
    print_statistics(df)

    print("\n✅ Schritt 1 abgeschlossen – raw_data.csv gespeichert")


if __name__ == "__main__":
    main()

# BTC/USD Volatility Regime Classification & Automated Trading

Machine Learning system that classifies BTC/USD market regimes and generates
automated trading signals executed via the Alpaca API.

## What it does

Analyzes hourly BTC/USD data and classifies the current market state into one
of five regimes based on volatility and price direction:

| Regime | Condition | Signal |
|--------|-----------|--------|
| 0 | Low Volatility | HOLD |
| 1 | Medium Vol + Up | BUY |
| 2 | Medium Vol + Down | SELL |
| 3 | High Vol + Up | STRONG BUY |
| 4 | High Vol + Down | STRONG SELL |

## Tech Stack

Python 3.13 · alpaca-py · pandas · numpy · matplotlib · seaborn · scikit-learn

## Project Structure

```
├── data/                   # Raw and processed data (CSV)
├── artifacts/images/       # Generated plots
├── scripts/
│   ├── 01_data_acquisition.py    # Fetch BTC/USD data via Alpaca API
│   ├── 02_data_understanding.py  # Exploratory data analysis & plots
│   └── 03_pre_split_prep.py      # Feature engineering & regime labeling
└── .venv/
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install alpaca-py pandas numpy matplotlib seaborn scikit-learn
```

Add your Alpaca API credentials to `01_data_acquisition.py`:
```python
API_KEY    = "YOUR_API_KEY"
API_SECRET = "YOUR_API_SECRET"
```

## Usage

```bash
python scripts/01_data_acquisition.py   # Fetches ~33,600 hourly bars (2022–2025)
python scripts/02_data_understanding.py # Generates 7 EDA plots
python scripts/03_pre_split_prep.py     # Builds feature matrix (~60 features)
```

## Data

- **Source:** Alpaca Markets Crypto Historical Data API
- **Asset:** BTC/USD · Hourly bars · July 2022 – June 2025
- **Raw data:** ~33,600 rows · 7 columns (OHLCV + trade_count + vwap)
- **Feature matrix:** ~33,400 rows · ~60 columns after engineering

## Status

- [x] Data Acquisition
- [x] Exploratory Data Analysis
- [x] Feature Engineering & Regime Labeling
- [ ] Train/Test Split
- [ ] Model Training (XGBoost / Random Forest)
- [ ] Backtesting
- [ ] Live Deployment via Alpaca API

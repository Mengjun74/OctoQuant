# 🐙 OctoQuant

**AI-Powered Open-Source Quantitative Trading Framework for Stocks & Crypto**

---

## 🧭 Overview

**OctoQuant** is an open-source **AI-driven quantitative trading framework** designed for both **stocks** and **cryptocurrencies**. It provides a unified pipeline for **research → backtesting → paper trading → live execution**, emphasizing modularity, transparency, and reproducibility.

With OctoQuant, you can:

* Build and test trading strategies with clean modular APIs.
* Run backtests with realistic transaction costs and slippage.
* Deploy paper/live trading with pluggable broker adapters.
* Extend easily for ML-based signal generation or portfolio optimization.

> ⚠️ **Disclaimer:** This project is for educational and research purposes only. It is **not investment advice**. Always test thoroughly before live trading.

---

## 🧱 Core Features

* 📊 Unified framework for **stocks (via yfinance)** and **crypto (via ccxt)**.
* 🧩 Modular components: DataFeed / Strategy / PositionSizer / RiskManager / Broker.
* ⚙️ Vectorized backtesting engine (fast and reproducible).
* 🧠 ML-ready pipeline for AI-based signal generation.
* 📦 Plug-and-play deployment via Docker.
* 🧪 Fully open-source, extensible, and community-friendly.

---

## 🗂️ Directory Structure

```
octoquant/
├─ config/              # YAML configuration files
├─ octoquant/           # Core modules
│   ├─ data/            # DataFeed for YF & CCXT
│   ├─ strat/           # Strategies (SMA, Momentum...)
│   ├─ backtest/        # Vectorized backtest engine
│   ├─ exec/            # Brokers (Paper, CCXT, Alpaca...)
│   └─ risk/            # Risk & Position management
├─ scripts/             # Data ingestion utilities
├─ tests/               # Unit tests
└─ main.py              # CLI entry point
```

---

## 🚀 Quick Start

```bash
# 1. Clone the repo
 git clone https://github.com/yourname/OctoQuant.git
 cd OctoQuant

# 2. Install dependencies
 pip install -r requirements.txt

# 3. Configure your strategy
 cp config/settings.example.yaml config/settings.yaml
```

The `strategy` block now supports:

* `rsi_pullback` – default in the sample config, built for small accounts (RSI(2) pullback with trend filter).
* `sma_cross` – the original dual moving average crossover (uncomment the provided settings).
* `ml_lightgbm` – loads a LightGBM regression model trained via `scripts/train_baselines.py`.
* `ml_autoreg` – uses an AutoReg (AR) time-series model (also produced by the baseline trainer).

# 4. Run backtest
```bash
python main.py backtest --cfg config/settings.yaml
```

Example output:

```
Date        Equity
2024-12-01  100000
2025-01-01  104253
2025-02-01  108974
```

---

## 🧠 AI Integration Ideas

* **Signal Modeling:** train ML models (XGBoost, LSTM, TSTransformer) for alpha prediction.
* **LLM Integration:** summarize research notebooks, auto-generate experiment logs.
* **Reinforcement Learning:** optimize policy-based trading decisions.

## 📈 Train Baseline Models

```bash
python scripts/train_baselines.py --cfg config/settings.yaml --output-dir models
```

This command will:

- build technical features from your configured data feed,
- train a LightGBM regression baseline and a simple AutoReg time-series model,
- store artifacts under `models/` (`lightgbm_baseline.txt`, `autoreg_baseline.pkl`, `baseline_summary.json`, etc.).

After training, point your `strategy` block to the generated files, for example:

```yaml
strategy:
  name: "ml_lightgbm"
  model_path: "models/lightgbm_baseline.txt"
  feature_path: "models/lightgbm_features.pkl"
  threshold: 0.0
  short: false
```

## 🛠️ Roadmap (Work-in-Progress)

- Build a baseline predictive model on existing historical data to benchmark strategy performance.
- Design a trading environment and reward function so we can plug in RL libraries (e.g., SB3, RLlib).
- Real-time execution and rich visualization dashboards are on hold until the first two pieces are stable.

---

## 📦 License

Released under the **MIT License** © 2025 **Mengjun Chen**.

---

## 🌐 Community & Contribution

We welcome pull requests, new strategies, bug fixes, and discussions.
Before submitting, please:

* Run tests with `pytest`
* Format code with `black` and `ruff`

> Let’s build a transparent and intelligent quant ecosystem — together.

---

**Build once. Trade anywhere. — OctoQuant 🐙**

# TrustCast

TrustCast is an independent research center that tests, adapts and explains time series
foundation models. It runs every model (TimesFM, Chronos, Moirai, Toto, TiRex, TimeGPT)
under the same conditions, against classical baselines, and publishes scorecards of
forecast accuracy and how reliable each model's confidence intervals are.

## Quick start

```bash
git clone https://github.com/drdataman01-ai/trustcast.git
cd trustcast
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python scripts/fetch_data.py                     # macro data from FRED
python scripts/etl.py sp500                      # S&P 500 prices
python scripts/forecast.py --dataset sp500 --series LMT --horizon 21 --models chronos-2 seasonal-naive
```

**Joining the team?** Read [CONTRIBUTING.md](CONTRIBUTING.md) for setup, the daily
workflow and the rules for data and licenses.

## What's here

| | |
| --- | --- |
| `scripts/etl.py` | Extract, clean and prepare a dataset defined in `configs/datasets.yaml` |
| `scripts/forecast.py` | Forecast one series with several models side by side |
| `scripts/backtest.py` | Walk-forward back-test: e.g. 21-day forecasts, origin moved a week at a time, scored against a random walk |
| `scripts/run_scorecard.py` | Back-test models and publish a scorecard to `archive/` |
| `src/trustcast/` | The library: data, models, evaluation, uncertainty |

Model install notes and licenses: [src/trustcast/models/README.md](src/trustcast/models/README.md).
Data layout: [data/README.md](data/README.md).

## License

Code: MIT (see `LICENSE`). Model weights keep their own licenses; TimesFM 3.0 and
Moirai 2.0 are for non-commercial research only.

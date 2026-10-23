# Models

Every model, foundation or classical, sits behind the same interface (`base.py`):

```python
model = get_model("chronos-2")                     # names from configs/models.yaml
fc = model.predict(y, horizon=21, quantiles=[0.1, 0.5, 0.9])
fc.mean, fc.quantiles[0.1], fc.interval(0.8)
```

so the scorecard and `scripts/forecast.py` treat them identically.

## Installing

Install only the models you'll use; each adapter imports its library when first called.

| Model | Adapter | Install | License |
| --- | --- | --- | --- |
| `timesfm-2.5` | `timesfm.py` | `pip install 'timesfm[mlx] @ git+https://github.com/google-research/timesfm.git@e51928e27119cb17bebc005be2696b75e0a9e688'` | Apache-2.0 |
| `timesfm-3.0` | `timesfm.py` | same as above | non-commercial |
| `chronos-2` | `chronos.py` | `pip install chronos-forecasting` | Apache-2.0 |
| `tirex-2` | `tirex.py` | `pip install tirex-2` | Apache-2.0 |
| `toto-2.0` | `toto.py` | `pip install toto-models` (Python 3.12+) | Apache-2.0 |
| `timegpt` | `timegpt.py` | `pip install nixtla`, and `NIXTLA_API_KEY=...` in `.env` | proprietary API |
| `moirai-2.0` | `moirai.py` | separate environment, see below | CC-BY-NC-4.0 |
| `arima`, `ets` | `classical.py` | `pip install statsforecast` | — |
| `seasonal-naive` | `classical.py` | built in | — |

All but Moirai share one environment (the repo's `.venv`):

```bash
pip install 'timesfm[mlx] @ git+https://github.com/google-research/timesfm.git@e51928e27119cb17bebc005be2696b75e0a9e688' \
            chronos-forecasting tirex-2 toto-models nixtla statsforecast
```

`nixtla` caps `pyarrow` below 24, so pip may downgrade it; that's fine.

### Moirai 2.0: its own environment

Salesforce's `uni2ts` pins `torch<2.5` and `numpy 1.26`, which conflict with the other
models and don't support Python 3.13+. Give it a separate Python 3.11 environment:

```bash
brew install python@3.11
python3.11 -m venv .venv-moirai
source .venv-moirai/bin/activate
pip install -e . uni2ts pandas pyyaml pyarrow matplotlib
python scripts/forecast.py --dataset sp500 --series LMT --horizon 21 --models moirai-2.0
deactivate
```

Results land in the same `outputs/` folder, so you can compare them with the others.

## Quantiles

TimesFM, Moirai, Toto and TiRex-2 predict deciles (10th to 90th percentile); Chronos-2
predicts 1st to 99th. The scorecard asks for 2.5% to 97.5% (for 95% intervals), so
`base.from_quantile_grid` interpolates inside each model's range and extends the tails
with a normal fit outside it. The conformal step in `uncertainty/` then checks every
model's intervals against what actually happened and widens or narrows them.

## Adding a model

1. Write `src/trustcast/models/<name>.py`: a `ForecastModel` subclass whose `predict`
   returns a `Forecast` (use `from_quantile_grid` if the model has a fixed quantile grid).
2. Add a branch for its `adapter` name in `registry.py`.
3. Add an entry in `configs/models.yaml` with `hf_id`, a pinned `revision`, its license
   and `commercial_ok`.

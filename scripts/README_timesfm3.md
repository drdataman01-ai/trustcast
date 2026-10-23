# Run TimesFM 3.0 locally on a Mac

`run_timesfm3.py` forecasts one series from `data/` (or `data/private/`) with Google's TimesFM 3.0 and
reports the median forecast with an 80% interval (10th to 90th percentile). On Apple
Silicon (M1 and later) it uses the MLX backend, which runs natively without PyTorch.

## 1. Set up (once)

You need Python 3.10 or newer (`python3 --version`; if older, `brew install python@3.12`).
From the repo folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e . pandas pyyaml matplotlib
pip install "timesfm[mlx] @ git+https://github.com/google-research/timesfm.git@e51928e27119cb17bebc005be2696b75e0a9e688"
```

TimesFM comes from Google's GitHub, pinned to a tested commit (Sept 29, 2026),
because the MLX backend may not be in the PyPI release yet.

## 2. Get the data

```bash
python scripts/fetch_data.py
```

This saves the macro series to `data/` (see `data/README.md`). Your own series go
there too, as `data/<name>.csv`.

## 3. Run

```bash
python scripts/run_timesfm3.py --list                          # what's in data/
python scripts/run_timesfm3.py --series unemployment           # 12 months ahead
python scripts/run_timesfm3.py --series gdp --horizon 4        # 4 quarters ahead
```

The first run downloads the model weights (about 1.3 GB) into
`~/.cache/huggingface`. Results go to `outputs/timesfm3/` as a CSV and a chart.

| Option | Default | Meaning |
| --- | --- | --- |
| `--series` | `unemployment` | File name in `data/` or `data/private/`, without `.csv` |
| `--horizon` | 12 | Steps to forecast, in the data's own frequency |
| `--context` | 512 | How many of the most recent points the model sees |
| `--no-plot` | | Skip the chart |

## Troubleshooting

- **`... not found. Run: python scripts/fetch_data.py`:** the series isn't in `data/` yet.
- **`CERTIFICATE_VERIFY_FAILED` during fetch:** with Python from python.org, run
  *Install Certificates.command* in your Python folder under Applications.
- **`No module named mlx`:** not on Apple Silicon, or the `[mlx]` extra didn't install.
  The script falls back to PyTorch on CPU; for that, `pip install torch`.

## License

TimesFM 3.0 weights are under Google's `timesfm-non-commercial-license-v1.0`:
research and non-production use only. Use TimesFM 2.5 (Apache-2.0) for commercial work.

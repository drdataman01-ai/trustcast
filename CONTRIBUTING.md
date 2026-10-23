# Working on TrustCast

How to set up the project and share your work with the team.

## 1. Get access

Send Chris your GitHub username. You'll get an invitation by email; accept it, and you
can push to [github.com/drdataman01-ai/trustcast](https://github.com/drdataman01-ai/trustcast).

## 2. Set up your computer (once)

You need Git and Python 3.12. Check with `python3 --version`; on a Mac, install it with
`brew install python@3.12` if needed. Python 3.12 works with every model we use.

```bash
git clone https://github.com/drdataman01-ai/trustcast.git
cd trustcast
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` also installs the project's own code (`src/trustcast`) in editable
mode, so your edits take effect without reinstalling.

**Models.** Install the foundation models you'll use. Instructions and licenses for each
are in [`src/trustcast/models/README.md`](src/trustcast/models/README.md). For most of them:

```bash
pip install 'timesfm[mlx] @ git+https://github.com/google-research/timesfm.git@e51928e27119cb17bebc005be2696b75e0a9e688' \
            tirex-2 toto-models nixtla
```

Drop `[mlx]` if you're not on an Apple Silicon Mac. Moirai 2.0 needs its own environment
(see the models README).

**Data.** Data is never shared through GitHub; each person builds their own copy:

```bash
python scripts/fetch_data.py       # US macro series from FRED
python scripts/etl.py sp500        # S&P 500 daily prices (takes a few minutes)
```

**Check it works:**

```bash
python scripts/forecast.py --dataset sp500 --series LMT --horizon 21 --models chronos-2 seasonal-naive
```

Each model downloads its weights the first time it runs.

## 3. Every time you work

Always start from the latest version, and make changes on a branch, never directly on `main`.

```bash
git checkout main
git pull                              # get everyone's latest changes
git checkout -b short-description     # a new branch for this piece of work
```

Make your changes, then save and share them:

```bash
git status                            # see what changed
git add -A
git commit -m "What you changed, in a short sentence"
git push -u origin short-description
```

GitHub then shows a link to **open a pull request**. Open it, describe what you changed
and why, and ask a teammate to review. Once it's approved, merge it on GitHub. Then:

```bash
git checkout main
git pull
```

## 4. What gets pushed

| File | Pushed to GitHub? |
| --- | --- |
| Code, scripts, `configs/` | Yes |
| `data/*.csv` (small public series, e.g. FRED) | Yes, so everyone uses identical data |
| `data/*.parquet` (large datasets such as S&P 500 prices) | No: rebuild with `scripts/etl.py` |
| `data/private/` (client and partner data) | Never |
| `outputs/` (forecasts, charts, reports) | No |
| `.env` (API keys) | Never |

`.gitignore` keeps everything marked "No" or "Never" out of git automatically. If
`git status` ever lists a data file or a key you didn't expect, stop and ask before committing.

## 5. Rules

- **Don't push to `main` directly.** Every change goes through a pull request with one
  approval. `main` also publishes the website.
- **Check licenses.** TimesFM 3.0 and Moirai 2.0 are licensed for research only; don't use
  them for paid work. The license of each model is in `configs/models.yaml`.
- **Keep client data local.** It goes in `data/private/`, never in git, and never through
  TimeGPT (which sends data to Nixtla's servers) without the client's permission.
- **Don't publish raw price data.** The repo is public, and data providers such as Yahoo
  don't allow redistribution.
- **Keep secrets in `.env`.** For example, `FRED_API_KEY=...` and `NIXTLA_API_KEY=...`.

## 6. Where things are

| Folder | What's in it |
| --- | --- |
| `src/trustcast/data/` | Data pipeline: sources, ETL, preparation |
| `src/trustcast/models/` | One adapter per model, plus the registry |
| `src/trustcast/evaluation/` | Back-testing, metrics, scorecards |
| `src/trustcast/uncertainty/` | Conformal intervals, ensembles, calibration checks |
| `configs/` | Datasets, models, scorecard settings |
| `scripts/` | Command-line tools: `etl.py`, `forecast.py`, `run_scorecard.py` |
| `data/` | Input data (see `data/README.md`) |
| `archive/` | Published scorecards, the public record |
| `experiments/` | One folder per research study |
| `docs/` | The website |

## 7. Getting help

Open an issue on GitHub for bugs and ideas, or ask in the team chat.

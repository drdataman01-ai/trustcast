# Data

All input data lives here. Programs read it through `trustcast.data.store`,
never by downloading on the fly, so every run uses the same files.

```
data/
  macro/                 one CSV per series: date,value
    gdp.csv
    unemployment.csv
    vintages/            every release of every value: date,realtime_start,value
  <next domain>/
  private/               client and partner data — never committed
```

File names match the series names in `configs/domains/<domain>.yaml`.

## Filling it

```bash
python scripts/fetch_data.py --domain macro              # latest values from FRED
python scripts/fetch_data.py --domain macro --vintages   # plus releases (needs FRED_API_KEY in .env)
```

To add your own series, save a CSV with a date column then a value column as
`data/<domain>/<name>.csv`.

## What goes in git

- **Committed:** small public data (the FRED macro series), so results can be reproduced.
- **Not committed:** `data/private/` and any file over a few MB. See `.gitignore`.

To keep data outside the repo (an external drive, a shared folder), set
`TRUSTCAST_DATA=/path/to/folder`; the same layout applies there.

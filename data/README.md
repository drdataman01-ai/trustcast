# Data

All input data lives here. Programs read it through `trustcast.data.store`,
never by downloading on the fly, so every run uses the same files.

```
data/
  gdp.csv                    one CSV per series: date,value
  unemployment.csv
  cpi.csv
  ...
  gdp_vintages.csv           optional: every release, date,realtime_start,value
  private/                   client and partner data, never committed
    client_sales.csv
```

A program asks for a series by name (`unemployment`) and gets `data/unemployment.csv`,
or `data/private/unemployment.csv` if there's no public file with that name. Keep
names unique across both.

## Filling it

```bash
python scripts/fetch_data.py               # macro series from FRED (names from configs/domains/macro.yaml)
python scripts/fetch_data.py --vintages    # plus every release (needs FRED_API_KEY in .env)
```

To add your own series, save a CSV with a date column then a value column as
`data/<name>.csv`, or `data/private/<name>.csv` if it must stay private.

## What goes in git

- **Committed:** small public data (the FRED series), so results can be reproduced.
- **Not committed:** everything in `data/private/` (see `.gitignore`), and large files.

To keep data outside the repo (an external drive, a shared folder), set
`TRUSTCAST_DATA=/path/to/folder`; the same layout applies there.

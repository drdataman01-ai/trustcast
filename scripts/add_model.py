"""Checklist for adding a newly released model (target: scored within weeks).

1. Read the model card and license. Record it in configs/models.yaml with
   hf_id, a pinned revision, license and commercial_ok.
2. Write an adapter in src/trustcast/models/ (copy chronos.py) and register it
   in registry.py.
3. Run tests:            pytest
4. Run the domain back-test against the current roster:
       python scripts/run_scorecard.py --domain macro --models <new> seasonal-naive arima
5. Open a pull request with the config, adapter and results.
"""

if __name__ == "__main__":
    print(__doc__)

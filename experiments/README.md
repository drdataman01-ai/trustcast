# Experiments

One folder per study, e.g. `2027-02-macro-lora-chronos/`, containing:

- `config.yaml` — models, series, back-test settings, random seeds
- `run.py` — the script that produces every number in the write-up
- `results/` — small result tables (large outputs stay in `outputs/`)
- `README.md` — question, method, findings

Studies planned in the proposal: when fine-tuning pays off (data revisions,
short histories, regime breaks), ensembles vs single models, and links between
forecast accuracy and decision quality.

"""Pre-download every Hugging Face model in configs/models.yaml (pinned revisions).

Usage: python scripts/download_models.py [--commercial]
"""

import sys

from huggingface_hub import snapshot_download

from trustcast.models import load_roster

commercial = "--commercial" in sys.argv

for name, spec in load_roster().items():
    hf_id = spec.get("hf_id")
    if not hf_id or hf_id == "TODO":
        continue
    if commercial and not spec.get("commercial_ok", True):
        print(f"skip {name}: research-only license")
        continue
    print(f"downloading {name} ({hf_id})")
    snapshot_download(repo_id=hf_id, revision=spec.get("revision"))

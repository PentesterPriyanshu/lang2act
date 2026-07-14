"""Deploy the demo to Hugging Face Spaces.

Bundles this directory plus the lang2act package into a Space repo and
uploads it. Requires a logged-in HF account (huggingface-cli login) or
HF_TOKEN in the environment.

Run: .venv/bin/python -m space.deploy [--space-name lang2act]
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).parent          # space/
REPO = ROOT.parent                    # lang2act repo root


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--space-name", default="lang2act")
    args = parser.parse_args()

    api = HfApi()
    user = api.whoami()["name"]
    repo_id = f"{user}/{args.space_name}"
    print(f"deploying to Space: {repo_id}")

    api.create_repo(repo_id, repo_type="space", space_sdk="gradio", exist_ok=True)

    with tempfile.TemporaryDirectory() as td:
        stage = Path(td)
        # space files (app, assets, README card, requirements, packages)
        for item in ROOT.iterdir():
            if item.name in {"__pycache__", "deploy.py", "record_episode.py", "__init__.py"}:
                continue
            dest = stage / item.name
            shutil.copytree(item, dest) if item.is_dir() else shutil.copy2(item, dest)
        # the lang2act package itself (robot layer imports)
        shutil.copytree(REPO / "lang2act", stage / "lang2act",
                        ignore=shutil.ignore_patterns("__pycache__"))
        api.upload_folder(repo_id=repo_id, repo_type="space", folder_path=str(stage),
                          commit_message="deploy lang2act demo")

    print(f"done: https://huggingface.co/spaces/{repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Build on a large disk using a small Linux helper container, without a GPU.

The locked wheel set is the one verified on AMD Python 3.14; model revisions are
also immutable. No credentials, privileged container, or registry push required.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser()
parser.add_argument("--work", type=Path, required=True)
args = parser.parse_args()
args.work.mkdir(parents=True, exist_ok=True)
temporary = args.work / "tmp"
temporary.mkdir(exist_ok=True)
os.environ["TMPDIR"] = str(temporary)
os.environ["HF_HOME"] = str(args.work / "hf-cache")
os.environ["HF_XET_CACHE"] = str(args.work / "xet-cache")
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"
os.environ["HF_HUB_ETAG_TIMEOUT"] = "60"
sys.path.insert(0, str(ROOT / "scripts"))
from build_without_docker import BASE, command, crane_binary

# Verify access to the large base before downloading any weights.
crane, _ = crane_binary(args.work)
command([crane, "config", "--platform", "linux/amd64", BASE], stdout=subprocess.DEVNULL)
command([sys.executable, "-m", "pip", "install", "--no-cache-dir", "huggingface-hub==0.36.2"])
command([sys.executable, ROOT / "scripts/download_models.py", "--destination", args.work / "models"])
command([sys.executable, ROOT / "scripts/build_without_docker.py", "--work", args.work,
         "--models", args.work / "models", "--vendor-lock", ROOT / "runtime-pins.txt"])

"""Build/setup-time model download, pinned to immutable Hugging Face revisions."""
import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import snapshot_download

parser = argparse.ArgumentParser()
parser.add_argument("--lock", type=Path, default=Path(__file__).resolve().parents[1] / "models.lock.json")
parser.add_argument("--destination", type=Path, default=Path("/models"))
args = parser.parse_args()
manifest = json.loads(args.lock.read_text(encoding="utf-8"))
for name, model in manifest.items():
    target = args.destination / name
    snapshot_download(
        model["repo"], revision=model["revision"], local_dir=target, max_workers=4,
        allow_patterns=["*.json", "*.txt", "*.safetensors", "1_Pooling/*", "README.md", "LICENSE*"],
    )
    if not list(target.glob("*.safetensors")):
        raise RuntimeError(f"No weights downloaded for {name}")
    files = []
    for path in sorted(target.rglob("*")):
        if path.is_file() and ".cache" not in path.parts:
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    digest.update(block)
            files.append({"file": path.relative_to(target).as_posix(), "bytes": path.stat().st_size,
                          "sha256": digest.hexdigest()})
    (target / "download-manifest.json").write_text(json.dumps({**model, "files": files}, indent=2), encoding="utf-8")
    print(f"Downloaded and hashed {name}: {sum(f['bytes'] for f in files) / 2**30:.2f} GiB", flush=True)

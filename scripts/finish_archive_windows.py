"""Finish crane's partial Docker archive using native Windows disk I/O.

Crane has already written the final image config, every base layer, and the
header of the last application layer. Only that layer's remaining bytes and
the Docker manifest are appended here. Both compressed and uncompressed layer
hashes are then verified. No image is loaded or published.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import tarfile

ROOT = Path(__file__).resolve().parents[1]
BASE = "rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0"


class HashingReader:
    def __init__(self, stream):
        self.stream = stream
        self.digest = hashlib.sha256()

    def read(self, size=-1):
        block = self.stream.read(size)
        self.digest.update(block)
        return block


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    args = parser.parse_args()
    work = args.work.resolve()
    if work == ROOT or work in ROOT.parents or ROOT in work.parents:
        raise ValueError("Use the dedicated build directory outside the source tree")
    image, application = work / "amd-match3.tar", work / "application.tar.gz"
    base_config = json.loads((work / "base-config.json").read_text())
    base_manifest = json.loads((work / "base-manifest.json").read_text())
    with tarfile.open(image) as archive:
        config_member = archive.next()
        config_bytes = archive.extractfile(config_member).read()
        config_digest = "sha256:" + hashlib.sha256(config_bytes).hexdigest()
        if config_member.name != config_digest:
            raise ValueError("Image config hash mismatch")
        config = json.loads(config_bytes)
        layers = []
        for _ in range(len(base_manifest["layers"]) + 1):
            member = archive.next()
            if member is None:
                raise ValueError("Application layer header has not been written yet")
            layers.append(member)
    if [m.name.removesuffix(".tar.gz") for m in layers[:-1]] != [m["digest"].removeprefix("sha256:") for m in base_manifest["layers"]]:
        raise ValueError("Written base layer inventory differs from the mandatory base")
    if config["rootfs"]["diff_ids"][:-1] != base_config["rootfs"]["diff_ids"]:
        raise ValueError("Mandatory base diff IDs were not preserved")
    last = layers[-1]
    if last.size != application.stat().st_size or not last.name.endswith(".tar.gz"):
        raise ValueError("Unexpected final application layer")
    present = min(image.stat().st_size - last.offset_data, last.size)
    if present < 0:
        raise ValueError("Archive does not contain the complete application header")
    print(f"Resuming application layer: {present / 2**30:.2f}/{last.size / 2**30:.2f} GiB", flush=True)
    with image.open("r+b", buffering=8 * 1024 * 1024) as target, application.open("rb", buffering=8 * 1024 * 1024) as source:
        target.seek(last.offset_data + present)
        source.seek(present)
        shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
        target.truncate()
        target.write(b"\0" * ((-last.size) % 512 + 1024))
        target.flush()
        os.fsync(target.fileno())
    manifest = [{"Config": config_member.name, "RepoTags": ["amd-match3:local"], "Layers": [m.name for m in layers]}]
    body = json.dumps(manifest, separators=(",", ":")).encode()
    with tarfile.open(image, "a", format=tarfile.PAX_FORMAT) as archive:
        member = tarfile.TarInfo("manifest.json")
        member.size, member.mode, member.mtime = len(body), 0o644, 0
        archive.addfile(member, io.BytesIO(body))
    print("Verifying every compressed digest, diff ID and total uncompressed size...", flush=True)
    sizes, compressed_digests = [], []
    with tarfile.open(image) as archive:
        for i, member in enumerate(layers):
            stream = archive.extractfile(member.name)
            marker = stream.read(2)
            stream.seek(0)
            compressed = HashingReader(stream)
            content = gzip.GzipFile(fileobj=compressed) if marker == b"\x1f\x8b" else compressed
            digest, size = hashlib.sha256(), 0
            for block in iter(lambda: content.read(8 * 1024 * 1024), b""):
                digest.update(block)
                size += len(block)
            if "sha256:" + digest.hexdigest() != config["rootfs"]["diff_ids"][i]:
                raise ValueError(f"Uncompressed layer hash mismatch: {i}")
            compressed_digest = "sha256:" + compressed.digest.hexdigest()
            if compressed_digest.removeprefix("sha256:") != member.name.removesuffix(".tar.gz"):
                raise ValueError(f"Compressed layer hash mismatch: {i}")
            compressed_digests.append(compressed_digest)
            sizes.append(size)
            print(f"Layer {i + 1}/{len(layers)}: {size / 2**30:.3f} GiB; both hashes verified", flush=True)
    total = sum(sizes)
    if total > 60 * 2**30:
        raise ValueError("Image exceeds the 60 GiB limit")
    app_hashes = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in (ROOT / "app").rglob("*") if p.is_file() and "__pycache__" not in p.parts}
    provenance = ROOT / "reports/validation-provenance.json"
    if provenance.exists() and app_hashes != json.loads(provenance.read_text())["app_files_sha256"]:
        raise ValueError("Packaged source differs from the GPU-tested source")
    report = {"kind": "docker_archive_build", "created": datetime.now(timezone.utc).isoformat(),
              "archive": str(image), "base": BASE, "base_manifest": base_manifest,
              "config_digest": config_digest, "base_diff_ids": base_config["rootfs"]["diff_ids"],
              "final_diff_ids": config["rootfs"]["diff_ids"], "base_layers_preserved": True,
              "all_uncompressed_hashes_verified": True, "all_compressed_hashes_verified": True,
              "compressed_layer_digests": compressed_digests,
              "uncompressed_bytes": total, "uncompressed_gib": total / 2**30,
              "layer_uncompressed_bytes": sizes, "archive_bytes": image.stat().st_size,
              "crane": json.loads((work / "crane-release.json").read_text()),
              "models": json.loads((ROOT / "models.lock.json").read_text()),
              "vendor_versions": (work / "vendor-versions.txt").read_text().splitlines(),
              "application_fingerprint": (work / "application-fingerprint.txt").read_text().strip(),
              "app_source_sha256": app_hashes,
              "archive_completion": "Final config and all layer headers written by official crane; native Windows I/O resumed the final payload and added manifest.json.",
              "limitation": "Archive integrity checked; a full container GPU launch remains unverified."}
    (work / "build-report.json").write_text(json.dumps(report, indent=2))
    print(f"IMAGE_READY: {image}; uncompressed={total / 2**30:.3f} GiB", flush=True)


if __name__ == "__main__":
    main()

"""Read the large archive natively on Windows, validate imports in Linux.

The official checksum-verified Windows crane binary reads an ephemeral,
read-only loopback view of the saved image. A tar filter omits weights and
GPU development assets, then binary pipes feed a network-disabled Linux
extractor. No Docker
image load, GPU inference, registry write, or Docker settings change occurs.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import tarfile
import urllib.request
import uuid
import zipfile
from archive_registry import ArchiveRegistry
from filter_import_check import filter_tar

ROOT = Path(__file__).resolve().parents[1]


def fetch(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "amd-mc3-verifier"}), timeout=120) as response:
        return response.read()


def windows_crane(work):
    executable = work / "crane.exe"
    proof_path = work / "crane-windows-release.json"
    if executable.exists() and proof_path.exists():
        return executable
    release = json.loads((work / "crane-release.json").read_text())["release"]
    metadata = json.loads(fetch("https://api.github.com/repos/google/go-containerregistry/releases/tags/" + release))
    asset = next(a for a in metadata["assets"] if "windows_x86_64" in a["name"].lower())
    url = asset["browser_download_url"]
    if not url.startswith("https://github.com/google/go-containerregistry/releases/download/"):
        raise ValueError("Unexpected official crane source")
    blob = fetch(url)
    checksum = "sha256:" + hashlib.sha256(blob).hexdigest()
    if asset.get("digest") != checksum:
        raise ValueError("Official Windows crane release checksum mismatch")
    if asset["name"].endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(blob)) as archive:
            binary = archive.read("crane.exe")
    else:
        with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as archive:
            binary = archive.extractfile("crane.exe").read()
    executable.write_bytes(binary)
    proof_path.write_text(json.dumps({"release": release, "url": url, "sha256": checksum}, indent=2))
    return executable


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--check-registry-only", action="store_true")
    parser.add_argument("--resume-rootfs", help="Dedicated rootfs-* directory to resume with per-file SHA256 checks")
    args = parser.parse_args()
    if os.name != "nt":
        raise RuntimeError("Use verify_image_filesystem.py directly on Linux")
    work = args.work.resolve()
    crane = windows_crane(work)
    if args.prepare_only:
        print("Official Windows crane is ready", flush=True)
        return
    if shutil.disk_usage(work).free < 85 * 2**30:
        raise RuntimeError("At least 85 GiB free space is required for export and extraction")
    root_name = args.resume_rootfs or "rootfs-" + uuid.uuid4().hex[:10]
    if not re.fullmatch(r"rootfs-[0-9a-f]{10}", root_name):
        raise ValueError("Invalid dedicated rootfs directory name")
    existing = work / root_name if args.resume_rootfs else None
    if existing and (not existing.is_dir() or existing.is_symlink()):
        raise ValueError("Resume rootfs must be an existing real directory in --work")
    common = ["docker", "run", "--rm", "--network", "none",
              "--mount", f"type=bind,source={work},target=/build",
              "--mount", f"type=bind,source={ROOT},target=/src,readonly",
              "-w", "/src"]
    with ArchiveRegistry(work / "amd-match3.tar") as registry:
        # This endpoint binds ONLY to 127.0.0.1, serves existing verified bytes,
        # and is destroyed after the check. Nothing is published externally.
        subprocess.run([str(crane), "config", registry.reference], check=True, stdout=subprocess.DEVNULL)
        if args.check_registry_only:
            print("LOOPBACK_ARCHIVE_READ_CHECK_PASSED", flush=True)
            return
        print("Exporting verified bytes over loopback; transferring only import-test files to Linux...", flush=True)
        exporter = subprocess.Popen([str(crane), "export", registry.reference, "-"], stdout=subprocess.PIPE)
        extractor = subprocess.Popen(common + ["--name", "mc3-image-extract-" + uuid.uuid4().hex[:10], "-i", "python:3.13-slim-bookworm",
                                     "python3", "scripts/verify_image_filesystem.py", "--work", "/build",
                                     "--extract", "/build/" + root_name] + (["--resume-extract"] if existing else []), stdin=subprocess.PIPE)
        try:
            inventory = filter_tar(exporter.stdout, extractor.stdin, existing_root=existing)
            extractor.stdin.close()
            exporter.stdout.close()
            extract_status = extractor.wait()
            export_status = exporter.wait()
        except BaseException:
            exporter.terminate()
            extractor.terminate()
            exporter.wait()
            extractor.wait()
            raise
        (work / "import-filter-report.json").write_text(json.dumps(inventory, indent=2))
    if extract_status or export_status:
        raise RuntimeError(f"Archive export/extraction failed: {export_status}/{extract_status}")
    devices = ["--tmpfs", f"/build/{root_name}/dev:rw,nosuid,dev,noexec,size=4m",
               "--tmpfs", f"/build/{root_name}/dev/shm:rw,nosuid,nodev,noexec,size=64m"]
    for name in ("null", "zero", "random", "urandom"):
        devices += ["--device", f"/dev/{name}:/build/{root_name}/dev/{name}:rw"]
    subprocess.run(common + devices + ["--name", "mc3-image-check", "python:3.13-slim-bookworm", "python3",
                             "scripts/verify_image_filesystem.py", "--work", "/build",
                             "--existing-rootfs", "/build/" + root_name], check=True)


if __name__ == "__main__":
    main()

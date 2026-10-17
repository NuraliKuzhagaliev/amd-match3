"""Create a Docker image archive on the GPU host without a Docker daemon.

This alternative is for the supplied Jupyter environment, which has no Docker.
It appends ONE application/weights layer to the unchanged mandated base using
Google's crane, preserving both compressed layer digests and rootfs diff IDs.
The regular Dockerfile remains the normal/recommended build route.

No registry push is performed. Run on Linux with sufficient TEMPORARY disk space.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
from importlib import metadata
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
import uuid

BASE = "rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0"
ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ["transformers", "accelerate", "pypdf", "python-docx", "openpyxl", "pypdfium2",
            "filelock", "huggingface-hub", "numpy", "packaging", "pyyaml", "regex", "requests",
            "safetensors", "tokenizers", "tqdm", "psutil", "typing-extensions", "fsspec",
            "certifi", "charset-normalizer", "idna", "urllib3", "lxml", "et-xmlfile", "Pillow"]


def fetch(url: str) -> bytes:
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "amd-mc3-builder"}), timeout=120) as response:
        return response.read()


def crane_binary(directory: Path) -> tuple[Path, dict]:
    executable = directory / "crane"
    release_path = directory / "crane-release.json"
    if executable.exists() and release_path.exists():
        return executable, json.loads(release_path.read_text())
    release = json.loads(fetch("https://api.github.com/repos/google/go-containerregistry/releases/latest"))
    asset = next(a for a in release["assets"] if a["name"].lower() == "go-containerregistry_linux_x86_64.tar.gz")
    url = asset["browser_download_url"]
    if not url.startswith("https://github.com/google/go-containerregistry/releases/download/"):
        raise ValueError("Unexpected crane binary source")
    blob = fetch(url)
    actual = "sha256:" + hashlib.sha256(blob).hexdigest()
    if asset.get("digest") != actual:
        raise ValueError("Official release checksum missing or mismatched")
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as archive:
        member = archive.getmember("crane")
        with archive.extractfile(member) as stream:
            executable.write_bytes(stream.read())
    executable.chmod(0o755)
    proof = {"release": release["tag_name"], "url": url, "sha256": actual}
    release_path.write_text(json.dumps(proof, indent=2))
    return executable, proof


def command(args, **kwargs):
    print("RUN", " ".join(map(str, args)), flush=True)
    try:
        return subprocess.run(list(map(str, args)), check=True, **kwargs)
    except subprocess.CalledProcessError as error:
        if error.stderr:
            print(error.stderr, file=sys.stderr, flush=True)
        raise


def fixed_metadata(info):
    info.uid = info.gid = 0
    info.uname = info.gname = "root"
    info.mtime = 0
    return info


def source_fingerprint(root: Path, requirements: list[str], models: Path) -> str:
    digest = hashlib.sha256(json.dumps(requirements, sort_keys=True).encode())
    candidates = [p for p in (root / "app").rglob("*")
                  if p.is_file() and "__pycache__" not in p.parts]
    candidates += [root / "models.lock.json", root / "THIRD_PARTY_NOTICES.md"]
    candidates += list((root / "licenses").glob("*"))
    for path in sorted(candidates):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    for path in sorted(models.glob("*/download-manifest.json")):
        digest.update(path.relative_to(models).as_posix().encode())
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--vendor-lock", type=Path,
                        help="Use exact package pins and Python 3.14 Linux wheels without importing local torch")
    args = parser.parse_args()
    work, models = args.work.resolve(), args.models.resolve()
    if work == Path("/") or work == ROOT or ROOT in work.parents or work in ROOT.parents:
        raise ValueError("Use a separate build directory outside the source tree")
    work.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(work).free < 100 * 2**30:
        raise RuntimeError("At least 100 GiB temporary build disk space is required")
    if args.vendor_lock is None:
        import torch
        if not torch.version.hip:
            raise RuntimeError("Use the matching ROCm environment, or provide --vendor-lock")
    crane, crane_proof = crane_binary(work)
    base_config = json.loads(command([crane, "config", "--platform", "linux/amd64", BASE], capture_output=True, text=True).stdout)
    base_manifest = json.loads(command([crane, "manifest", "--platform", "linux/amd64", BASE], capture_output=True, text=True).stdout)
    (work / "base-config.json").write_text(json.dumps(base_config, indent=2))
    (work / "base-manifest.json").write_text(json.dumps(base_manifest, indent=2))
    requirements = ([line.strip() for line in args.vendor_lock.read_text().splitlines()
                     if line.strip() and not line.startswith("#")] if args.vendor_lock
                    else [f"{name}=={metadata.version(name)}" for name in PACKAGES])
    (work / "vendor-versions.txt").write_text("\n".join(requirements))
    requirement_id = hashlib.sha256("\n".join(requirements).encode()).hexdigest()[:16]
    vendor_root = work / "vendor"
    vendor_root.mkdir(exist_ok=True)
    vendor_final = vendor_root / requirement_id
    vendor = vendor_final
    if not (vendor_final / ".complete").exists():
        vendor = vendor_root / (requirement_id + ".partial-" + uuid.uuid4().hex[:8])
        if args.vendor_lock:
            wheelhouse = work / "wheels"
            wheelhouse.mkdir(exist_ok=True)
            pip_args = [sys.executable, "-m", "pip", "download", "--no-cache-dir", "--no-deps",
                        "--only-binary=:all:", "--dest", wheelhouse, "--python-version", "3.14",
                        "--implementation", "cp", "--abi", "cp314", "--abi", "abi3", "--abi", "none"]
            for platform in ("manylinux_2_39_x86_64", "manylinux_2_38_x86_64", "manylinux_2_35_x86_64",
                             "manylinux_2_34_x86_64", "manylinux_2_31_x86_64", "manylinux_2_28_x86_64",
                             "manylinux_2_27_x86_64", "manylinux_2_26_x86_64", "manylinux_2_24_x86_64",
                             "manylinux_2_17_x86_64", "manylinux2014_x86_64"):
                pip_args.extend(["--platform", platform])
            command(pip_args + requirements)
            vendor.mkdir(exist_ok=True)
            for wheel in wheelhouse.glob("*.whl"):
                with zipfile.ZipFile(wheel) as archive:
                    for member in archive.infolist():
                        relative = Path(member.filename)
                        if member.is_dir():
                            continue
                        if relative.is_absolute() or ".." in relative.parts:
                            raise ValueError("Unsafe path in wheel")
                        parts = relative.parts
                        if len(parts) > 2 and parts[0].endswith(".data") and parts[1] in {"purelib", "platlib"}:
                            relative = Path(*parts[2:])
                        target = vendor / relative
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_bytes(archive.read(member))
        else:
            command([sys.executable, "-m", "pip", "install", "--no-cache-dir", "--no-deps", "--target", vendor, *requirements])
        (vendor / ".complete").write_text("installed")
        os.replace(vendor, vendor_final)
        vendor = vendor_final
    layer = work / "application.tar.gz"
    fingerprint = source_fingerprint(ROOT, requirements, models)
    stamp = work / "application-fingerprint.txt"
    cached = stamp.read_text().strip() if stamp.exists() else None
    if not layer.exists() or cached != fingerprint:
        print("Creating application layer with model weights...", flush=True)
        partial = layer.with_suffix(".partial")
        with partial.open("wb", buffering=8 * 1024 * 1024) as raw:
            # Safetensors are nearly incompressible. Stored gzip avoids spending
            # an hour compressing weights on resource-limited helper hosts.
            with gzip.GzipFile(fileobj=raw, mode="wb", compresslevel=0, mtime=0) as compressed:
                with tarfile.open(fileobj=compressed, mode="w|", bufsize=4 * 1024 * 1024,
                                  copybufsize=4 * 1024 * 1024) as archive:
                    for directory in ("app", "app/corpus", "app/output", "app/index", "models"):
                        info = tarfile.TarInfo(directory)
                        info.type, info.mode = tarfile.DIRTYPE, 0o755
                        archive.addfile(fixed_metadata(info))
                    for parent, prefix in [(ROOT / "app", "app"), (vendor, "app/vendor"), (models, "models")]:
                        for path in sorted(parent.rglob("*")):
                            if not path.is_file() or path.is_symlink() or any(x in path.parts for x in ("__pycache__", ".cache")):
                                continue
                            archive.add(path, arcname=prefix + "/" + path.relative_to(parent).as_posix(),
                                        recursive=False, filter=fixed_metadata)
                    archive.add(ROOT / "models.lock.json", arcname="app/models.lock.json", filter=fixed_metadata)
                    archive.add(ROOT / "THIRD_PARTY_NOTICES.md", arcname="app/THIRD_PARTY_NOTICES.md", filter=fixed_metadata)
                    for path in (ROOT / "licenses").glob("*"):
                        archive.add(path, arcname="app/licenses/" + path.name, filter=fixed_metadata)
        os.replace(partial, layer)
        if source_fingerprint(ROOT, requirements, models) != fingerprint:
            raise RuntimeError("Source changed while the application layer was being built; rerun the build")
        stamp.write_text(fingerprint)
    image = work / "amd-match3.tar"
    inherited_path = next((x.split("=", 1)[1] for x in base_config["config"].get("Env", []) if x.startswith("PYTHONPATH=")), "")
    env = {"PYTHONPATH": "/app/vendor" + (":" + inherited_path if inherited_path else ""),
           "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
           "TOKENIZERS_PARALLELISM": "false", "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "4",
           "MC3_MODEL_DIR": "/models/vision", "MC3_EMBED_DIR": "/models/embedding"}
    args = [crane, "mutate", "--platform", "linux/amd64", "--append", layer,
            "--cmd", "python3,-m,mc3.worker", "--entrypoint=", "--workdir", "/app", "--user", "0",
            "--output", image, "--tag", "amd-match3:local"]
    for key, value in env.items():
        args.extend(["--env", key + "=" + value])
    args.append(BASE)
    command(args)
    print("Validating layer identity and exact uncompressed size...", flush=True)
    with tarfile.open(image) as archive:
        saved_manifest = json.load(archive.extractfile("manifest.json"))[0]
        config_bytes = archive.extractfile(saved_manifest["Config"]).read()
        final_config = json.loads(config_bytes)
        inherited = base_config["rootfs"]["diff_ids"]
        final_ids = final_config["rootfs"]["diff_ids"]
        if final_ids[:-1] != inherited:
            raise RuntimeError("Base layer identity was not preserved")
        layer_sizes = []
        for i, member in enumerate(saved_manifest["Layers"]):
            stream = archive.extractfile(member)
            marker = stream.read(2)
            stream.seek(0)
            source = gzip.GzipFile(fileobj=stream) if marker == b"\x1f\x8b" else stream
            digest, size = hashlib.sha256(), 0
            for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
                digest.update(block)
                size += len(block)
            if "sha256:" + digest.hexdigest() != final_ids[i]:
                raise RuntimeError(f"Uncompressed layer checksum mismatch at {i}")
            layer_sizes.append(size)
            print(f"Layer {i + 1}/{len(final_ids)}: {size / 2**30:.3f} GiB verified", flush=True)
    total = sum(layer_sizes)
    if total > 60 * 2**30:
        raise RuntimeError(f"Image exceeds size limit: {total / 2**30:.2f} GiB")
    proof = {"kind": "docker_archive_build", "created": datetime.now(timezone.utc).isoformat(),
             "archive": str(image), "base": BASE, "base_manifest": base_manifest,
             "config_digest": "sha256:" + hashlib.sha256(config_bytes).hexdigest(),
             "base_diff_ids": inherited, "final_diff_ids": final_ids,
             "base_layers_preserved": True, "all_uncompressed_hashes_verified": True,
             "uncompressed_bytes": total, "uncompressed_gib": total / 2**30,
             "layer_uncompressed_bytes": layer_sizes, "archive_bytes": image.stat().st_size,
             "crane": crane_proof, "models": json.loads((ROOT / "models.lock.json").read_text()),
             "vendor_versions": requirements,
             "application_fingerprint": fingerprint,
             "app_source_sha256": {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                                   for path in (ROOT / "app").rglob("*")
                                   if path.is_file() and "__pycache__" not in path.parts},
             "limitation": "Archive assembled and validated; must still be run as a container on AMD GPU."}
    (work / "build-report.json").write_text(json.dumps(proof, indent=2))
    print(f"IMAGE_READY {image} uncompressed={total / 2**30:.2f} GiB", flush=True)


if __name__ == "__main__":
    main()

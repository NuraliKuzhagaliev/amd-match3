"""Transfer exact import-test files, omitting weights and GPU development data."""
import io
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile


def filter_tar(source, target, existing_root=None):
    inventory = {"skipped_weight_payloads": [], "omitted_gpu_development_assets": [],
                 "omitted_test_and_bytecode_assets": [], "reused_files_sha256": {},
                 "reused_directories": 0, "resumed_partial_filesystem": existing_root is not None}
    existing_root = Path(existing_root).resolve() if existing_root else None

    def category(name):
        name = name.lstrip("./")
        if name.startswith(("models/vision/", "models/embedding/")) and name.endswith(".safetensors"):
            return "skipped_weight_payloads"
        parts = PurePosixPath(name).parts
        if "__pycache__" in parts or ("site-packages" in parts and "tests" in parts):
            return "omitted_test_and_bytecode_assets"
        rocm = (len(parts) > 1 and parts[0] == "opt" and parts[1].startswith("rocm")) or any(p.startswith("_rocm_sdk") for p in parts)
        if rocm or ("site-packages" in parts and "torch" in parts):
            if "include" in parts or PurePosixPath(name).suffix in {".a", ".co", ".hsaco", ".dat"}:
                return "omitted_gpu_development_assets"
        return None

    with tarfile.open(fileobj=source, mode="r|", bufsize=8 * 1024 * 1024,
                      copybufsize=8 * 1024 * 1024) as incoming:
        with tarfile.open(fileobj=target, mode="w|", bufsize=8 * 1024 * 1024,
                          copybufsize=8 * 1024 * 1024) as outgoing:
            for member in incoming:
                key = category(member.name)
                if key:
                    inventory[key].append(member.name.lstrip("./"))
                    continue
                if existing_root:
                    candidate = existing_root.joinpath(*PurePosixPath(member.name.lstrip("./")).parts)
                    try:
                        resolved = candidate.resolve(strict=True)
                        safe = resolved.is_relative_to(existing_root) and not candidate.is_symlink()
                    except (OSError, RuntimeError):
                        safe = False
                    if safe and member.isdir() and candidate.is_dir():
                        inventory["reused_directories"] += 1
                        continue
                    if safe and member.isfile() and candidate.is_file() and candidate.stat().st_size == member.size:
                        with candidate.open("rb") as previous:
                            expected = hashlib.file_digest(previous, "sha256").hexdigest()
                        content = incoming.extractfile(member)
                        digest = hashlib.sha256()
                        for block in iter(lambda: content.read(8 * 1024 * 1024), b""):
                            digest.update(block)
                        if digest.hexdigest() != expected:
                            raise RuntimeError(f"Resume file hash mismatch; remove the partial cache file and retry: {candidate}")
                        inventory["reused_files_sha256"][member.name] = expected
                        continue
                outgoing.addfile(member, incoming.extractfile(member) if member.isfile() else None)
            body = json.dumps(inventory).encode()
            marker = tarfile.TarInfo(".mc3-filter-inventory.json")
            marker.size, marker.mode = len(body), 0o644
            outgoing.addfile(marker, io.BytesIO(body))
    return inventory

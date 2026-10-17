"""Verify imports inside the assembled image's actual filesystem, offline.

This is a packaging check, NOT a GPU container benchmark. Extraction occurs in
a chroot so absolute container symlinks cannot escape into the helper host.
The original archive is never altered or flattened for submission.
"""
from __future__ import annotations

import argparse
import bz2
import gzip
import hashlib
import json
import lzma
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tarfile
import uuid


def extract_inside(root: Path, resume=False) -> None:
    # Import codecs and compression handlers before entering the empty root.
    import encodings.utf_8
    import encodings.ascii
    import pwd
    import grp
    root.mkdir(parents=True, exist_ok=resume)
    os.chroot(root)
    os.chdir("/")
    skipped = []
    with tarfile.open(fileobj=sys.stdin.buffer, mode="r|", bufsize=8 * 1024 * 1024,
                      copybufsize=8 * 1024 * 1024) as archive:
        for member in archive:
            name = member.name.lstrip("./")
            if name.startswith(("models/vision/", "models/embedding/")) and name.endswith(".safetensors"):
                # Layer hashes already cover every weight byte. This check only
                # imports dependencies and loads processor/tokenizer configs.
                skipped.append(name)
                continue
            archive.extract(member, "/", set_attrs=False, numeric_owner=True, filter="fully_trusted")
            if member.isfile() and member.mode & 0o111:
                os.chmod(member.name, 0o755)
    inventory_path = Path("/.mc3-filter-inventory.json")
    inventory = json.loads(inventory_path.read_text()) if inventory_path.exists() else {}
    inventory["skipped_weight_payloads"] = inventory.get("skipped_weight_payloads", []) + skipped
    Path("/.mc3-extraction-check.json").write_text(json.dumps(inventory))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--extract", type=Path)
    parser.add_argument("--resume-extract", action="store_true")
    parser.add_argument("--existing-rootfs", type=Path,
                        help="Validate an already extracted rootfs produced by this script")
    args = parser.parse_args()
    if args.extract:
        target = args.extract.resolve()
        if target.parent != args.work.resolve() or not target.name.startswith("rootfs-"):
            raise ValueError("Extraction target must be a dedicated rootfs-* directory inside --work")
        extract_inside(target, resume=args.resume_extract)
        return
    work = args.work.resolve()
    (work / "tmp").mkdir(exist_ok=True)
    os.environ["TMPDIR"] = str(work / "tmp")
    if shutil.disk_usage(work).free < 55 * 2**30:
        raise RuntimeError("Not enough space for an extracted image filesystem")
    image = work / "amd-match3.tar"
    root = (args.existing_rootfs.resolve() if args.existing_rootfs
            else work / ("rootfs-" + uuid.uuid4().hex[:10]))
    if root.parent != work or not root.name.startswith("rootfs-"):
        raise ValueError("Rootfs must be a dedicated rootfs-* directory directly inside --work")
    if not args.existing_rootfs:
        print(f"Extracting actual image filesystem into {root}", flush=True)
        with image.open("rb") as stream:
            exporter = subprocess.Popen([str(work / "crane"), "export", "-", "-"],
                                        stdin=stream, stdout=subprocess.PIPE)
            extractor = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--work", str(work),
                                          "--extract", str(root)], stdin=exporter.stdout)
            exporter.stdout.close()
            extract_result = extractor.wait()
            if extract_result:
                exporter.terminate()
            export_result = exporter.wait()
            if extract_result or export_result:
                raise RuntimeError(f"Filesystem export/extraction failed: {export_result}/{extract_result}")
    extraction = json.loads((root / ".mc3-extraction-check.json").read_text())
    skipped = extraction["skipped_weight_payloads"]
    if sum(name.startswith("models/vision/") for name in skipped) != 4 or sum(name.startswith("models/embedding/") for name in skipped) != 1:
        raise RuntimeError("Unexpected model weight shard inventory")
    with tarfile.open(image) as archive:
        manifest = json.load(archive.extractfile("manifest.json"))[0]
        config_bytes = archive.extractfile(manifest["Config"]).read()
        config = json.loads(config_bytes)["config"]
    environment = [item for item in config.get("Env", []) if not item.startswith(("HF_HUB_OFFLINE=", "TRANSFORMERS_OFFLINE="))]
    environment += ["HF_HUB_OFFLINE=1", "TRANSFORMERS_OFFLINE=1"]
    # Docker normally supplies these devices at runtime; a bare chroot does not.
    # On Windows bind mounts the wrapper supplies a private tmpfs /dev and the
    # same standard devices. On an ordinary Linux filesystem, mknod suffices.
    (root / "dev").mkdir(exist_ok=True)
    for name, minor in (("null", 3), ("zero", 5), ("random", 8), ("urandom", 9)):
        device = root / "dev" / name
        if not device.exists():
            os.mknod(device, stat.S_IFCHR | 0o666, os.makedev(1, minor))
        details = device.stat()
        if not stat.S_ISCHR(details.st_mode) or details.st_rdev != os.makedev(1, minor):
            raise RuntimeError(f"Expected a real standard virtual device: {device}")
    (root / "dev/shm").mkdir(exist_ok=True)
    with (root / "dev/urandom").open("rb", buffering=0) as entropy:
        assert len(entropy.read(1)) == 1
    program = """
import json, hashlib, importlib.metadata as metadata
import torch, pypdf, docx, openpyxl, pypdfium2, numpy, PIL
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, AutoTokenizer, AutoModel
import mc3.pipeline, mc3.worker
assert torch.version.hip and '+rocm' in torch.__version__
processor=AutoProcessor.from_pretrained('/models/vision', local_files_only=True, trust_remote_code=False)
tokenizer=AutoTokenizer.from_pretrained('/models/embedding', local_files_only=True, trust_remote_code=False)
from pathlib import Path
assert Path('/app/app.py').is_file()
assert Path('/app/licenses/APACHE-2.0.txt').is_file()
result={'kind':'offline_image_filesystem_import_check','torch':torch.__version__,'hip':torch.version.hip,
'transformers':metadata.version('transformers'),'python':__import__('sys').version,
'processor_loaded_offline':True,'tokenizer_loaded_offline':True,'runtime_imports_passed':True,
'limitation':'No AMD GPU is exposed. Weight payloads and optional GPU development/kernel assets may be omitted from this import check and are inventoried separately; all image bytes are covered by archive layer hashes. Inference was tested separately in AMD Jupyter.'}
app_files=[Path('/app/app.py'),Path('/app/requirements.txt'),*Path('/app/mc3').glob('*.py')]
result['app_source_sha256']={'app/'+p.relative_to('/app').as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in app_files}
print(json.dumps(result))
"""
    result = subprocess.run(["chroot", str(root), "/usr/bin/env", "-i", *environment,
                             "PYTHONPATH=/app/vendor:/app", "python3", "-c", program],
                            capture_output=True, text=True, timeout=180)
    print(result.stdout, flush=True)
    if result.stderr:
        print(result.stderr, file=sys.stderr, flush=True)
    if result.returncode:
        raise RuntimeError(f"Image filesystem import check failed: {result.returncode}")
    report = json.loads(result.stdout.strip().splitlines()[-1])
    report["config_digest"] = "sha256:" + hashlib.sha256(config_bytes).hexdigest()
    report["archive_bytes"] = image.stat().st_size
    report["standard_virtual_devices_provided"] = True
    provenance_path = Path(__file__).resolve().parents[1] / "reports/validation-provenance.json"
    if provenance_path.exists():
        if report["app_source_sha256"] != json.loads(provenance_path.read_text())["app_files_sha256"]:
            raise RuntimeError("Extracted image application differs from recorded GPU-tested source")
        report["app_matches_recorded_gpu_provenance"] = True
    report["rootfs"] = str(root)
    report.update(extraction)
    report["diagnostic_metadata"] = "Extraction normalizes ownership/timestamps and preserves executable permission; the original Docker archive and its metadata are unchanged."
    (work / "filesystem-check.json").write_text(json.dumps(report, indent=2))
    print("OFFLINE_IMAGE_FILESYSTEM_CHECK_PASSED", flush=True)


if __name__ == "__main__":
    main()

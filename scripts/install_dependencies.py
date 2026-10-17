"""Install packages without allowing pip to replace the ROCm base stack."""
from importlib import metadata
from pathlib import Path
import subprocess
import sys
import tempfile

import torch

if not torch.version.hip or "+rocm" not in torch.__version__:
    raise SystemExit("Expected ROCm torch from the mandated base image")
before = torch.__version__
constraints = []
for name in ("torch", "torchvision", "torchaudio", "triton", "pytorch-triton-rocm"):
    try:
        constraints.append(f"{name}=={metadata.version(name)}")
    except metadata.PackageNotFoundError:
        pass
with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / "base-constraints.txt"
    path.write_text("\n".join(constraints), encoding="utf-8")
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-cache-dir", "-c", str(path),
                    "-r", sys.argv[1]], check=True)
# A fresh process verifies the installed package rather than the old import.
subprocess.run([sys.executable, "-c", "import torch; "
                f"assert torch.__version__ == {before!r}, torch.__version__; "
                "assert torch.version.hip; print('ROCm torch preserved:', torch.__version__)"], check=True)

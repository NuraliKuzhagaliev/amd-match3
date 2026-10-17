"""Container PID 1: one GPU model, sequential Unix-socket requests."""
from pathlib import Path
import logging
import os
import signal
import socket
import time

from .backend import Backend
from .config import Config
from .io import receive, send
from .pipeline import Pipeline


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    started = time.monotonic()
    cfg = Config()
    pipeline = Pipeline(Backend(cfg), cfg, started)
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    address = Path(cfg.socket)
    if address.exists():
        # Refuse to take over a live socket belonging to another worker.
        probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            probe.connect(cfg.socket)
        except ConnectionRefusedError:
            address.unlink()
        else:
            raise RuntimeError("Another worker is already serving this socket")
        finally:
            probe.close()
    listener.bind(cfg.socket)
    os.chmod(cfg.socket, 0o600)
    listener.listen(8)

    def shutdown(signum, frame):
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    logging.info("READY socket=%s startup=%.2fs", cfg.socket, time.monotonic() - started)
    try:
        while True:
            connection, _ = listener.accept()
            with connection:
                connection.settimeout(cfg.startup_seconds)
                try:
                    message = receive(connection)
                    operation = message.get("op")
                    if operation == "health":
                        torch = pipeline.backend.torch
                        result = {"ok": True, "indexed": pipeline.search is not None,
                                  "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
                                  "hip": torch.version.hip, "pid": os.getpid(),
                                  "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                                  "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30}
                    elif operation == "index":
                        result = pipeline.build(Path(message["corpus"]))
                    elif operation == "answer":
                        result = pipeline.answer(Path(message["corpus"]), message["query"])
                    else:
                        raise ValueError("Unknown operation")
                    send(connection, result)
                except Exception as error:
                    logging.exception("Request failed")
                    try:
                        send(connection, {"error": f"{type(error).__name__}: {error}"})
                    except (OSError, ValueError):
                        pass
    finally:
        listener.close()
        address.unlink(missing_ok=True)


if __name__ == "__main__":
    main()

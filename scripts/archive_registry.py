"""Ephemeral, read-only, loopback-only view of a verified Docker archive."""
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import tarfile
import threading
from urllib.parse import urlsplit


class ArchiveRegistry:
    def __init__(self, image: Path):
        self.image = image
        with tarfile.open(image) as archive:
            saved = json.load(archive.extractfile("manifest.json"))[0]
            config = archive.getmember(saved["Config"])
            config_bytes = archive.extractfile(config).read()
            config_digest = "sha256:" + hashlib.sha256(config_bytes).hexdigest()
            self.blobs = {config_digest: (config.offset_data, config.size)}
            descriptors = []
            for name in saved["Layers"]:
                member = archive.getmember(name)
                digest = "sha256:" + name.removesuffix(".tar.gz")
                self.blobs[digest] = (member.offset_data, member.size)
                descriptors.append({"mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
                                    "size": member.size, "digest": digest})
        self.manifest = json.dumps({"schemaVersion": 2,
            "mediaType": "application/vnd.docker.distribution.manifest.v2+json",
            "config": {"mediaType": "application/vnd.docker.container.image.v1+json", "size": config.size,
                       "digest": config_digest}, "layers": descriptors}, separators=(",", ":")).encode()
        self.digest = "sha256:" + hashlib.sha256(self.manifest).hexdigest()
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_HEAD(self):
                self.send_content(False)

            def do_GET(self):
                self.send_content(True)

            def send_content(self, body):
                path = urlsplit(self.path).path
                if path in {"/v2", "/v2/"}:
                    self.send_response(200)
                    self.send_header("Docker-Distribution-API-Version", "registry/2.0")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                if path in {"/v2/mc3/manifests/local", "/v2/mc3/manifests/" + owner.digest}:
                    self.send_response(200)
                    self.send_header("Content-Type", "application/vnd.docker.distribution.manifest.v2+json")
                    self.send_header("Docker-Content-Digest", owner.digest)
                    self.send_header("Content-Length", str(len(owner.manifest)))
                    self.end_headers()
                    if body:
                        self.wfile.write(owner.manifest)
                    return
                prefix = "/v2/mc3/blobs/"
                digest = path[len(prefix):] if path.startswith(prefix) else None
                if digest not in owner.blobs:
                    self.send_error(404)
                    return
                offset, size = owner.blobs[digest]
                start, end = 0, size - 1
                requested = self.headers.get("Range")
                if requested:
                    match = re.fullmatch(r"bytes=(\d+)-(\d*)", requested)
                    if not match:
                        self.send_error(416)
                        return
                    start = int(match[1])
                    end = min(int(match[2]), end) if match[2] else end
                    if not 0 <= start <= end < size:
                        self.send_error(416)
                        return
                self.send_response(206 if requested else 200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Docker-Content-Digest", digest)
                self.send_header("Content-Length", str(end - start + 1))
                if requested:
                    self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
                self.end_headers()
                if body:
                    try:
                        with owner.image.open("rb", buffering=8 * 1024 * 1024) as stream:
                            stream.seek(offset + start)
                            remaining = end - start + 1
                            while remaining:
                                block = stream.read(min(4 * 1024 * 1024, remaining))
                                if not block:
                                    raise EOFError("Truncated verified archive")
                                self.wfile.write(block)
                                remaining -= len(block)
                    except (BrokenPipeError, ConnectionResetError):
                        pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.reference = f"127.0.0.1:{self.server.server_port}/mc3:local"

    def __enter__(self):
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)

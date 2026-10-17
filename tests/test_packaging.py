from pathlib import Path
import gzip
import hashlib
import io
import json
import subprocess
import tarfile
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_without_docker import source_fingerprint


class PackagingTests(unittest.TestCase):
    def test_code_overlay_preserves_base_and_verifies_layer_payloads(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'source/app').mkdir(parents=True)
            (root/'source/app/app.py').write_text('print("verified")\n')
            payload = b'fixture original layer'
            compressed = gzip.compress(payload, mtime=0)
            name = hashlib.sha256(compressed).hexdigest() + '.tar.gz'
            config = json.dumps({'rootfs': {'type': 'layers', 'diff_ids': ['sha256:' + hashlib.sha256(payload).hexdigest()]},
                                 'config': {'Cmd': ['python3', '/app/app.py']}}).encode()
            config_hash = hashlib.sha256(config).hexdigest()
            with tarfile.open(root/'base.tar', 'w') as archive:
                for path, data in [(config_hash+'.json', config), (name, compressed),
                                   ('manifest.json', json.dumps([{'Config': config_hash+'.json', 'Layers': [name],
                                                                  'RepoTags': ['fixture:old']}]).encode())]:
                    info = tarfile.TarInfo(path)
                    info.size = len(data)
                    archive.addfile(info, io.BytesIO(data))
            (root/'proof.json').write_text(json.dumps({'config_digest': 'sha256:' + config_hash}))
            script = Path(__file__).resolve().parents[1]/'scripts/update_archive.py'
            proc = subprocess.run([sys.executable, str(script), '--base', str(root/'base.tar'),
                                   '--source', str(root/'source'), '--output', str(root/'updated.tar'),
                                   '--base-report', str(root/'proof.json')], capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            report = json.loads((root/'updated.build-report.json').read_text())
            self.assertEqual(len(report['layers']), 2)
            self.assertEqual(report['layers'][0]['diff_id'], 'sha256:' + hashlib.sha256(payload).hexdigest())
            with tarfile.open(root/'updated.tar') as archive:
                self.assertEqual(archive.extractfile(name).read(), compressed)
            self.assertEqual(report['app_sha256']['app.py'], hashlib.sha256((root/'source/app/app.py').read_bytes()).hexdigest())

    def test_layer_cache_changes_with_code_dependencies_or_model_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "app").mkdir()
            (root / "licenses").mkdir()
            (root / "models/vision").mkdir(parents=True)
            for name in ["models.lock.json", "THIRD_PARTY_NOTICES.md"]:
                (root / name).write_text("fixture")
            code = root / "app/app.py"
            code.write_text("value = 1")
            first = source_fingerprint(root, ["package==1"], root / "models")
            code.write_text("value = 2")
            second = source_fingerprint(root, ["package==1"], root / "models")
            self.assertNotEqual(first, second)
            self.assertNotEqual(second, source_fingerprint(root, ["package==2"], root / "models"))
            (root / "models/vision/download-manifest.json").write_text('{"sha":"different"}')
            self.assertNotEqual(second, source_fingerprint(root, ["package==1"], root / "models"))


if __name__ == "__main__":
    unittest.main()

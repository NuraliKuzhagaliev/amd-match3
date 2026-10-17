"""Append a small code-only layer to a verified local Docker archive.

Preserves every original layer and model byte. No registry, daemon, flattening,
or C-drive image import. Rechecks compressed and uncompressed layer identities.
"""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile

BUFFER = 8 * 1024 * 1024


def sha(stream):
    digest, count = hashlib.sha256(), 0
    while block := stream.read(BUFFER):
        digest.update(block)
        count += len(block)
    return digest.hexdigest(), count


def member(name, data):
    info = tarfile.TarInfo(name)
    info.size, info.mode, info.mtime = len(data), 0o644, 0
    return info, io.BytesIO(data)


class HashReader:
    def __init__(self, source):
        self.source, self.digest = source, hashlib.sha256()

    def read(self, size=-1):
        data = self.source.read(size)
        self.digest.update(data)
        return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--base-report', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.resolve() == args.base.resolve():
        raise ValueError('Output must be a new archive; the original is preserved')
    app = args.source.resolve() / 'app'
    paths = sorted(p for p in app.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    code_hashes = {p.relative_to(app).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    layer = args.output.with_suffix('.code-layer.tar.gz')
    with layer.open('wb') as target:
        with gzip.GzipFile(fileobj=target, mode='wb', mtime=0, filename='') as compressed:
            with tarfile.open(fileobj=compressed, mode='w|', format=tarfile.PAX_FORMAT) as archive:
                for path in paths:
                    relative = 'app/' + path.relative_to(app).as_posix()
                    archive.addfile(*member(relative, path.read_bytes()))
    with layer.open('rb') as stream:
        compressed_hash, compressed_size = sha(stream)
    with gzip.open(layer, 'rb') as stream:
        diff_id, added_size = sha(stream)
    base_report = json.loads(args.base_report.read_text(encoding='utf-8'))
    with tarfile.open(args.base, 'r:') as original:
        old_manifest = json.load(original.extractfile('manifest.json'))[0]
        config = json.load(original.extractfile(old_manifest['Config']))
        old_config_digest = hashlib.sha256(original.extractfile(old_manifest['Config']).read()).hexdigest()
        if base_report.get('config_digest') != 'sha256:' + old_config_digest:
            raise ValueError('Base archive does not match its verified build report')
        original_ids = list(config['rootfs']['diff_ids'])
        if len(original_ids) != len(old_manifest['Layers']):
            raise ValueError('Base layer count mismatch')
        config['rootfs']['diff_ids'].append('sha256:' + diff_id)
        config.setdefault('history', []).append({'created': datetime.now(timezone.utc).isoformat(),
                                                'created_by': 'Match 3 verified code update'})
        payload = json.dumps(config, separators=(',', ':')).encode()
        config_name = hashlib.sha256(payload).hexdigest() + '.json'
        new_layer_name = compressed_hash + '.tar.gz'
        manifest = {'Config': config_name, 'RepoTags': ['amd-match3:improved'],
                    'Layers': old_manifest['Layers'] + [new_layer_name]}
        with tarfile.open(args.output, 'w', format=tarfile.PAX_FORMAT, copybufsize=BUFFER) as result:
            result.addfile(*member(config_name, payload))
            for name in old_manifest['Layers']:
                print('COPY', name, flush=True)
                info = original.getmember(name)
                with original.extractfile(info) as stream:
                    source = HashReader(stream)
                    clean = tarfile.TarInfo(name)
                    clean.size, clean.mode, clean.mtime = info.size, 0o644, 0
                    result.addfile(clean, source)
                    expected = Path(name).name.split('.')[0]
                    if len(expected) == 64 and source.digest.hexdigest() != expected:
                        raise ValueError('Copied compressed layer digest mismatch: ' + name)
            clean = tarfile.TarInfo(new_layer_name)
            clean.size, clean.mode, clean.mtime = compressed_size, 0o644, 0
            with layer.open('rb') as stream:
                result.addfile(clean, stream)
            result.addfile(*member('manifest.json', json.dumps([manifest]).encode()))
    verified, total = [], 0
    with tarfile.open(args.output, 'r:') as archive:
        for name, expected in zip(manifest['Layers'], config['rootfs']['diff_ids']):
            print('VERIFY', name, flush=True)
            with archive.extractfile(name) as stream:
                compressed_stream = HashReader(stream)
                with gzip.GzipFile(fileobj=compressed_stream) as expanded:
                    actual, size = sha(expanded)
                if 'sha256:' + actual != expected:
                    raise ValueError('Uncompressed layer mismatch: ' + name)
                expected_compressed = Path(name).name.split('.')[0]
                if len(expected_compressed) == 64 and compressed_stream.digest.hexdigest() != expected_compressed:
                    raise ValueError('Final compressed layer mismatch: ' + name)
                total += size
                verified.append({'file': name, 'compressed_sha256': compressed_stream.digest.hexdigest(),
                                 'diff_id': expected, 'uncompressed_bytes': size})
    if total > 60 * 2**30:
        raise ValueError('Image exceeds size limit')
    report = {'kind': 'verified_code_overlay', 'archive': str(args.output.resolve()),
              'archive_bytes': args.output.stat().st_size, 'config_digest': 'sha256:' + config_name[:-5],
              'parent_config_digest': 'sha256:' + old_config_digest,
              'mandatory_base_preserved': config['rootfs']['diff_ids'][:len(original_ids)] == original_ids,
              'base_validation_report': base_report, 'uncompressed_bytes': total, 'uncompressed_gib': total/2**30,
              'added_uncompressed_bytes': added_size, 'layers': verified, 'app_sha256': code_hashes,
              'container_gpu_launch_verified': False}
    args.output.with_suffix('.build-report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('layers','base_validation_report','app_sha256')}, indent=2))


if __name__ == '__main__':
    main()

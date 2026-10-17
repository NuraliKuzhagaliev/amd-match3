"""Full Docker/AMD check with explicit GPU devices, offline mode and DAC policy.

Run on a Linux AMD GPU host after loading/pulling the final image. This is our
additional integration harness; it does not impersonate the official grader.
"""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import uuid

from evaluate import DeviceMemoryMonitor, normalize

BASE = 'rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0'


def run(args, timeout=60):
    proc = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    if proc.returncode:
        raise RuntimeError(' '.join(args[:4]) + ': ' + (proc.stderr or proc.stdout)[-4000:])
    return proc.stdout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('image')
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--questions', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    preflight = {'linux': sys.platform.startswith('linux'), 'amd_kfd': Path('/dev/kfd').exists(),
                 'dri': Path('/dev/dri').is_dir()}
    try:
        preflight['docker'] = bool(run(['docker', 'version', '--format', '{{.Server.Version}}']).strip())
    except Exception as error:
        preflight['docker'], preflight['docker_error'] = False, str(error)
    report = {'kind': 'full_container_gpu_integration', 'image': args.image,
              'preflight': preflight, 'status': 'blocked', 'container_gpu_launch_verified': False}
    if args.preflight_only or not all(preflight.get(k) for k in ('linux','amd_kfd','dri','docker')):
        report['reason'] = 'Requires Linux Docker host with compatible AMD /dev/kfd and /dev/dri devices'
        args.report.write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps(report, indent=2))
        return 2
    name = 'mc3-gpu-check-' + uuid.uuid4().hex[:10]
    monitor = DeviceMemoryMonitor()
    monitor.start()
    created = False
    try:
        image = json.loads(run(['docker', 'image', 'inspect', args.image]))[0]
        base = json.loads(run(['docker', 'image', 'inspect', BASE]))[0]
        prefix = base['RootFS']['Layers']
        if image['RootFS']['Layers'][:len(prefix)] != prefix:
            raise ValueError('Mandatory ROCm base layer identity mismatch')
        if image['Size'] > 60 * 2**30:
            raise ValueError('Image size exceeds 60 GiB')
        report.update(image_id=image['Id'], uncompressed_bytes=image['Size'], mandatory_base_preserved=True)
        command = ['docker','run','-d','--name',name,'--network','none','--cap-drop','DAC_OVERRIDE',
                   '--device','/dev/kfd','--device','/dev/dri','--ipc=host']
        gids = {Path('/dev/kfd').stat().st_gid}
        gids.update(p.stat().st_gid for p in Path('/dev/dri').glob('renderD*'))
        for gid in sorted(gids):
            command.extend(['--group-add', str(gid)])
        started = time.monotonic()
        created = True
        run(command + [args.image])
        run(['docker','cp', str(args.corpus.resolve()) + '/.', name + ':/app/corpus/'], timeout=120)
        run(['docker','exec',name,'python3','/app/app.py','--index','/app/corpus'], timeout=max(1, 600-(time.monotonic()-started)))
        report['startup_seconds'] = time.monotonic()-started
        queries = json.loads(args.questions.read_text(encoding='utf-8'))['queries']
        rows = []
        for number, question in enumerate(queries, 1):
            query_id = f'query_{number:03d}'
            before = time.monotonic()
            run(['docker','exec',name,'python3','/app/app.py','--corpus','/app/corpus',
                 '--query-id',query_id,'--query',question['query']], timeout=30)
            result = json.loads(run(['docker','exec',name,'cat',f'/app/output/{query_id}_output.json']))
            elapsed = time.monotonic()-before
            if not isinstance(result.get('answer'), str) or not isinstance(result.get('citations'), list):
                raise ValueError('Malformed answer JSON')
            aliases = [question['expected_answer'], *question.get('answer_aliases', [])]
            answer_ok = normalize(result['answer']) in {normalize(x) for x in aliases}
            citations_ok = set(result['citations']) == set(question['expected_citations'])
            rows.append({'id':query_id, 'query':question['query'], 'actual':result,
                         'answer_ok':answer_ok, 'citations_ok':citations_ok,
                         'passed':answer_ok and citations_ok and elapsed <= 30, 'elapsed_seconds':elapsed})
        health = json.loads(run(['docker','exec',name,'python3','-c',
                                 'from mc3.io import request; from mc3.config import Config; import json; print(json.dumps(request(Config().socket,{"op":"health"},10)))']))
        memory = monitor.finish()
        peaks = list(memory['peak_gib_by_device'].values())
        memory_pass = bool(peaks) and 1 <= max(peaks) <= 48*1.01
        still_running = run(['docker','inspect','--format','{{.State.Running}}',name]).strip() == 'true'
        batches_pass = all(sum(r['elapsed_seconds'] for r in rows[i:i+10]) <= 600 for i in range(0,len(rows),10))
        report.update(questions=rows, hardware=health, device_memory=memory, container_still_running=still_running,
                      passed_count=sum(r['passed'] for r in rows), total=len(rows), memory_pass=memory_pass,
                      container_gpu_launch_verified=True)
        passed = all(r['passed'] for r in rows) and report['startup_seconds'] <= 600 and memory_pass and still_running and batches_pass
        report['status'] = 'passed' if passed else 'failed'
    except Exception as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        if created:
            try:
                report['container_logs'] = run(['docker','logs',name])[-12000:]
            except Exception:
                pass
    finally:
        report.setdefault('device_memory', monitor.finish())
        if created:
            subprocess.run(['docker','rm','-f',name], capture_output=True)
        args.report.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('questions','device_memory')}, indent=2))
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    sys.exit(main())

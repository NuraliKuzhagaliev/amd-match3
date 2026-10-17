"""GPU integration runner: a NEW app.py process for EVERY question.

Expected answers are confined to this test runner; they are never shipped in /app.
Supports running against an already resident worker or starting a fresh one.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
from mc3.io import atomic_json, request


def normalize(text):
    return re.sub(r"[\s\-.·_]", "", text.upper())


class DeviceMemoryMonitor:
    """Whole-device AMD VRAM from the kernel, independent of torch's allocator."""
    def __init__(self):
        self.stop_event = threading.Event()
        self.samples, self.errors = [], []
        self.paths = []
        for path in Path('/sys/class/drm').glob('card*/device/mem_info_vram_used'):
            if path.parent.resolve() not in [p.parent.resolve() for p in self.paths]:
                self.paths.append(path)
        self.thread = threading.Thread(target=self.run, daemon=True)

    def run(self):
        while not self.stop_event.is_set():
            row = {}
            for path in self.paths:
                try:
                    row[str(path)] = int(path.read_text().strip())
                except (OSError, ValueError) as error:
                    message = str(error)
                    if message not in self.errors:
                        self.errors.append(message)
            if row:
                self.samples.append({'monotonic': time.monotonic(), 'used_bytes': row})
            self.stop_event.wait(1)

    def start(self):
        self.thread.start()

    def finish(self):
        self.stop_event.set()
        self.thread.join(timeout=2)
        peaks = {str(path): max((r['used_bytes'].get(str(path), 0) for r in self.samples), default=0) / 2**30
                 for path in self.paths}
        return {'method': 'AMD kernel mem_info_vram_used; sampled every 1 second',
                'available': bool(self.samples), 'peak_gib_by_device': peaks,
                'errors': self.errors, 'samples': self.samples}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=ROOT / "reports/gpu-evaluation.json")
    parser.add_argument("--model-dir", type=Path, default=Path("/models"))
    parser.add_argument("--reuse-worker", action="store_true")
    args = parser.parse_args()
    args.report = args.report.resolve()
    args.model_dir = args.model_dir.resolve()
    corpus = args.corpus.resolve()
    questions = json.loads(args.questions.read_text(encoding="utf-8"))["queries"]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    monitor = DeviceMemoryMonitor()
    monitor.start()
    env = dict(os.environ, PYTHONPATH=str(ROOT / "app"), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1")
    if not args.reuse_worker:
        run_dir = args.report.parent / ("run-" + str(int(time.time())))
        run_dir.mkdir()
        env.update(MC3_INDEX_DIR=str(run_dir / "index"), MC3_OUTPUT_DIR=str(run_dir / "output"),
                   MC3_SOCKET=str(run_dir / "worker.sock"),
                   MC3_MODEL_DIR=str(args.model_dir / "vision"), MC3_EMBED_DIR=str(args.model_dir / "embedding"))
        logfile = (run_dir / "worker.log").open("w", encoding="utf-8")
        worker = subprocess.Popen([sys.executable, "-m", "mc3.worker"], cwd=ROOT / "app", env=env,
                                  stdout=logfile, stderr=subprocess.STDOUT)
    else:
        worker, logfile = None, None
    address = env.get("MC3_SOCKET", "/tmp/mc3.sock")
    output = Path(env.get("MC3_OUTPUT_DIR", "/app/output"))
    started = time.monotonic()
    rows = []
    try:
        if worker:
            ready_deadline = started + 570
            while time.monotonic() < ready_deadline:
                if worker.poll() is not None:
                    raise RuntimeError(f"GPU worker exited with code {worker.returncode}; inspect {run_dir / 'worker.log'}")
                try:
                    request(address, {"op": "health"}, 1)
                    break
                except (OSError, TimeoutError):
                    time.sleep(0.2)
            else:
                raise TimeoutError("GPU worker startup exceeded the allowed budget")
        index = subprocess.run([sys.executable, str(ROOT / "app/app.py"), "--index", str(corpus)],
                               env=env, capture_output=True, text=True, timeout=600)
        startup = time.monotonic() - started
        if index.returncode:
            raise RuntimeError(index.stderr + index.stdout)
        print(f"Startup including index: {startup:.2f}s", flush=True)
        for number, question in enumerate(questions, start=1):
            query_id = f"query_{number:02d}"
            before = time.monotonic()
            failure = None
            try:
                proc = subprocess.run([sys.executable, str(ROOT / "app/app.py"), "--corpus", str(corpus),
                                       "--query-id", query_id, "--query", question["query"]],
                                      env=env, capture_output=True, text=True, timeout=30)
                if proc.returncode:
                    raise RuntimeError(proc.stderr[-2000:])
                value = json.loads((output / f"{query_id}_output.json").read_text(encoding="utf-8"))
                if not isinstance(value["answer"], str) or not isinstance(value["citations"], list):
                    raise ValueError("Malformed output")
            except Exception as error:
                failure, value = str(error), {"answer": None, "citations": []}
            elapsed = time.monotonic() - before
            aliases = [question["expected_answer"], *question.get("answer_aliases", [])]
            answer_ok = isinstance(value["answer"], str) and normalize(value["answer"]) in {normalize(x) for x in aliases}
            citations_ok = set(value["citations"]) == set(question["expected_citations"])
            passed = answer_ok and citations_ok and not failure and elapsed <= 30
            row = {"id": query_id, "query": question["query"], "actual": value,
                   "family": question.get("family", "sample"),
                   "expected_answer": question["expected_answer"], "expected_citations": question["expected_citations"],
                   "answer_ok": answer_ok, "citations_ok": citations_ok, "passed": passed,
                   "elapsed_seconds": elapsed, "failure": failure}
            rows.append(row)
            print(f"{query_id}: {'PASS' if passed else 'FAIL'} {elapsed:.2f}s {value}", flush=True)
        report = {"kind": "real_gpu_integration", "questions": rows, "startup_seconds": startup,
                  "score": 20 * sum(row["passed"] for row in rows), "max_score": len(rows) * 20,
                  "total_questions_seconds": sum(row["elapsed_seconds"] for row in rows),
                  "hardware": request(address, {"op": "health"}, 10)}
        report['device_memory'] = monitor.finish()
        report['app_sha256'] = {p.relative_to(ROOT / 'app').as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in sorted((ROOT / 'app').rglob('*.py'))}
        report['question_batches'] = [{'start': i+1, 'count': len(rows[i:i+10]),
                                      'seconds': sum(r['elapsed_seconds'] for r in rows[i:i+10])}
                                     for i in range(0, len(rows), 10)]
        report['limits'] = {'startup_pass': startup <= 600,
                            'per_question_pass': all(r['elapsed_seconds'] <= 30 for r in rows),
                            'all_ten_question_batches_pass': all(b['seconds'] <= 600 for b in report['question_batches'])}
        atomic_json(args.report, report)
        print(f"Score: {report['score']}/{report['max_score']}; report: {args.report}", flush=True)
        return 0 if all(row["passed"] for row in rows) and startup <= 600 else 1
    finally:
        monitor.finish()
        if worker:
            worker.terminate()
            try:
                worker.wait(timeout=15)
            except subprocess.TimeoutExpired:
                worker.kill()
                worker.wait()
            logfile.close()


if __name__ == "__main__":
    sys.exit(main())

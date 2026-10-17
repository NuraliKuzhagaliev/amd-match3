"""Infrastructure tests use a deliberately fake backend; not model benchmarks."""
from pathlib import Path
import json
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
from mc3.config import Config
from mc3.pipeline import Pipeline, walk_files


class FakeBackend:
    def embed(self, texts, query=False):
        return np.ones((len(texts), 4), dtype=np.float32) * 0.5

    def count_tokens(self, text):
        return len(text) // 4


class PipelineTests(unittest.TestCase):
    def test_audit_trim_preserves_topic_bridge_without_query_identifier(self):
        class BridgeBackend(FakeBackend):
            labels = None

            def generate(self, prompt, **kwargs):
                if self.labels is None:
                    records = [json.loads(line) for line in prompt.splitlines() if line.startswith('{"source":')]
                    self.labels = {r['file']: r['source'] for r in records}
                    return json.dumps({'answer':'4.3.2','sources':list(self.labels.values())})
                return json.dumps({'answer':'4.3.2','sources':[self.labels['tickets.csv']], 'supported':True})

        (self.corpus/'events.log').write_text('Thermal throttle incident tracked as INC-56789')
        (self.corpus/'tickets.csv').write_text('ticket,fixed_in\nINC-56789,4.3.2\n')
        pipeline = Pipeline(BridgeBackend(), self.cfg)
        pipeline.build(self.corpus)
        result = pipeline.answer(self.corpus, 'Which firmware version fixed the thermal throttle incident?')
        self.assertEqual(result['citations'], ['events.log','tickets.csv'])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.corpus = self.root / "corpus"
        self.corpus.mkdir()
        self.cfg = Config(index=self.root / "index", parse_seconds=12)

    def tearDown(self):
        self.temp.cleanup()

    def test_empty_corpus_and_persisted_reload(self):
        pipeline = Pipeline(FakeBackend(), self.cfg)
        self.assertEqual(pipeline.build(self.corpus)["chunks"], 0)
        self.assertEqual(pipeline.answer(self.corpus, "Unknown value?"),
                         {"answer": "", "citations": [], "confidence": 0.0})
        other = Pipeline(FakeBackend(), self.cfg)
        other.load(self.corpus)
        self.assertEqual(len(other.search.chunks), 0)

    def test_bad_files_do_not_drop_good_files(self):
        (self.corpus / "a-broken.pdf").write_bytes(b"not a PDF")
        (self.corpus / "b-unknown.dat").write_bytes(b"unknown")
        (self.corpus / "c-valid.txt").write_text("NX-71 timeout is 287 seconds", encoding="utf-8")
        (self.corpus / "empty").mkdir()
        pipeline = Pipeline(FakeBackend(), self.cfg)
        report = pipeline.build(self.corpus)
        self.assertEqual(report["files"], 1)
        self.assertEqual(report["skipped"], 2)
        self.assertIn("287", pipeline.search.chunks[0].text)

    def test_rebuild_replaces_old_content(self):
        source = self.corpus / "source.txt"
        source.write_text("Current temperature is 87")
        pipeline = Pipeline(FakeBackend(), self.cfg)
        pipeline.build(self.corpus)
        source.write_text("Current temperature is 92")
        pipeline.build(self.corpus)
        self.assertIn("92", pipeline.search.chunks[0].text)
        self.assertNotIn("87", pipeline.search.chunks[0].text)

    def test_wrong_corpus_is_rejected(self):
        pipeline = Pipeline(FakeBackend(), self.cfg)
        pipeline.build(self.corpus)
        other = self.root / "other"
        other.mkdir()
        with self.assertRaises(ValueError):
            pipeline.answer(other, "Any value?")

    def test_unknown_source_is_repaired_against_original_text(self):
        class RepairBackend(FakeBackend):
            calls = 0

            def generate(self, prompt, **kwargs):
                self.calls += 1
                label = "MISSING" if self.calls == 1 else "C01"
                return '{"answer":"287","sources":["' + label + '"]}'

        (self.corpus / "source.txt").write_text("NX-71 timeout: 287")
        backend = RepairBackend()
        pipeline = Pipeline(backend, self.cfg)
        pipeline.build(self.corpus)
        result = pipeline.answer(self.corpus, "NX-71 timeout?")
        self.assertEqual(result["answer"], "287")
        self.assertEqual(result["citations"], ["source.txt"])
        self.assertEqual(backend.calls, 3)  # repair plus independent semantic audit

    def test_semantic_audit_can_refuse_wrong_attribute(self):
        class AuditBackend(FakeBackend):
            calls = 0

            def generate(self, prompt, **kwargs):
                self.calls += 1
                return ('{"answer":"94","sources":["C01"]}' if self.calls == 1
                        else '{"answer":"","sources":[],"supported":false}')

        (self.corpus / 'source.txt').write_text('NX-71 maximum junction temperature: 94')
        backend = AuditBackend()
        pipeline = Pipeline(backend, self.cfg)
        pipeline.build(self.corpus)
        self.assertEqual(pipeline.answer(self.corpus, 'NX-71 storage temperature?')['answer'], '')
        self.assertEqual(backend.calls, 2)

    def test_symlink_to_outside_corpus_is_skipped(self):
        outside = self.root / "outside.txt"
        outside.write_text("do not index")
        try:
            (self.corpus / "link.txt").symlink_to(outside)
        except OSError:
            self.skipTest("Symlink creation unavailable")
        self.assertEqual(list(walk_files(self.corpus, [])), [])


if __name__ == "__main__":
    unittest.main()

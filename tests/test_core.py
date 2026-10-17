from dataclasses import replace
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import threading
import unittest

APP = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP))

from mc3.evidence import parse_json, validate, validate_selection, value_only, value_in_text
from mc3.io import atomic_json, receive, send
from mc3.parse_one import decode, parse
from mc3.retrieval import Chunk, SearchIndex, chunk_blocks, identifiers, tokens


class CoreTests(unittest.TestCase):
    def test_scalar_units_follow_requested_attribute(self):
        for answer, query, expected in (
            ('32 V', 'What is the supply limit?', '32'),
            ('119 C', 'What is the junction limit?', '119'),
            ('72 W', 'What is the approved power limit?', '72'),
            ('-21 C', 'What is the minimum temperature?', '-21'),
            ('12.75 m', 'What is the maximum cable length in metres?', '12.75'),
            ('6 weeks', 'What is the lead time in weeks?', '6'),
            ('REV-C2', 'What is the board revision?', 'REV-C2'),
            ('Q3 FY27', 'What is the sampling quarter?', 'Q3 FY27'),
            ('32V', 'What is the part number?', '32V'),
            ('32 V', 'What is the maximum power?', '32 V'),
        ):
            with self.subTest(answer=answer, query=query):
                self.assertEqual(value_only(answer, query), expected)

    def test_normalized_units_still_require_literal_source_grounding(self):
        sources = {'C01': Chunk('a', 'limits.txt', 'text', 'AB-219 supply limit 32 V')}
        value = {'answer': '32 V', 'sources': ['C01']}
        result, _ = validate_selection(value, sources, 'What supply limit applies to AB-219?')
        self.assertEqual(result['answer'], '32')
        result, _ = validate_selection({'answer': '33 V', 'sources': ['C01']}, sources,
                                       'What supply limit applies to AB-219?')
        self.assertEqual(result['answer'], '')

    def test_ticket_to_release_bridge_cannot_be_removed(self):
        sources = {'C01': Chunk('a','tickets.csv','row','INC-56789 resolved by REL-99887'),
                   'C02': Chunk('b','release.txt','text','REL-99887 firmware version 18.2.83')}
        query = 'Which firmware version resolves ticket INC-56789?'
        result, _ = validate_selection({'answer':'18.2.83','sources':['C01','C02']}, sources, query)
        self.assertEqual(result['citations'], ['release.txt','tickets.csv'])
        result, trace = validate_selection({'answer':'18.2.83','sources':['C02']}, sources, query)
        self.assertEqual(result['answer'], '')
        self.assertEqual(trace['reason'], 'question_identifier_not_in_sources')

    def test_missing_middle_file_breaks_three_file_chain(self):
        sources = {'C01': Chunk('a','log.txt','text','VX-771A incident linked to INC-56789'),
                   'C02': Chunk('b','tickets.csv','row','INC-56789 resolved by REL-99887'),
                   'C03': Chunk('c','release.txt','text','REL-99887 firmware version 18.2.83')}
        query = 'Which firmware fixed the incident on VX-771A?'
        result, _ = validate_selection({'answer':'18.2.83','sources':list(sources)}, sources, query)
        self.assertEqual(len(result['citations']), 3)
        result, trace = validate_selection({'answer':'18.2.83','sources':['C01','C03']}, sources, query)
        self.assertEqual(result['answer'], '')
        self.assertEqual(trace['reason'], 'disconnected_identifier_chain')

    def test_missing_product_suffix_and_signal_are_not_substituted(self):
        index = SearchIndex([Chunk('a', 'map.txt', 'text', 'VX-771A THERM_ALERT# on pin D18')])
        self.assertEqual(index.missing_identifiers('VX-771A THERM_ALERT# pin?'), set())
        self.assertIn('VX771APRO', index.missing_identifiers('VX-771A-PRO sampling?'))
        self.assertIn('OVERCURRENTALERT', index.missing_identifiers('OVERCURRENT_ALERT# on VX-771A?'))
        self.assertEqual(index.missing_identifiers('field-replaceable fan?'), set())

    def test_three_file_chain_survives_many_related_distractors(self):
        chunks = [Chunk('a','events.log','text','VX-771A thermal throttle incident tracked as INC-88111'),
                  Chunk('b','tickets.csv','row','INC-88111 resolved by REL-99222'),
                  Chunk('c','release.txt','text','REL-99222 firmware version 18.2.83')]
        chunks += [Chunk(str(i), f'noise/{i}.txt','text',f'VX-{100+i}B thermal throttle firmware incident THERM_ALERT#') for i in range(80)]
        index = SearchIndex(chunks)
        found = index.retrieve('Which firmware fixed the thermal throttle incident on VX-771A?')
        self.assertTrue({'events.log','tickets.csv','release.txt'} <= {c.path for c in found})

    def test_grounding_respects_complete_numeric_and_identifier_values(self):
        self.assertFalse(value_in_text('94', 'Junction temperature: 194 C'))
        self.assertFalse(value_in_text('94', 'Junction temperature: -94 C'))
        self.assertFalse(value_in_text('REV-C2', 'Board: REV-C21'))
        self.assertTrue(value_in_text('94', 'Junction temperature: 94 C'))
        self.assertTrue(value_in_text('7.2.1', 'Fixed in 7.2.1'))
        self.assertTrue(value_in_text('Q3 FY27', 'Sampling Q3\nFY27'))
        self.assertTrue(value_in_text('REV-C2', 'Board: REV C2'))

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_identifier_variants(self):
        self.assertIn("tq40", tokens("TQ-40"))
        self.assertIn("ORR1847", identifiers("Ticket ORR-1847 raised E7731"))

    def test_chunk_boundaries_and_overlap(self):
        text = "\n".join(f"Line {i}: " + "z" * 40 for i in range(100))
        chunks = chunk_blocks("x.txt", [{"text": text, "locator": "body"}], 240, 40)
        self.assertGreater(len(chunks), 1)
        for i in range(100):
            self.assertTrue(any(f"Line {i}:" in c.text for c in chunks), i)
        self.assertTrue(all(c.path == "x.txt" for c in chunks))

    def test_very_long_single_line_is_not_lost(self):
        chunks = chunk_blocks("x.txt", [{"text": "A" * 9000 + "TARGET-927", "locator": "body"}], 240, 40)
        self.assertTrue(any("TARGET-927" in c.text for c in chunks))
        self.assertTrue(all(len(c.text) <= 280 for c in chunks))

    def test_identifier_bridge_retrieves_non_matching_row(self):
        chunks = [Chunk("a", "log.txt", "line", "Thermal throttle incident E7731 tracked as ORR-1847"),
                  Chunk("b", "tickets.csv", "row", "ORR-1847 | handler retry race | fixed_in: 4.3.2"),
                  Chunk("c", "notes.txt", "text", "Thermal throttle is a supported safety feature")]
        found = SearchIndex(chunks).retrieve("Which firmware fixes the thermal throttle incident?")
        self.assertIn("tickets.csv", [c.path for c in found])

    def test_empty_search(self):
        self.assertEqual(SearchIndex([]).retrieve("unknown"), [])

    def test_withdrawn_ranks_below_current(self):
        chunks = [Chunk("a", "r1_WITHDRAWN.pdf", "page", "TQ-40 maximum junction 105"),
                  Chunk("b", "r2.pdf", "page", "TQ-40 maximum junction 94")]
        self.assertEqual(SearchIndex(chunks).retrieve("TQ-40 maximum junction")[0].path, "r2.pdf")

    def test_valid_multisource_evidence(self):
        sources = {"C01": Chunk("a", "log.txt", "line", "incident maps to ORR-1847"),
                   "C02": Chunk("b", "bugs.csv", "row", "ORR-1847 fixed_in: 4.3.2")}
        value = {"answer": "4.3.2", "evidence": [{"source": "C01", "quote": "maps to ORR-1847"},
                 {"source": "C02", "quote": "ORR-1847 fixed_in: 4.3.2"}]}
        result, _ = validate(value, sources)
        self.assertEqual(result["citations"], ["bugs.csv", "log.txt"])
        self.assertEqual(result["answer"], "4.3.2")

    def test_unknown_source_is_rejected(self):
        result, _ = validate({"answer": "94", "evidence": [{"source": "FAKE", "quote": "94"}]}, {})
        self.assertEqual(result["answer"], "")

    def test_fabricated_quote_is_rejected(self):
        result, _ = validate({"answer": "94", "evidence": [{"source": "C01", "quote": "maximum 94"}]},
                             {"C01": Chunk("a", "x.txt", "text", "maximum 105")})
        self.assertEqual(result["citations"], [])

    def test_invented_value_is_rejected(self):
        result, _ = validate({"answer": "105", "evidence": [{"source": "C01", "quote": "maximum 94"}]},
                             {"C01": Chunk("a", "x.txt", "text", "maximum 94")})
        self.assertEqual(result["answer"], "")

    def test_refusal_is_explicit(self):
        result, _ = validate({"answer": "", "evidence": []}, {})
        self.assertEqual(result, {"answer": "", "citations": [], "confidence": 0.0})

    def test_source_selection_generates_original_evidence(self):
        sources = {"C01": Chunk("a", "bugs.csv", "row", "ABC-902 fixed_in: 7.2.1")}
        result, trace = validate_selection({"answer": "7.2.1", "sources": ["C01"]}, sources, "Which firmware?")
        self.assertEqual(result["citations"], ["bugs.csv"])
        self.assertEqual(trace["evidence"][0]["original_text"], sources["C01"].text)

    def test_source_selection_rejects_invented_values(self):
        result, _ = validate_selection({"answer": "8.9.0", "sources": ["C01"]},
                                       {"C01": Chunk("a", "bugs.csv", "row", "fixed_in: 7.2.1")}, "Which firmware?")
        self.assertEqual(result["answer"], "")

    def test_numeric_units_removed_without_dropping_revision_qualifiers(self):
        self.assertEqual(value_only("90 C", "maximum junction temperature?"), "90")
        self.assertEqual(value_only("275 seconds", "batch timeout in seconds?"), "275")
        self.assertEqual(value_only("REV-C2", "board revision?"), "REV-C2")
        self.assertEqual(value_only("Q3 FY29", "customer sampling quarter?"), "Q3 FY29")

    def test_qualifiers_survive_validation(self):
        for value in ("Q3 FY27", "REV-C2", "ORR-FAN-2214-B"):
            result, _ = validate({"answer": value, "evidence": [{"source": "C01", "quote": value}]},
                                 {"C01": Chunk("a", "x.txt", "text", value)})
            self.assertEqual(result["answer"], value)

    def test_confidence_nonfinite_is_sanitized(self):
        result, _ = validate({"answer": "94", "confidence": float("nan"),
                              "evidence": [{"source": "C01", "quote": "94"}]},
                             {"C01": Chunk("a", "x.txt", "text", "94")})
        self.assertEqual(result["confidence"], 0.8)

    def test_fenced_json_and_truncated_output(self):
        self.assertEqual(parse_json('```json\n{"answer":"94"}\n```')["answer"], "94")
        with self.assertRaises(ValueError):
            parse_json('{"answer": "94", "evidence": [')

    def test_atomic_output_is_utf8(self):
        path = self.root / "out.json"
        atomic_json(path, {"answer": "京A·12345"})
        self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["answer"], "京A·12345")
        self.assertEqual(len(list(self.root.iterdir())), 1)

    @unittest.skipUnless(os.name == "posix", "Socket integration is exercised on Linux")
    def test_socket_framing(self):
        left, right = socket.socketpair()
        try:
            payload = {"query": "temperature?", "unicode": "温度"}
            thread = threading.Thread(target=send, args=(left, payload))
            thread.start()
            self.assertEqual(receive(right), payload)
            thread.join()
        finally:
            left.close()
            right.close()

    def test_unknown_type_is_skipped(self):
        path = self.root / "x.bin"
        path.write_bytes(b"bad")
        self.assertEqual(parse(path, self.root / "rendered")["skip"], "unknown type")

    @unittest.skipUnless(os.name == "posix", "Linux permissions test")
    def test_unreadable_file_raises(self):
        path = self.root / "unreadable.txt"
        path.write_text("private")
        path.chmod(0)
        try:
            try:
                path.read_bytes()
            except PermissionError:
                pass
            else:
                self.skipTest("Process can bypass DAC permissions; run with --cap-drop DAC_OVERRIDE")
            with self.assertRaises(PermissionError):
                parse(path, self.root)
        finally:
            path.chmod(0o600)

    def test_python_is_not_executed(self):
        path = self.root / "source.py"
        path.write_text("raise RuntimeError('must not execute')\nBATCH_TIMEOUT = 287\n", encoding="utf-8")
        self.assertIn("BATCH_TIMEOUT = 287", parse(path, self.root)["blocks"][0]["text"])

    def test_utf16_and_utf8_bom(self):
        self.assertEqual(decode("温度 94".encode("utf-16")), "温度 94")
        self.assertEqual(decode(b"\xef\xbb\xbfvalue"), "value")

    def test_semicolon_csv_preserves_column_relationships(self):
        path = self.root / "bugs.csv"
        path.write_text('ticket;description;fixed_in\nABC-999;"retry; race";7.2.1\n', encoding="utf-8")
        text = parse(path, self.root)["blocks"][0]["text"]
        self.assertIn("fixed_in: 7.2.1", text)
        self.assertIn("description: retry; race", text)

    def test_word_tables_are_read(self):
        from docx import Document
        doc = Document()
        table = doc.add_table(rows=2, cols=2)
        table.cell(0, 0).text, table.cell(0, 1).text = "Product", "Customer sampling"
        table.cell(1, 0).text, table.cell(1, 1).text = "NX-71", "Q4 FY29"
        path = self.root / "roadmap.docx"
        doc.save(path)
        text = "\n".join(b["text"] for b in parse(path, self.root)["blocks"])
        self.assertIn("Customer sampling", text)
        self.assertIn("Q4 FY29", text)

    def test_all_excel_sheets_are_read(self):
        from openpyxl import Workbook
        book = Workbook()
        book.active.append(["Name", "Part"])
        book.active.append(["wrong fan", "OLD-001"])
        sheet = book.create_sheet("Current parts")
        sheet.append(["Name", "Part"])
        sheet.append(["NX fan", "NX-FAN-927"])
        path = self.root / "parts.xlsx"
        book.save(path)
        blocks = parse(path, self.root)["blocks"]
        self.assertTrue(any("Part: NX-FAN-927" in b["text"] for b in blocks))

    def test_encrypted_pdf_is_not_opened(self):
        from pypdf import PdfWriter
        writer = PdfWriter()
        writer.add_blank_page(width=100, height=100)
        writer.encrypt("not-given")
        path = self.root / "secret.pdf"
        writer.write(path)
        result = parse(path, self.root)
        self.assertEqual(result["skip"], "encrypted PDF")
        self.assertEqual(result["blocks"], [])


if __name__ == "__main__":
    unittest.main()

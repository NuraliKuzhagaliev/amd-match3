from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
from mc3.parse_one import parse


class ExtendedParsingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_excel_merged_banner_does_not_replace_column_headers(self):
        from openpyxl import Workbook
        book = Workbook()
        sheet = book.active
        sheet.title = 'Approved'
        sheet.merge_cells('A1:C1')
        sheet['A1'] = 'Service catalog'
        sheet.append([])
        sheet.append(['Product', 'Assembly', 'Part'])
        sheet.append(['NX-81', 'Fan', 'FAN-9876-K'])
        path = self.root / 'parts.xlsx'
        book.save(path)
        blocks = parse(path, self.root)['blocks']
        target = next(b['text'] for b in blocks if 'FAN-9876-K' in b['text'])
        self.assertIn('Part: FAN-9876-K', target)
        self.assertIn('Product: NX-81', target)
        self.assertIn('Sheet: Approved', target)
        self.assertIn('Service catalog', target)

    def test_word_merged_title_and_column_labels_remain_together(self):
        from docx import Document
        doc = Document()
        table = doc.add_table(rows=3, cols=3)
        table.cell(0, 0).merge(table.cell(0, 2)).text = 'Approved roadmap'
        for c, v in zip(table.rows[1].cells, ['Product', 'Milestone', 'Quarter']):
            c.text = v
        for c, v in zip(table.rows[2].cells, ['NX-81', 'Customer sampling', 'Q4 FY31']):
            c.text = v
        path = self.root / 'roadmap.docx'
        doc.save(path)
        block = next(b['text'] for b in parse(path, self.root)['blocks'] if 'Q4 FY31' in b['text'])
        self.assertIn('Quarter: Q4 FY31', block)
        self.assertIn('Product: NX-81', block)
        self.assertIn('Approved roadmap', block)

    def test_mixed_pdf_keeps_text_and_schedules_image_transcription(self):
        import zlib
        from pypdf import PdfWriter
        from pypdf.generic import DictionaryObject, NameObject, NumberObject, DecodedStreamObject, EncodedStreamObject
        writer = PdfWriter()
        page = writer.add_blank_page(width=612, height=792)
        font = DictionaryObject({NameObject('/Type'): NameObject('/Font'), NameObject('/Subtype'): NameObject('/Type1'),
                                 NameObject('/BaseFont'): NameObject('/Helvetica')})
        image = EncodedStreamObject()
        image._data = zlib.compress(b'\xff\xff\xff' * 800 * 400)
        image.update({NameObject('/Type'): NameObject('/XObject'), NameObject('/Subtype'): NameObject('/Image'),
                      NameObject('/Width'): NumberObject(800), NameObject('/Height'): NumberObject(400),
                      NameObject('/ColorSpace'): NameObject('/DeviceRGB'), NameObject('/BitsPerComponent'): NumberObject(8),
                      NameObject('/Filter'): NameObject('/FlateDecode')})
        page[NameObject('/Resources')] = DictionaryObject({
            NameObject('/Font'): DictionaryObject({NameObject('/F1'): writer._add_object(font)}),
            NameObject('/XObject'): DictionaryObject({NameObject('/Im1'): writer._add_object(image)})})
        stream = DecodedStreamObject()
        stream.set_data(b'BT /F1 12 Tf 20 750 Td (Inspection document for NX-81. This text header is longer than sixty characters.) Tj ET\nq 500 0 0 250 20 300 cm /Im1 Do Q')
        page[NameObject('/Contents')] = writer._add_object(stream)
        path = self.root / 'mixed.pdf'
        writer.write(path)
        parsed = parse(path, self.root / 'rendered')
        self.assertIn('NX-81', parsed['blocks'][0]['text'])
        self.assertEqual(len(parsed['images']), 1)
        self.assertTrue(Path(parsed['images'][0]['path']).is_file())


if __name__ == '__main__':
    unittest.main()

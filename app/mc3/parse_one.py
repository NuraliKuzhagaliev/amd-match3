"""Isolated file parser, invoked in a subprocess with a per-file timeout.

Documents are data: source files are read as text, never imported or executed.
Only password-free PDFs are read. External document relationships are not fetched.
"""
from __future__ import annotations

import csv
from datetime import date, datetime
import io
import json
from pathlib import Path
import sys
import zipfile

SUPPORTED = {".pdf", ".docx", ".xlsx", ".csv", ".txt", ".log", ".py", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
MAX_BYTES = 128 * 1024 * 1024
MAX_EXPANDED = 256 * 1024 * 1024


def decode(raw: bytes) -> str:
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16")
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            pass
    return raw.decode("utf-8", errors="replace")


def check_zip(path: Path) -> None:
    with zipfile.ZipFile(path) as archive:
        if sum(x.file_size for x in archive.infolist()) > MAX_EXPANDED:
            raise ValueError("Office archive expands beyond the per-file budget")
        if any(x.flag_bits & 1 for x in archive.infolist()):
            raise ValueError("Encrypted Office archive")


def parse(path: Path, scratch: Path) -> dict:
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED:
        return {"blocks": [], "images": [], "skip": "unknown type"}
    if path.stat().st_size > MAX_BYTES:
        raise ValueError("File exceeds the per-file byte budget")
    # Actually open the file; access failures remain visible even when running as root
    # with DAC_OVERRIDE removed by the evaluator.
    with path.open("rb") as stream:
        stream.read(1)
    blocks, images = [], []

    def add(text: str, where: str) -> None:
        text = text.replace("\x00", "").strip()
        if text:
            blocks.append({"text": text, "locator": where})

    if suffix in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
        images.append({"path": str(path), "locator": "image"})
    elif suffix == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(path, strict=False)
        if reader.is_encrypted:
            return {"blocks": [], "images": [], "skip": "encrypted PDF"}
        rendered = None
        for i, page in enumerate(reader.pages):
            text = page.extract_text(extraction_mode="layout") or ""
            add(text, f"page {i + 1}")
            resources = page.get('/Resources', {})
            if hasattr(resources, 'get_object'):
                resources = resources.get_object()
            objects = resources.get('/XObject', {})
            if hasattr(objects, 'get_object'):
                objects = objects.get_object()
            large_image = any(obj.get_object().get('/Subtype') == '/Image'
                              and int(obj.get_object().get('/Width', 0)) * int(obj.get_object().get('/Height', 0)) >= 180000
                              for obj in objects.values())
            if len(text.strip()) < 60 or (large_image and len(text.strip()) < 2000):
                import pypdfium2 as pdfium
                if rendered is None:
                    rendered = pdfium.PdfDocument(str(path))
                scratch.mkdir(parents=True, exist_ok=True)
                target = scratch / f"page-{i + 1}.png"
                bitmap = rendered[i].render(scale=1.6)
                image = bitmap.to_pil()
                image.thumbnail((2000, 2000))
                image.save(target)
                images.append({"path": str(target), "locator": f"page {i + 1} (scan)"})
                image.close()
                bitmap.close()
        if rendered is not None:
            rendered.close()
    elif suffix == ".docx":
        check_zip(path)
        from docx import Document
        from docx.table import Table
        doc = Document(path)
        for section in doc.sections:
            for part in (section.header, section.footer):
                for paragraph in part.paragraphs:
                    add(paragraph.text, "header/footer")
        for n, element in enumerate(doc.iter_inner_content()):
            if isinstance(element, Table):
                rows = [[c.text.strip() for c in row.cells] for row in element.rows]
                if not rows:
                    continue
                # Retain merged title rows, then use the first distinct multi-cell
                # row as column labels. Include both raw rows and column-value pairs.
                header_index = next((i for i, row in enumerate(rows[:8])
                                     if len({v for v in row if v}) >= 2), 0)
                header = rows[header_index]
                title = ' | '.join(dict.fromkeys(v for row in rows[:header_index] for v in row if v))
                for i, row in enumerate(rows):
                    pairs = ' | '.join(f'{header[j] if j < len(header) and header[j] else "column " + str(j+1)}: {v}'
                                       for j, v in enumerate(row) if v)
                    add((title + '\n' if title else '') + ' | '.join(header) + '\n' + ' | '.join(row) + '\n' + pairs,
                        f"table {n + 1}, row {i+1}")
                if len(rows) == 1:
                    add(' | '.join(header), f"table {n + 1}")
            else:
                add(element.text, f"paragraph {n + 1}")
    elif suffix == ".xlsx":
        check_zip(path)
        from openpyxl import load_workbook
        book = load_workbook(path, read_only=True, data_only=True)
        try:
            for sheet in book:
                # Read-only workbooks can carry incorrect producer dimensions.
                # Reset them so later sheets/rows are not silently truncated.
                sheet.reset_dimensions()
                prefix, header = [], None
                for number, row in enumerate(sheet.iter_rows(values_only=True), start=1):
                    values = [str(v.isoformat() if isinstance(v, (date, datetime)) else v)
                              if v is not None else "" for v in row]
                    if not any(values):
                        continue
                    if header is None and sum(bool(v) for v in values) < 2:
                        prefix.append(' | '.join(v for v in values if v))
                        add('Sheet: ' + sheet.title + '\n' + ' | '.join(values), f"sheet {sheet.title}, row {number}")
                        continue
                    if header is None:
                        header = values
                    pairs = [f"{header[i] if i < len(header) and header[i] else 'column ' + str(i+1)}: {v}"
                             for i, v in enumerate(values) if v]
                    add('Sheet: ' + sheet.title + '\n' + '\n'.join(prefix[-3:]) + '\n'
                        + ' | '.join(values) + '\n' + ' | '.join(pairs), f"sheet {sheet.title}, row {number}")
        finally:
            book.close()
    elif suffix == ".csv":
        text = decode(path.read_bytes())
        try:
            dialect = csv.Sniffer().sniff(text[:8192], delimiters=",;\t|")
        except csv.Error:
            dialect = csv.excel
        rows = csv.reader(io.StringIO(text), dialect)
        header = next(rows, [])
        for number, row in enumerate(rows, start=2):
            add(" | ".join(f"{header[i] if i < len(header) else 'column ' + str(i + 1)}: {v}"
                           for i, v in enumerate(row)), f"row {number}")
    else:
        add(decode(path.read_bytes()), "text")
    return {"blocks": blocks, "images": images}


if __name__ == "__main__":
    try:
        print(json.dumps(parse(Path(sys.argv[1]), Path(sys.argv[2])), ensure_ascii=False))
    except Exception as error:
        print(json.dumps({"error": f"{type(error).__name__}: {error}"}))
        sys.exit(1)

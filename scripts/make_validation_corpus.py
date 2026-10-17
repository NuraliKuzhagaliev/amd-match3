"""Generate an independent corpus with different facts and identifiers.

This is validation data, never part of the submission image. It is designed to
exercise generalization and citation chains rather than repeat sample answers.
"""
import argparse
import json
from pathlib import Path
import random

from docx import Document
from openpyxl import Workbook
from PIL import Image, ImageDraw, ImageFont
from pypdf import PdfWriter
from pypdf.generic import DictionaryObject, NameObject, DecodedStreamObject


def pdf(path: Path, lines: list[str], encrypted: bool = False):
    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    font = DictionaryObject({NameObject("/Type"): NameObject("/Font"),
                             NameObject("/Subtype"): NameObject("/Type1"),
                             NameObject("/BaseFont"): NameObject("/Helvetica")})
    page[NameObject("/Resources")] = DictionaryObject({NameObject("/Font"): DictionaryObject({NameObject("/F1"): writer._add_object(font)})})
    commands = ["BT /F1 13 Tf 40 740 Td"]
    for i, line in enumerate(lines):
        escaped = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        if i:
            commands.append("0 -24 Td")
        commands.append(f"({escaped}) Tj")
    commands.append("ET")
    stream = DecodedStreamObject()
    stream.set_data("\n".join(commands).encode("ascii"))
    page[NameObject("/Contents")] = writer._add_object(stream)
    if encrypted:
        writer.encrypt("validation-only-secret-not-provided-to-app")
    writer.write(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("destination", type=Path)
    parser.add_argument("--seed", type=int, default=3817)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    root = args.destination
    root.mkdir(parents=True, exist_ok=False)
    for name in ["specs", "planning", "support", "logs", "engineering", "vendor", "empty"]:
        (root / name).mkdir()
    product = f"NX-{rng.randrange(61, 90)}"
    temperature = str(rng.randrange(81, 99))
    quarter = f"Q{rng.randrange(1,5)} FY29"
    part = f"NX-FAN-{rng.randrange(3000,9000)}-D"
    ticket = f"NXB-{rng.randrange(6000,9900)}"
    firmware = f"7.{rng.randrange(1,8)}.{rng.randrange(1,9)}"
    error = f"F{rng.randrange(8000,9900)}"
    timeout = str(rng.randrange(220,420))
    pin = f"C{rng.randrange(11,24)}"
    revision = f"REV-D{rng.randrange(3,9)}"
    pdf(root / "specs/current.pdf", [f"Product {product} technical specification", "Revision 3 CURRENT",
                                     "This specification supersedes revision 1.", f"Maximum junction temperature: {temperature} C"])
    pdf(root / "specs/r1_WITHDRAWN.pdf", [f"Product {product} technical specification", "WITHDRAWN Revision 1",
                                          "Maximum junction temperature: 112 C"])
    doc = Document()
    doc.add_heading("FY29 Customer Roadmap", 0)
    table = doc.add_table(rows=2, cols=3)
    for cell, value in zip(table.rows[0].cells, ["Product", "Milestone", "Date"]):
        cell.text = value
    for cell, value in zip(table.rows[1].cells, [product, "Customer sampling", quarter]):
        cell.text = value
    doc.save(root / "planning/roadmap.docx")
    book = Workbook()
    book.active.title = "Legacy"
    book.active.append(["Product", "Assembly", "Part"])
    book.active.append(["NX-20", "Fan assembly", "NX-FAN-0001-A"])
    current = book.create_sheet("Current")
    current.append(["Product", "Assembly", "Part"])
    current.append([product, "Field-replaceable fan assembly", part])
    book.save(root / "support/parts.xlsx")
    (root / "support/bugs.csv").write_text(
        f"ticket,description,status,fixed_in\n{ticket},scheduler event bookkeeping race,closed,{firmware}\n"
        "NXB-1111,network startup fault,closed,5.1.0\n", encoding="utf-8")
    (root / "logs/production.log").write_text(
        f"2028-05-11T12:31:08Z ERROR {error} thermal throttle engaged on {product}\n"
        f"2028-05-11T12:31:09Z Incident linked to tracking ticket {ticket}\n", encoding="utf-8")
    (root / "engineering/ingest.py").write_text(
        f"# Old batch timeout was 75; do not use that legacy value.\nDEFAULT_BATCH_TIMEOUT_SECONDS = {timeout}\n", encoding="utf-8")
    (root / "engineering/release-notes.txt").write_text(
        "The firmware family supports thermal safety monitoring. See the bug tracker for fixes.", encoding="utf-8")
    font = ImageFont.load_default(size=30)
    image = Image.new("RGB", (1100, 500), "white")
    draw = ImageDraw.Draw(image)
    for y, text in enumerate([f"{product} BACKPLANE PIN MAP", "PIN       SIGNAL", "C02       GROUND",
                              f"{pin}       THERM_ALERT#", "C30       AUX_POWER"]):
        draw.text((40, 30 + y * 80), text, fill="black", font=font)
    image.save(root / "specs/pin-map.png")
    label = Image.new("RGB", (1000, 420), "#e7e5d8")
    draw = ImageDraw.Draw(label)
    for y, text in enumerate(["NEXUS ASSET LABEL", f"PRODUCT: {product}", f"BOARD: {revision}", "SERIAL: NX-581002"]):
        draw.text((40, 30 + y * 85), text, fill="black", font=font)
    label.save(root / "support/asset.jpg", quality=88)
    pdf(root / "vendor/locked.pdf", ["Unit price at 10000 units: 5331"], encrypted=True)
    (root / "vendor/unknown.dat").write_bytes(b"unrecognized-format")
    questions = [
        (f"What is the maximum junction temperature of the {product}?", temperature, ["specs/current.pdf"]),
        (f"In which quarter does {product} enter customer sampling?", quarter, ["planning/roadmap.docx"]),
        (f"What is the part number of the field-replaceable fan assembly for {product}?", part, ["support/parts.xlsx"]),
        (f"Which firmware version fixed ticket {ticket}?", firmware, ["support/bugs.csv"]),
        ("What error code is logged when the thermal throttle engages?", error, ["logs/production.log"]),
        ("What is the default batch timeout in seconds in the ingest service?", timeout, ["engineering/ingest.py"]),
        (f"Which backplane pin carries THERM_ALERT# on {product}?", pin, ["specs/pin-map.png"]),
        ("What board revision is printed on the asset label?", revision, ["support/asset.jpg"]),
        ("The production log shows a thermal throttle incident. Which firmware release fixed the underlying defect?", firmware,
         ["logs/production.log", "support/bugs.csv"]),
        (f"What is the unit price of {product} at 10,000 unit volume?", "", []),
    ]
    path = root.parent / (root.name + "-questions.json")
    path.write_text(json.dumps({"kind": "generated_validation", "seed": args.seed,
                               "queries": [{"query": q, "expected_answer": a, "expected_citations": c}
                                           for q, a, c in questions]}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Corpus: {root}; questions: {path}")


if __name__ == "__main__":
    main()

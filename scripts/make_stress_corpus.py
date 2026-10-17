"""Held-out stress fixtures. Never included in the runtime application layer.

Different document structures and distractors, not only randomized sample facts.
Generate locally; gold answers stay outside the corpus given to the application.
"""
import argparse
import json
from pathlib import Path
import random

from docx import Document
from openpyxl import Workbook
from PIL import Image, ImageDraw, ImageFont
from reportlab.pdfgen.canvas import Canvas
from reportlab.lib.utils import ImageReader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("destination", type=Path)
    parser.add_argument("--seed", type=int, default=90217)
    parser.add_argument('--variant', choices=('development', 'holdout'), default='development')
    args = parser.parse_args()
    rng = random.Random(args.seed)
    root = args.destination
    root.mkdir(parents=True, exist_ok=False)
    for name in ("specs", "planning", "parts", "incidents", "releases", "images", "src", "noise", "vendor", "empty"):
        (root / name).mkdir()
    questions = []

    def q(family, query, answer, *paths):
        questions.append({"family": family, "query": query, "expected_answer": answer,
                          "expected_citations": sorted(paths)})

    font = ImageFont.load_default(size=30)
    for i in range(8):
        product = f"VX-{rng.randrange(200, 950)}{chr(65+i)}"
        temperature, old = str(rng.randrange(70, 99)), str(rng.randrange(103, 120))
        quarter = f"Q{rng.randrange(1,5)} FY{31+i}"
        part = f"FAN-{rng.randrange(20000,90000)}-K{i}"
        ticket, release = f"INC-{rng.randrange(30000,90000)}", f"REL-{rng.randrange(30000,90000)}"
        firmware, timeout = f"{11+i}.{rng.randrange(1,9)}.{rng.randrange(10,90)}", str(rng.randrange(300,600))
        pin, revision = f"D{rng.randrange(11,29)}", f"REV-H{i+2}"
        current_path, previous_path = f"specs/document_{i:02d}_a.pdf", f"specs/document_{i:02d}_b.pdf"
        for path, value, status in ((current_path, temperature, "Revision 4 CURRENT. Supersedes revision 2."),
                                    (previous_path, old, "Revision 2 WITHDRAWN. Superseded by revision 4.")):
            canvas = Canvas(str(root / path))
            canvas.setFont("Helvetica", 14)
            canvas.drawString(35, 780, f"Technical specification: {product}")
            canvas.drawString(35, 750, status)
            if args.variant == 'holdout':
                canvas.drawString(35, 720, 'Scope: current production qualification records')
                canvas.showPage()
                canvas.drawString(35, 780, f'{product} {status}')
            canvas.drawString(35, 720, f"Maximum junction temperature: {value} C")
            canvas.showPage()
            canvas.drawString(35, 780, f"{product} nominal operating voltage: 24 V")
            canvas.save()
        roadmap_path = f"planning/plan_{i}.docx"
        doc = Document()
        doc.add_heading(f"{product} approved customer roadmap", 0)
        doc.add_paragraph("The following schedule is approved. Draft schedules are obsolete.")
        table = doc.add_table(rows=4, cols=3)
        table.cell(0, 0).merge(table.cell(0, 2)).text = f"Milestones for {product}"
        columns = ("Quarter", "Product", "Milestone") if args.variant == 'holdout' else ("Product", "Milestone", "Quarter")
        values = (quarter, product, "Customer sampling") if args.variant == 'holdout' else (product, "Customer sampling", quarter)
        other = ("Q1 FY40", product + '-L', "Customer sampling") if args.variant == 'holdout' else (product + '-L', "Customer sampling", "Q1 FY40")
        for cell, value in zip(table.rows[1].cells, columns):
            cell.text = value
        for cell, value in zip(table.rows[2].cells, values):
            cell.text = value
        for cell, value in zip(table.rows[3].cells, other):
            cell.text = value
        doc.save(root / roadmap_path)
        parts_path = f"parts/catalog_{i}.xlsx"
        book = Workbook()
        book.active.title = "Retired"
        book.active.append(["Product", "Assembly", "Part"])
        book.active.append([product, "Fan assembly", f"LEGACY-000{i}"])
        sheet = book.create_sheet("Approved service parts")
        sheet.merge_cells("A1:C1")
        sheet["A1"] = "Approved field-replaceable service catalog"
        sheet.append([])
        sheet.append(["Product", "Assembly", "Part"])
        sheet.append([product + "-L", "Fan assembly", f"OTHER-000{i}"])
        sheet.append([product, "Fan assembly", part])
        book.save(root / parts_path)
        if args.variant == 'holdout':
            (root / f'noise/draft_{i}.txt').write_text(
                f'{product} draft roadmap. NOT APPROVED. Obsolete planning estimate only.\n'
                'Customer sampling: Q1 FY42. Consult the approved customer roadmap for the valid date.\n', encoding='utf-8')
        log_path, bug_path, release_path = f"incidents/event_{i}.log", f"incidents/register_{i}.csv", f"releases/bulletin_{i}.txt"
        (root / log_path).write_text(f"Production incident on {product}: thermal throttle engaged.\n"
                                    f"Underlying defect is tracked by {ticket}.\n", encoding="utf-8")
        (root / bug_path).write_text("ticket;description;resolved_by_release\n"
                                    f"{ticket};scheduler accounting race;{release}\n"
                                    f"INC-000{i};unrelated fault;REL-000{i}\n", encoding="utf-8")
        (root / release_path).write_text(f"Release identifier {release}\nFirmware version: {firmware}\n"
                                        "This bulletin maps release identifiers to firmware versions.\n", encoding="utf-8")
        code_path = f"src/worker_{i}.py"
        (root / code_path).write_text(f'"""Ingest service for {product}."""\n'
                                     '# Previous timeout: 90 seconds.\n'
                                     f'DEFAULT_BATCH_TIMEOUT_SECONDS = {timeout}\n'
                                     'def ingest(timeout=DEFAULT_BATCH_TIMEOUT_SECONDS):\n    return timeout\n', encoding="utf-8")
        pin_path, label_path = f"images/map_{i}.png", f"images/label_{i}.jpg"
        image = Image.new("RGB", (1250, 700), "white")
        draw = ImageDraw.Draw(image)
        for row, line in enumerate((f"{product} backplane connector", "PIN        SIGNAL", "D02        GROUND",
                                    f"{pin}        THERM_ALERT#", "D30        AUX_POWER")):
            draw.text((45, 50+row*110), line, font=font, fill="black")
        if i % 3 == 1:
            image = image.rotate(90, expand=True)
        image.save(root / pin_path)
        image.close()
        image = Image.new("RGB", (2200, 550), "#eeeece")
        draw = ImageDraw.Draw(image)
        draw.text((35, 40), f"ASSET IDENTIFICATION / {product}", font=font, fill="black")
        draw.text((35, 155), f"BOARD REVISION: {revision}", font=font, fill="black")
        draw.text((35, 280), f"Serial number: SN-{rng.randrange(100000,900000)}", font=font, fill="black")
        image.save(root / label_path, quality=72)
        if i in (2, 6):
            scan_path = f"specs/inspection_{i}.pdf"
            canvas = Canvas(str(root / scan_path))
            scan = Image.new("RGB", (1200, 700), "white")
            draw = ImageDraw.Draw(scan)
            for row, line in enumerate((f"{product} final inspection", f"Approved board revision: {revision}", "Disposition: PASS")):
                draw.text((35, 50+row*100), line, font=font, fill="black")
            canvas.drawImage(ImageReader(scan), 20, 300, width=560, height=327)
            canvas.save()
            scan.close()
            q("scanned_pdf", f"What approved board revision is recorded in the final inspection for {product}?", revision, scan_path)
        else:
            q("image_label", f"What board revision is printed on the asset identification label for {product}?", revision, label_path)
        image.close()
        q("current_revision", f"What is the current maximum junction temperature of {product}?", temperature, current_path)
        q("historical_revision", f"What maximum junction temperature did WITHDRAWN revision 2 specify for {product}?", old, previous_path)
        q("word_merged_table", f"In which quarter does {product} enter customer sampling according to the approved roadmap?", quarter, roadmap_path)
        q("excel_title_and_sheet", f"What is the approved field-replaceable fan assembly part number for {product}?", part, parts_path)
        q("code_default", f"What is the default batch timeout in seconds for the {product} ingest service?", timeout, code_path)
        q("rotated_image" if i % 3 == 1 else "pin_diagram", f"Which backplane pin carries THERM_ALERT# on {product}?", pin, pin_path)
        q("three_file_chain", f"Which firmware version fixed the underlying defect in the production thermal throttle incident on {product}?",
          firmware, log_path, bug_path, release_path)
        q("two_file_chain", f"Which firmware version corresponds to the resolving release for ticket {ticket}?", firmware, bug_path, release_path)
        q("direct_mapping", f"Which firmware version is associated with release identifier {release}?", firmware, release_path)
        q("absent_price", f"What is the unit price of {product} at 10,000 units?", "")
        q("absent_qualifier", f"What is the approved customer sampling quarter for {product}-PRO?", "")
        q("absent_attribute", f"What is the maximum storage temperature of {product}?", "")
        q("absent_pin", f"Which pin carries OVERCURRENT_ALERT# on {product}?", "")
        q("absent_code", f"What is the default network retry count of the {product} ingest service?", "")
    for i in range(80):
        # Hard negatives: related words and plausible numbers, different subjects.
        (root / f"noise/record_{i:03d}.txt").write_text(
            f"Archived note for ZA-{1000+i}. Status: DRAFT; not an approved VX product specification.\n"
            f"Maximum storage temperature: {120+i} C. Sampling: Q2 FY45. Revision: REV-Z{i}.\n"
            f"Thermal throttle validation ticket AUX-{9000+i}. No firmware fix mapping is supplied.\n",
            encoding="utf-8")
    (root / "vendor/unknown.dat").write_bytes(b"unsupported payload")
    (root / "vendor/broken.pdf").write_bytes(b"invalid pdf")
    (root / "vendor/unreadable.txt").write_text("Not evidence. Must be chmod 000 on Linux.")
    # Permutation prevents sequence being a reliable cue to the next answer.
    rng.shuffle(questions)
    gold = root.parent / (root.name + "-questions.json")
    gold.write_text(json.dumps({"kind": "structural_stress", "seed": args.seed, 'variant': args.variant, "queries": questions}, indent=2), encoding="utf-8")
    print(json.dumps({"corpus": str(root), "files": sum(p.is_file() for p in root.rglob("*")),
                      "questions": len(questions), "gold": str(gold)}))


if __name__ == "__main__":
    main()

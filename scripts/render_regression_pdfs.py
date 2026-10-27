"""Render final case PDFs for human visual QA (requires Poppler and Pillow).

Text extraction and AppTest are separate checks. Generated contacts are evidence
for human inspection, not an automatic claim that layout is correct.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from PIL import Image, ImageDraw
from pypdf import PdfReader


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cases', type=Path, help='The cases/ directory from run_batch_d.py')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    tool = shutil.which('pdftoppm')
    if not tool:
        parser.error('Poppler pdftoppm is required; do not claim visual QA without rendering')
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    for pdf in sorted(args.cases.glob('*/report.pdf')):
        case = pdf.parent.name
        out = args.output / case
        out.mkdir(exist_ok=True)
        pages = len(PdfReader(pdf).pages)
        contact = Image.new('RGB', (1200, ((pages + 4) // 5) * 365), 'white')
        draw = ImageDraw.Draw(contact)
        for index in range(pages):
            prefix = out / f'page-{index + 1:03}'
            subprocess.run([tool, '-f', str(index + 1), '-l', str(index + 1), '-singlefile',
                            '-scale-to', '1200', '-jpeg', str(pdf), str(prefix)], check=True, capture_output=True, timeout=30)
            with Image.open(str(prefix) + '.jpg') as image:
                image.load()  # A truncated image is an error, never silently accepted.
                image.thumbnail((234, 337))
                x, y = (index % 5) * 240, (index // 5) * 365
                contact.paste(image, (x, y + 23))
                draw.text((x + 5, y + 5), f'{case} / {index + 1}', fill='black')
        contact_path = args.output / (case + '-contact.jpg')
        contact.save(contact_path, quality=90)
        results.append({'case': case, 'pages': pages, 'pdf_sha256': hashlib.sha256(pdf.read_bytes()).hexdigest(),
                        'contact': contact_path.name, 'all_pages_decoded': True, 'visual_review': 'pending_human_inspection'})
    if len(results) != 4:
        parser.error('Expected four case PDFs; rendering is incomplete')
    (args.output / 'rendered-pages.json').write_text(json.dumps(results, indent=2))
    print(json.dumps({'cases': len(results), 'pages': sum(r['pages'] for r in results)}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

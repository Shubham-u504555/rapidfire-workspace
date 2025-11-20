import pathlib
from pypdf import PdfReader
from rapidfire.utils import ensure_dir

SRC = pathlib.Path("data/knowra")
DST = pathlib.Path("data/corpora")

def extract_text(pdf_path):
    try:
        reader = PdfReader(str(pdf_path))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        return text.strip()
    except Exception as e:
        print(f"⚠️ Failed to read {pdf_path.name}: {e}")
        return ""

def main():
    ensure_dir(DST)
    pdfs = list(SRC.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {SRC}")
        return

    for pdf in pdfs:
        txt = extract_text(pdf)
        if not txt:
            continue
        out_path = DST / f"{pdf.stem}.txt"
        out_path.write_text(txt, encoding="utf-8")
        print(f"✅ Saved {out_path}")

if __name__ == "__main__":
    main()

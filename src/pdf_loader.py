import fitz
from pathlib import Path
def load_pdfs(raw_dir="data/raw"):
    pdf_files=Path(raw_dir).glob("*.pdf")
    documents=[]
    for pdf in pdf_files:
        document=fitz.open(pdf)
        pages=[]
        for page_number, page in enumerate(document, start=1):
            pages.append({
                "page": page_number,
                "text": page.get_text()
            })
        document.close()
        documents.append({
            "filename": pdf.name,
            "pages": pages
        })
    return documents
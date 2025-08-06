# # io_utils.py
# import os, hashlib, pdfplumber
# from docx import Document as DocxDocument
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.schema import Document as LCDocument
# from config import CHUNK_SIZE, CHUNK_OVERLAP

# def read_pdf(file_path: str) -> str:
#     parts = []
#     with pdfplumber.open(file_path) as pdf:
#         for page in pdf.pages:
#             text = page.extract_text()
#             if text: parts.append(text.strip())
#             for table in page.extract_tables() or []:
#                 for row in table:
#                     row = [str(c).strip() for c in row if c and str(c).strip()]
#                     if row: parts.append(" | ".join(row))
#     return "\n".join(parts)

# def read_docx(file_path: str) -> str:
#     doc = DocxDocument(file_path)
#     parts = []
#     for para in doc.paragraphs:
#         t = para.text.strip()
#         if t: parts.append(t)
#     for table in doc.tables:
#         for row in table.rows:
#             cells = [c.text.strip() for c in row.cells if c.text.strip()]
#             if cells: parts.append(" | ".join(cells))
#     return "\n".join(parts)

# def file_text(path: str) -> str:
#     p = path.lower()
#     if p.endswith(".pdf"): return read_pdf(path)
#     if p.endswith(".docx"): return read_docx(path)
#     raise ValueError("Unsupported file type (only .pdf, .docx).")

# def content_sha256(path: str) -> str:
#     h = hashlib.sha256()
#     with open(path, "rb") as f:
#         for chunk in iter(lambda: f.read(1024*1024), b""): h.update(chunk)
#     return h.hexdigest()

# def split_to_documents(text: str, meta: dict) -> list[LCDocument]:
#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
#         separators=["\n\n", "\n", " ", ""]
#     )
#     return [LCDocument(page_content=ch, metadata={**meta, "chunk_index": i})
#             for i, ch in enumerate(splitter.split_text(text))]

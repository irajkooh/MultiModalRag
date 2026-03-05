"""
Multimodal document processor: handles PDFs (text, tables, charts/images), 
DOCX, XLSX, CSV, and scanned images via OCR.
"""
import os
import io
import base64
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import hashlib

from PIL import Image
import pytesseract
from pypdf import PdfReader
import pandas as pd

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {
    ".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif",
    ".docx", ".xlsx", ".csv", ".txt"
}


def get_file_hash(filepath: str) -> str:
    """Compute MD5 hash of file for dedup."""
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def image_to_base64(image: Image.Image, max_size: Tuple[int, int] = (1024, 1024)) -> str:
    """Resize and encode a PIL image to base64."""
    image.thumbnail(max_size, Image.LANCZOS)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def ocr_image(image: Image.Image) -> str:
    """Run Tesseract OCR on a PIL image."""
    try:
        text = pytesseract.image_to_string(image, config="--oem 3 --psm 6")
        return text.strip()
    except Exception as e:
        logger.warning(f"OCR failed: {e}")
        return ""


def extract_pdf(filepath: str) -> List[Dict[str, Any]]:
    """
    Extract content from PDF:
    - Text pages → text chunks
    - Pages with embedded images → OCR + base64 stored in metadata
    - Tables detected via simple heuristic (pipe/tab-separated lines)
    Returns list of chunk dicts: {text, metadata}
    """
    chunks = []
    reader = PdfReader(filepath)
    filename = Path(filepath).name

    for page_num, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        
        # Detect table-like content
        lines = page_text.split("\n")
        table_lines = [l for l in lines if l.count("|") > 2 or l.count("\t") > 2]
        has_table = len(table_lines) > 3

        chunk_meta = {
            "source": filename,
            "page": page_num,
            "type": "table" if has_table else "text",
            "file_hash": get_file_hash(filepath),
        }

        if page_text.strip():
            chunks.append({
                "text": f"[Source: {filename}, Page {page_num}]\n{page_text.strip()}",
                "metadata": chunk_meta,
            })

        # Extract embedded images from page
        try:
            if hasattr(page, "images") and page.images:
                for img_idx, img_obj in enumerate(page.images):
                    try:
                        pil_img = Image.open(io.BytesIO(img_obj.data))
                        ocr_text = ocr_image(pil_img)
                        img_b64 = image_to_base64(pil_img)
                        img_meta = {
                            **chunk_meta,
                            "type": "image",
                            "image_index": img_idx,
                            "image_b64": img_b64,
                        }
                        text_content = ocr_text if ocr_text else f"[Image on page {page_num}]"
                        chunks.append({
                            "text": f"[Source: {filename}, Page {page_num}, Image {img_idx}]\n{text_content}",
                            "metadata": img_meta,
                        })
                    except Exception as e:
                        logger.debug(f"Skipping embedded image: {e}")
        except Exception as e:
            logger.debug(f"Image extraction error on page {page_num}: {e}")

    return chunks


def extract_image(filepath: str) -> List[Dict[str, Any]]:
    """OCR a standalone image file."""
    filename = Path(filepath).name
    pil_img = Image.open(filepath).convert("RGB")
    ocr_text = ocr_image(pil_img)
    img_b64 = image_to_base64(pil_img)

    return [{
        "text": f"[Source: {filename}]\n{ocr_text if ocr_text else '[Image with no detectable text]'}",
        "metadata": {
            "source": filename,
            "type": "image",
            "image_b64": img_b64,
            "file_hash": get_file_hash(filepath),
        },
    }]


def extract_docx(filepath: str) -> List[Dict[str, Any]]:
    """Extract text and tables from DOCX."""
    from docx import Document
    filename = Path(filepath).name
    doc = Document(filepath)
    chunks = []
    file_hash = get_file_hash(filepath)

    full_text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    if full_text:
        chunks.append({
            "text": f"[Source: {filename}]\n{full_text}",
            "metadata": {"source": filename, "type": "text", "file_hash": file_hash},
        })

    for t_idx, table in enumerate(doc.tables):
        rows = [[cell.text.strip() for cell in row.cells] for row in table.rows]
        table_text = "\n".join(" | ".join(row) for row in rows)
        if table_text.strip():
            chunks.append({
                "text": f"[Source: {filename}, Table {t_idx+1}]\n{table_text}",
                "metadata": {"source": filename, "type": "table", "table_index": t_idx, "file_hash": file_hash},
            })
    return chunks


def extract_xlsx(filepath: str) -> List[Dict[str, Any]]:
    """Extract all sheets from XLSX as text."""
    filename = Path(filepath).name
    chunks = []
    file_hash = get_file_hash(filepath)
    xf = pd.ExcelFile(filepath)
    for sheet in xf.sheet_names:
        df = pd.read_excel(filepath, sheet_name=sheet)
        text = df.to_string(index=False)
        chunks.append({
            "text": f"[Source: {filename}, Sheet: {sheet}]\n{text}",
            "metadata": {"source": filename, "type": "table", "sheet": sheet, "file_hash": file_hash},
        })
    return chunks


def extract_csv(filepath: str) -> List[Dict[str, Any]]:
    filename = Path(filepath).name
    df = pd.read_csv(filepath)
    text = df.to_string(index=False)
    return [{
        "text": f"[Source: {filename}]\n{text}",
        "metadata": {"source": filename, "type": "table", "file_hash": get_file_hash(filepath)},
    }]


def extract_txt(filepath: str) -> List[Dict[str, Any]]:
    filename = Path(filepath).name
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    return [{
        "text": f"[Source: {filename}]\n{text}",
        "metadata": {"source": filename, "type": "text", "file_hash": get_file_hash(filepath)},
    }]


def process_document(filepath: str) -> List[Dict[str, Any]]:
    """Route file to the correct extractor."""
    ext = Path(filepath).suffix.lower()
    if ext == ".pdf":
        return extract_pdf(filepath)
    elif ext in {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif"}:
        return extract_image(filepath)
    elif ext == ".docx":
        return extract_docx(filepath)
    elif ext == ".xlsx":
        return extract_xlsx(filepath)
    elif ext == ".csv":
        return extract_csv(filepath)
    elif ext == ".txt":
        return extract_txt(filepath)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> List[str]:
    """Split long text into overlapping chunks."""
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start += chunk_size - overlap
    return chunks


def process_document_chunked(filepath: str) -> List[Dict[str, Any]]:
    """Process a document and chunk large text blocks."""
    raw_chunks = process_document(filepath)
    final_chunks = []
    for chunk in raw_chunks:
        text = chunk["text"]
        meta = chunk["metadata"]
        sub_texts = chunk_text(text)
        for i, sub in enumerate(sub_texts):
            final_chunks.append({
                "text": sub,
                "metadata": {**meta, "chunk_index": i},
            })
    return final_chunks

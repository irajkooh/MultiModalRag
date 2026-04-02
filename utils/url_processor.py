"""
Web URL crawler and content extractor for RAG indexing.
Fetches a URL and linked pages up to a configurable depth.
Also downloads and processes any PDF files linked on the pages.
"""
import re
import logging
import tempfile
import os
from pathlib import Path
from typing import List, Dict, Any, Set, Tuple
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from utils.document_processor import chunk_text, process_document_chunked

logger = logging.getLogger(__name__)

MAX_PAGES = 50
REQUEST_TIMEOUT = 15
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; MultimodalRAG/1.0)"}


def _is_same_domain(url: str, base_url: str) -> bool:
    """Return True if url shares the same host (or is a subdomain) as base_url."""
    try:
        base_host = urlparse(base_url).netloc.lower()
        url_host = urlparse(url).netloc.lower()
        return url_host == base_host or url_host.endswith("." + base_host)
    except Exception:
        return False


def _fetch_html(url: str) -> Tuple[str | None, str]:
    """GET a URL and return (text, content_type). Returns (None, '') on failure."""
    try:
        resp = requests.get(url, timeout=REQUEST_TIMEOUT, headers=HEADERS, allow_redirects=True)
        resp.raise_for_status()
        ct = resp.headers.get("content-type", "").lower()
        return resp.text, ct
    except Exception as e:
        logger.warning(f"Failed to fetch {url}: {e}")
        return None, ""


def _fetch_binary(url: str) -> bytes | None:
    """Download raw bytes (for PDFs). Returns None on failure."""
    try:
        resp = requests.get(url, timeout=REQUEST_TIMEOUT, headers=HEADERS, allow_redirects=True)
        resp.raise_for_status()
        return resp.content
    except Exception as e:
        logger.warning(f"Failed to download {url}: {e}")
        return None


def _extract_text_and_links(
    html: str, base_url: str
) -> Tuple[str, List[str], List[str]]:
    """
    Parse HTML and return:
      clean_text   — visible page text with boilerplate removed
      html_links   — absolute http/https links to other HTML pages
      pdf_links    — absolute http/https links whose path ends in .pdf
    """
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
        tag.decompose()

    text = soup.get_text(separator="\n", strip=True)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    html_links: List[str] = []
    pdf_links: List[str] = []
    seen: Set[str] = set()

    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        absolute = urljoin(base_url, href)
        parsed = urlparse(absolute)
        if parsed.scheme not in ("http", "https"):
            continue
        # Normalise: drop fragment
        clean = parsed._replace(fragment="").geturl()
        if clean in seen:
            continue
        seen.add(clean)

        path_lower = parsed.path.lower()
        if path_lower.endswith(".pdf") or ".pdf?" in path_lower:
            pdf_links.append(clean)
        else:
            html_links.append(clean)

    return text, html_links, pdf_links


def _process_pdf_url(pdf_url: str, start_url: str) -> List[Dict[str, Any]]:
    """
    Download a PDF from pdf_url, process it with the existing PDF extractor,
    fix up source metadata to point at the PDF URL, and return the chunks.
    """
    logger.info(f"Downloading PDF: {pdf_url}")
    data = _fetch_binary(pdf_url)
    if not data:
        return []

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    try:
        chunks = process_document_chunked(tmp_path)
        tmp_name = Path(tmp_path).name
        for chunk in chunks:
            # Replace the temp filename with the real PDF URL in chunk text
            chunk["text"] = chunk["text"].replace(tmp_name, pdf_url)
            chunk["metadata"]["source"] = start_url
            chunk["metadata"]["page_url"] = pdf_url
            chunk["metadata"]["pdf_source"] = pdf_url
        return chunks
    except Exception as e:
        logger.warning(f"Failed to process PDF {pdf_url}: {e}")
        return []
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def crawl_url(
    start_url: str,
    max_depth: int = 2,
    max_pages: int = MAX_PAGES,
    same_domain_only: bool = True,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    BFS-crawl start_url up to max_depth link levels.

    - HTML pages are scraped for text.
    - PDF files linked from any crawled page are downloaded and indexed.
    - All chunks share source=start_url so they can be removed as a unit.

    Returns:
        chunks       — list of {text, metadata} dicts for VectorStoreManager
        crawled_urls — list of all URLs successfully processed
    """
    visited_html: Set[str] = set()
    visited_pdf: Set[str] = set()
    crawled_urls: List[str] = []
    all_chunks: List[Dict[str, Any]] = []

    # BFS queue: (url, depth)
    queue: List[Tuple[str, int]] = [(start_url, 0)]

    while queue and len(visited_html) < max_pages:
        url, depth = queue.pop(0)
        if url in visited_html:
            continue
        if same_domain_only and depth > 0 and not _is_same_domain(url, start_url):
            continue
        visited_html.add(url)

        logger.info(f"Crawling depth={depth}: {url}")
        content, ct = _fetch_html(url)
        if not content or ("text/html" not in ct and "text/plain" not in ct):
            continue

        text, html_links, pdf_links = _extract_text_and_links(content, url)

        if text:
            crawled_urls.append(url)
            sub_texts = chunk_text(text)
            for i, sub in enumerate(sub_texts):
                all_chunks.append({
                    "text": f"[Source: {url}]\n{sub}",
                    "metadata": {
                        "source": start_url,
                        "page_url": url,
                        "type": "web",
                        "depth": depth,
                        "chunk_index": i,
                    },
                })

        # Process PDFs found on this page (no depth restriction for PDFs)
        for pdf_url in pdf_links:
            if pdf_url not in visited_pdf:
                if same_domain_only and not _is_same_domain(pdf_url, start_url):
                    continue
                visited_pdf.add(pdf_url)
                pdf_chunks = _process_pdf_url(pdf_url, start_url)
                if pdf_chunks:
                    all_chunks.extend(pdf_chunks)
                    crawled_urls.append(pdf_url)

        # Enqueue child HTML pages if depth limit not yet reached
        if depth < max_depth:
            for link in html_links:
                if link not in visited_html:
                    queue.append((link, depth + 1))

    logger.info(
        f"Crawl complete: {len(crawled_urls)} pages/files, "
        f"{len(all_chunks)} chunks from {start_url}"
    )
    return all_chunks, crawled_urls

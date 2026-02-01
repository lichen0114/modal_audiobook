"""EPUB file parser - extracts chapters and text content."""

import re
from dataclasses import dataclass
from pathlib import Path

import ebooklib
from bs4 import BeautifulSoup
from ebooklib import epub


@dataclass
class Chapter:
    """Represents a chapter from an EPUB book."""

    index: int
    title: str
    content: str

    def __repr__(self) -> str:
        return f"Chapter({self.index}, '{self.title}', {len(self.content)} chars)"


def extract_text_from_html(html_content: bytes | str) -> str:
    """Extract clean text from HTML content."""
    if isinstance(html_content, bytes):
        html_content = html_content.decode("utf-8", errors="ignore")

    soup = BeautifulSoup(html_content, "lxml")

    # Remove script and style elements
    for element in soup(["script", "style", "head", "meta", "link"]):
        element.decompose()

    # Get text with paragraph preservation
    paragraphs = []
    for p in soup.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6", "div"]):
        text = p.get_text(strip=True)
        if text:
            paragraphs.append(text)

    # If no paragraphs found, get all text
    if not paragraphs:
        text = soup.get_text(separator="\n", strip=True)
        paragraphs = [line.strip() for line in text.split("\n") if line.strip()]

    return "\n\n".join(paragraphs)


def extract_title_from_html(html_content: bytes | str) -> str | None:
    """Try to extract chapter title from HTML."""
    if isinstance(html_content, bytes):
        html_content = html_content.decode("utf-8", errors="ignore")

    soup = BeautifulSoup(html_content, "lxml")

    # Try heading tags first
    for tag in ["h1", "h2", "h3"]:
        heading = soup.find(tag)
        if heading:
            title = heading.get_text(strip=True)
            if title and len(title) < 200:  # Reasonable title length
                return title

    # Try title tag
    title_tag = soup.find("title")
    if title_tag:
        title = title_tag.get_text(strip=True)
        if title and len(title) < 200:
            return title

    return None


def is_content_chapter(item: epub.EpubItem, text: str) -> bool:
    """Determine if an item is actual book content (not TOC, cover, etc.)."""
    if not text or len(text.strip()) < 100:
        return False

    # Check filename for non-content indicators
    filename = item.get_name().lower() if item.get_name() else ""
    skip_patterns = [
        "toc", "nav", "cover", "title", "copyright",
        "dedication", "halftitle", "frontmatter", "backmatter",
        "index", "appendix", "colophon", "about"
    ]

    for pattern in skip_patterns:
        if pattern in filename:
            # Still include if there's substantial content
            if len(text.strip()) > 500:
                return True
            return False

    return True


def parse_epub(epub_path: str | Path) -> list[Chapter]:
    """
    Parse an EPUB file and extract chapters.

    Args:
        epub_path: Path to the EPUB file

    Returns:
        List of Chapter objects with extracted content
    """
    epub_path = Path(epub_path)
    if not epub_path.exists():
        raise FileNotFoundError(f"EPUB file not found: {epub_path}")

    book = epub.read_epub(str(epub_path))
    chapters = []
    chapter_index = 1

    # Get spine items (reading order)
    spine_items = []
    for item_id, linear in book.spine:
        item = book.get_item_with_id(item_id)
        if item:
            spine_items.append(item)

    # If no spine, fall back to all documents
    if not spine_items:
        spine_items = list(book.get_items_of_type(ebooklib.ITEM_DOCUMENT))

    for item in spine_items:
        if item.get_type() != ebooklib.ITEM_DOCUMENT:
            continue

        content = item.get_content()
        text = extract_text_from_html(content)

        if not is_content_chapter(item, text):
            continue

        # Try to get title
        title = extract_title_from_html(content)
        if not title:
            title = f"Chapter {chapter_index}"

        chapters.append(Chapter(
            index=chapter_index,
            title=title,
            content=text
        ))
        chapter_index += 1

    return chapters


def get_book_metadata(epub_path: str | Path) -> dict:
    """Extract metadata from EPUB file."""
    book = epub.read_epub(str(epub_path))

    metadata = {
        "title": None,
        "author": None,
        "language": None,
    }

    # Get title
    title = book.get_metadata("DC", "title")
    if title:
        metadata["title"] = title[0][0]

    # Get author
    creator = book.get_metadata("DC", "creator")
    if creator:
        metadata["author"] = creator[0][0]

    # Get language
    language = book.get_metadata("DC", "language")
    if language:
        metadata["language"] = language[0][0]

    return metadata

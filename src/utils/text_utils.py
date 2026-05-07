"""Text processing utilities for research content."""

from __future__ import annotations

import re


def truncate(text: str, max_chars: int = 50000) -> str:
    """Truncate text to max_chars, adding an indicator if truncated."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n\n... [truncated, {len(text) - max_chars} chars omitted]"


def clean_html_to_text(html: str) -> str:
    """Strip HTML tags and normalize whitespace for plain text."""
    # Remove script and style blocks
    text = re.sub(r"<(script|style)[^>]*>[\s\S]*?</\1>", "", html, flags=re.IGNORECASE)
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Decode common HTML entities
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    text = text.replace("&quot;", '"').replace("&#39;", "'").replace("&nbsp;", " ")
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_markdown_sections(markdown: str) -> dict[str, str]:
    """Split a markdown document into sections by headings.

    Returns a dict mapping heading text to section content.
    """
    sections: dict[str, str] = {}
    current_heading = "_preamble"
    current_lines: list[str] = []

    for line in markdown.split("\n"):
        heading_match = re.match(r"^(#{1,6})\s+(.+)$", line)
        if heading_match:
            # Save previous section
            if current_lines:
                sections[current_heading] = "\n".join(current_lines).strip()
            current_heading = heading_match.group(2).strip()
            current_lines = []
        else:
            current_lines.append(line)

    # Save last section
    if current_lines:
        sections[current_heading] = "\n".join(current_lines).strip()

    return sections


_QUERY_NOISE_WORDS = [
    "arxiv", "github", "gitlab", "hugging face", "semantic scholar",
    "papers with code", "site:arxiv.org", "site:github.com",
    "site:gitlab.com", "site:huggingface.co",
    "open source implementation repository",
]


def clean_search_query(query: str) -> str:
    """Remove platform names and filler words that pollute search queries."""
    cleaned = query
    for word in _QUERY_NOISE_WORDS:
        cleaned = re.sub(rf"\b{re.escape(word)}\b", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b(19|20)\d{2}\b", "", cleaned)  # stray years
    return re.sub(r"\s+", " ", cleaned).strip()


def normalize_url(url: str) -> str:
    """Normalize a URL by removing trailing slashes and fragments."""
    url = url.split("#")[0]  # Remove fragment
    url = url.rstrip("/")
    return url

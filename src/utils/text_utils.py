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


def normalize_url(url: str) -> str:
    """Normalize a URL by removing trailing slashes and fragments."""
    url = url.split("#")[0]  # Remove fragment
    url = url.rstrip("/")
    return url

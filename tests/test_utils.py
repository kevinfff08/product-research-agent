"""Tests for utility modules."""

from __future__ import annotations

import pytest

from src.utils.json_repair import repair_json
from src.utils.text_utils import (
    truncate,
    clean_html_to_text,
    extract_markdown_sections,
    normalize_url,
    english_search_query,
    is_useful_english_query,
)


class TestJsonRepair:
    def test_valid_json(self):
        assert repair_json('{"key": "value"}') == {"key": "value"}

    def test_json_array(self):
        assert repair_json('[1, 2, 3]') == [1, 2, 3]

    def test_markdown_fences(self):
        text = '```json\n{"key": "value"}\n```'
        assert repair_json(text) == {"key": "value"}

    def test_trailing_comma(self):
        text = '{"key": "value",}'
        assert repair_json(text) == {"key": "value"}

    def test_trailing_comma_array(self):
        text = '[1, 2, 3,]'
        assert repair_json(text) == [1, 2, 3]

    def test_json_with_surrounding_text(self):
        text = 'Here is the result: {"key": "value"} hope this helps'
        assert repair_json(text) == {"key": "value"}

    def test_empty_string(self):
        assert repair_json("") is None

    def test_none_text(self):
        assert repair_json("") is None

    def test_completely_invalid(self):
        assert repair_json("not json at all") is None

    def test_nested_json(self):
        text = '{"a": {"b": [1, 2]}, "c": true}'
        result = repair_json(text)
        assert result == {"a": {"b": [1, 2]}, "c": True}


class TestTruncate:
    def test_short_text(self):
        assert truncate("hello", 100) == "hello"

    def test_exact_length(self):
        text = "a" * 100
        assert truncate(text, 100) == text

    def test_truncation(self):
        text = "a" * 200
        result = truncate(text, 100)
        assert result.startswith("a" * 100)
        assert "truncated" in result
        assert "100 chars omitted" in result


class TestCleanHtmlToText:
    def test_simple_html(self):
        assert clean_html_to_text("<p>Hello</p>") == "Hello"

    def test_script_removal(self):
        html = "<p>Text</p><script>alert('hi')</script><p>More</p>"
        result = clean_html_to_text(html)
        assert "alert" not in result
        assert "Text" in result
        assert "More" in result

    def test_entity_decoding(self):
        html = "AT&amp;T &lt;test&gt; &quot;quoted&quot;"
        result = clean_html_to_text(html)
        assert 'AT&T <test> "quoted"' == result

    def test_whitespace_normalization(self):
        html = "<p>  lots   of   spaces  </p>"
        result = clean_html_to_text(html)
        assert result == "lots of spaces"


class TestExtractMarkdownSections:
    def test_simple_sections(self):
        md = "# Title\nContent 1\n## Sub\nContent 2"
        sections = extract_markdown_sections(md)
        assert "Title" in sections
        assert "Sub" in sections
        assert "Content 1" in sections["Title"]
        assert "Content 2" in sections["Sub"]

    def test_preamble(self):
        md = "Before any heading\n# First"
        sections = extract_markdown_sections(md)
        assert "_preamble" in sections
        assert "Before any heading" in sections["_preamble"]

    def test_empty(self):
        sections = extract_markdown_sections("")
        assert sections == {"_preamble": ""}


class TestNormalizeUrl:
    def test_trailing_slash(self):
        assert normalize_url("https://example.com/") == "https://example.com"

    def test_fragment_removal(self):
        assert normalize_url("https://example.com/page#section") == "https://example.com/page"

    def test_clean_url(self):
        assert normalize_url("https://example.com/page") == "https://example.com/page"


class TestEnglishSearchQuery:
    def test_translates_common_chinese_terms(self):
        query = english_search_query("文本到图像生成 benchmark evaluation survey")
        assert "text to image generation" in query
        assert "文本" not in query

    def test_rejects_generic_leftovers(self):
        assert not is_useful_english_query("survey benchmark")
        assert is_useful_english_query("document to PowerPoint generation benchmark")

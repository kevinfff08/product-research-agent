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

_ZH_QUERY_TRANSLATIONS = {
    "大语言模型": "large language models",
    "长上下文处理": "long context processing",
    "文档解析": "document parsing",
    "文本摘要与信息抽取": "text summarization information extraction",
    "基于大语言模型的端到端文稿转PPT生成": "LLM document to PowerPoint generation",
    "端到端文稿转PPT生成": "end-to-end document to PowerPoint generation",
    "AI ppt制作": "AI PowerPoint generation",
    "PPT制作": "PowerPoint generation",
    "先提纲后排版的两阶段生成路线": "outline to slide layout generation",
    "层级大纲生成": "hierarchical outline generation",
    "章节分割": "section segmentation",
    "页面级内容规划": "slide-level content planning",
    "模板匹配": "template matching",
    "基于模板库与规则引擎的高可控PPT生成": "template and rule based controllable PowerPoint generation",
    "PPT模板工程": "PowerPoint template engineering",
    "模板工程": "template engineering",
    "规则引擎": "rule engine",
    "设计系统": "design systems",
    "页面语义分类": "slide page semantic classification",
    "PPT渲染引擎": "PowerPoint rendering engine",
    "占位符映射": "placeholder mapping",
    "品牌规范控制": "brand guideline control",
    "样式约束": "style constraints",
    "多模态图文增强的演示生成路线": "multimodal image grounded presentation generation",
    "多模态大模型": "multimodal large language models",
    "文本到图像生成": "text to image generation",
    "图像检索": "image retrieval",
    "图标检索与生成": "icon retrieval and generation",
    "OCR与图像理解": "OCR and image understanding",
    "信息图生成": "infographic generation",
    "版权与素材管理": "copyright and asset management",
    "视觉风格控制": "visual style control",
    "面向数据报告的图表与表格自动生成路线": "automatic chart and table generation for data reports",
    "表格解析": "table parsing",
    "数据抽取": "data extraction",
    "文本到图表映射": "text to chart mapping",
    "图表推荐": "chart recommendation",
}

_GENERIC_QUERY_TERMS = {
    "survey benchmark",
    "benchmark survey",
    "paper benchmark",
    "research paper",
    "benchmark evaluation",
    "evaluation benchmark",
}


def clean_search_query(query: str) -> str:
    """Remove platform names and filler words that pollute search queries."""
    cleaned = query
    for word in _QUERY_NOISE_WORDS:
        cleaned = re.sub(rf"\b{re.escape(word)}\b", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b(19|20)\d{2}\b", "", cleaned)  # stray years
    return re.sub(r"\s+", " ", cleaned).strip()


def english_search_query(query: str, *, ascii_only: bool = False) -> str:
    """Return a search-provider friendly English query.

    The research planner is instructed to emit English, but LLMs can still
    leak Chinese path titles or technology names into expanded queries.  This
    function translates common technical phrases, removes remaining CJK text,
    and rejects empty/generic leftovers before hitting strict search APIs.
    """
    cleaned = clean_search_query(query)
    for zh, en in sorted(_ZH_QUERY_TRANSLATIONS.items(), key=lambda item: len(item[0]), reverse=True):
        cleaned = cleaned.replace(zh, en)
    cleaned = re.sub(r"[\u3400-\u9fff]+", " ", cleaned)
    cleaned = cleaned.replace("PPT", "PowerPoint")
    cleaned = re.sub(r"[^\x00-\x7F]+", " ", cleaned)
    cleaned = clean_search_query(cleaned)
    if ascii_only:
        cleaned = cleaned.encode("ascii", errors="ignore").decode("ascii")
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" -_:;,")
    return cleaned


def is_useful_english_query(query: str, *, min_chars: int = 5) -> bool:
    """Return True when a normalized query still carries domain information."""
    normalized = re.sub(r"\s+", " ", query.lower()).strip()
    if len(normalized) < min_chars:
        return False
    if normalized in _GENERIC_QUERY_TERMS:
        return False
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9+\-.]{1,}", normalized)
    informative = [
        word for word in words
        if word not in {"survey", "benchmark", "paper", "research", "evaluation", "recent"}
    ]
    return len(informative) >= 1


def normalize_url(url: str) -> str:
    """Normalize a URL by removing trailing slashes and fragments."""
    url = url.split("#")[0]  # Remove fragment
    url = url.rstrip("/")
    return url

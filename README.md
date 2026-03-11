# Product & Industry Research Agent

AI-powered research agent that takes a product idea or concept as input and generates a comprehensive technology landscape report.

## Features

- **Idea Decomposition**: Analyzes vague product concepts into concrete, researched paths
- **Multi-Source Research**: Searches industry products, academic papers, open-source repos, and tech blogs
- **PhD-Level Analysis**: Deep analysis of academic papers including principles, methods, experiments, conclusions, and limitations
- **Technology Maturity Mapping**: Maps each technology across 5 maturity stages (early prototype -> mature -> academic frontier)
- **Reputation Scoring**: Evaluates source credibility (company reputation, lab prestige, publication venue, community reception)
- **Engineering Analysis**: Assesses deployment readiness, code quality, and implementation feasibility
- **Structured Reports**: Generates comprehensive Markdown reports with citations

## Architecture

The system uses a **Coordinator-driven multi-agent pipeline**:

```
User Input (vague idea)
    |
    v
[IdeaDecomposer] -> DecompositionResult (multiple research paths)
    |
    v
[ResearchPlanner] -> ResearchPlan (search queries per source)
    |
    +--- parallel ---+--- parallel ---+
    v                v                v
[IndustryResearcher] [AcademicResearcher] [EngineeringAnalyst]
    |                |                |
    +------- all ----+-------+--------+
             |               |
             v               v
    [ReputationScorer]  [MaturityMapper]
             |               |
             +-------+-------+
                     v
           [ReportGenerator] -> ResearchReport
                     |
                     v
           [MarkdownReporter] -> output/report.md
```

Each "agent" is a Python class that uses Claude (via CLIProxyAPI) with carefully crafted prompt templates and structured Pydantic output models.

## Prerequisites

- **Python 3.14+** (via conda `research_tools` environment)
- **CLIProxyAPI** installed and configured (for setup-token mode), OR an Anthropic API key
- **Tavily API key** (for web search) - get one at https://tavily.com

### Optional API Keys

- **Semantic Scholar API key** - increases rate limits for academic search
- **GitHub personal access token** - increases rate limits for repo analysis

## Installation

1. **Clone and enter the project directory**:
   ```bash
   cd D:\Kevin\PhD\Project\ResearchTools\ProductResearch
   ```

2. **Activate the conda environment**:
   ```bash
   conda activate research_tools
   ```

3. **Install the project** (editable mode):
   ```bash
   pip install -e .
   ```

4. **Install dev dependencies** (for testing):
   ```bash
   pip install -e ".[dev]"
   ```

5. **Configure environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

## Configuration

### Environment Variables (`.env`)

| Variable | Required | Description |
|----------|----------|-------------|
| `LLM_MODE` | Yes | `api-key` (direct) or `setup-token` (CLIProxyAPI proxy) |
| `ANTHROPIC_API_KEY` | If api-key mode | Your Anthropic API key |
| `LLM_PROXY_URL` | No | CLIProxyAPI address (default: `http://localhost:8317`) |
| `LLM_MODEL` | No | Claude model to use (default: `claude-sonnet-4-20250514`) |
| `TAVILY_API_KEY` | Yes | Tavily AI search API key |
| `SEMANTIC_SCHOLAR_API_KEY` | No | Semantic Scholar API key (increases rate limits) |
| `GITHUB_TOKEN` | No | GitHub personal access token (increases rate limits) |

### Research Configuration (`config/default.yaml`)

Key settings:
- `research.max_paths`: Maximum number of decomposition paths (default: 5)
- `weights`: Research type weights (industry: 0.60, academic: 0.25, community: 0.15)
- `api.*`: Rate limits and result limits for each API
- `llm.*`: Temperature and token settings per task type

## Usage

### Quick Start (Windows)

```batch
start.bat research "AI-powered code review tool"
```

### Quick Start (Linux/macOS)

```bash
chmod +x start.sh
./start.sh research "AI-powered code review tool"
```

### CLI Commands

```bash
# Run a research session
python -m src research "your product idea here"

# With options
python -m src research "real-time video translation app" --depth deep --max-paths 3

# List past sessions
python -m src list-sessions

# Show details of a past session
python -m src show <session_id>

# Check status of a session
python -m src status <session_id>
```

### Research Depth Options

| Depth | Description | Estimated Time |
|-------|-------------|----------------|
| `quick` | Fast scan with fewer sources | 5-10 min |
| `comprehensive` | Balanced depth and breadth (default) | 15-30 min |
| `deep` | Maximum depth, more papers, more repos | 30-60 min |

### Output Formats

- `markdown` (default): Generates `output/<session_id>/report.md`
- `docx`: Generates `output/<session_id>/report.docx` (requires `pip install -e ".[docx]"`)
- `both`: Generates both formats

## Project Structure

```
ProductResearch/
├── src/
│   ├── cli.py                  # Typer CLI entry point
│   ├── orchestrator.py         # Central pipeline coordinator
│   ├── logging_config.py       # RotatingFileHandler (10MB rotation)
│   ├── llm/
│   │   ├── client.py           # LLMClient (CLIProxyAPI compatible)
│   │   └── prompts/v1/         # Prompt templates
│   ├── agents/                 # Research agent classes
│   │   ├── base.py             # BaseAgent abstract class
│   │   ├── idea_decomposer.py  # Vague input -> research paths
│   │   ├── research_planner.py # Paths -> search queries
│   │   ├── industry_researcher.py
│   │   ├── academic_researcher.py
│   │   ├── engineering_analyst.py
│   │   ├── reputation_scorer.py
│   │   ├── maturity_mapper.py
│   │   └── report_generator.py
│   ├── apis/                   # External API clients
│   │   ├── base.py             # BaseAPIClient (rate limiting + cache)
│   │   ├── tavily_client.py
│   │   ├── semantic_scholar.py
│   │   ├── github_client.py
│   │   ├── arxiv_client.py
│   │   └── web_scraper.py
│   ├── models/                 # Pydantic v2 data models
│   ├── storage/                # JSON session storage
│   ├── reporters/              # Report generation
│   └── utils/                  # Utilities
├── config/                     # YAML configuration
├── tests/                      # pytest test suite
├── data/                       # Runtime data (gitignored)
├── output/                     # Generated reports (gitignored)
└── logs/                       # Log files (gitignored)
```

## Logging

- Logs are written to `logs/product_research.log`
- Log files rotate at **10 MB** (new file created automatically)
- Old log files are **never deleted** (up to 999 backups kept)
- Console output shows WARNING and above; file captures DEBUG and above

## Testing

```bash
# Run all tests
conda run -n research_tools python -m pytest tests/ -v

# Run with coverage
conda run -n research_tools python -m pytest tests/ -v --cov=src --cov-report=term-missing

# Run only unit tests (no API access needed)
conda run -n research_tools python -m pytest tests/ -v -m "not integration"
```

## Development Status

### Completed
- **Phase 1 - Foundation**: Project skeleton, logging, LLM client, all Pydantic models, BaseAgent, storage, utilities
- **Phase 2 - API Clients**: Tavily, Semantic Scholar, GitHub, arXiv (XML), web scraper (httpx + BeautifulSoup)
- **Phase 3 - Strategy Agents**: IdeaDecomposer (idea -> research paths), ResearchPlanner (paths -> search queries)
- **Phase 4 - Research Agents**: IndustryResearcher (web + repos), AcademicResearcher (S2 + arXiv), EngineeringAnalyst (code quality + deployment)
- **Phase 5 - Scoring Agents**: ReputationScorer (credibility scoring), MaturityMapper (technology maturity stages)
- **Phase 6 - Integration**: Orchestrator (async pipeline), ReportGenerator (LLM synthesis), MarkdownReporter, Typer CLI
- **Full test suite**: 139 tests passing

### Optional Future Work
- DOCX report generation (`pip install -e ".[docx]"`)
- Integration tests with real API keys
- Web dashboard for browsing reports

## License

MIT

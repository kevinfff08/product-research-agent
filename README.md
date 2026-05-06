# 产品技术调研 Agent

这是一个本地优先的命令行调研工具。它接收两阶段输入：先输入一个短标题，再输入更具体的细致描述。系统会据此拆解研究路径，并行搜索产业产品、学术论文和开源实现，评估工程可行性、技术成熟度和来源可信度，最后生成可复核的 Markdown 或 DOCX 调研报告。

项目目标不是罗列搜索结果，而是形成可决策的技术 landscape：哪些路线值得做，哪些能力可以复用，哪些风险需要先验证，以及证据强度是否足够支撑结论。

## 核心能力

- 两阶段输入：标题式输入用于命名和聚焦，详细描述用于说明目标用户、功能、约束和具体问题。
- 交互式启动：默认入口是 `start` 向导，每一轮都会提示必填/选填、输入格式和示例。
- 多 Provider LLM：支持 `openai`、`deepseek`、`google` 三类 OpenAI-compatible 调用，也保留 CLIProxyAPI 的 `setup-token` 模式。
- 并行子 Agent：每个研究路径都会并发运行产业、学术、工程三个子 Agent；每个子 Agent 内部也会并发检索和分析。
- 扩展型搜索规划：围绕产品、竞品、开源仓库、论文、benchmark、社区反馈生成多意图查询，并做去重和限额。
- 决策型报告：报告包含研究问题、方法、决策矩阵、关键 claim、证据和反证、置信度、证据缺口、建议策略。
- 统一命名：日志、session 和报告输出都使用 `日期时间_标题` 命名，例如 `20260504_120000_实时视频翻译工具.md`。
- 非轮转日志：每次运行写入一个独立日志文件，不再按文件大小切分轮转。

## 快速开始

Windows 交互式启动：

```bat
start.bat
```

Linux/macOS 交互式启动：

```bash
chmod +x start.sh
./start.sh
```

也可以显式调用 CLI 的 `start` 命令：

```bash
python -m src start
```

交互式流程会依次询问：

1. 标题式输入（必填）：一句短标题，例如“实时视频翻译工具”。
2. 二阶段细致描述（必填）：可输入多段文字，单独一行 `END` 结束。
3. 关注重点（选填）：逗号分隔，例如“低延迟, 开源实现, 学术评测”。
4. 调研深度（选填）：`quick`、`comprehensive`、`deep`。
5. 最大研究路径数（选填）：1-10 的整数。
6. 输出格式（选填）：`markdown`、`docx`、`both`。

## 非交互命令

如果已经知道完整输入，可以直接运行：

```bash
python -m src research "实时视频翻译工具" --description "面向跨国会议团队，要求实时字幕、语音翻译、低延迟和会议软件集成。" --depth deep --max-paths 3 --format both
```

常用管理命令：

```bash
python -m src list-sessions
python -m src show <session_id>
python -m src status <session_id>
```

## 安装

```bash
conda activate research_tools
pip install -e .
pip install -e ".[dev]"
```

如果需要 DOCX 输出：

```bash
pip install -e ".[docx]"
```

## 环境配置

复制环境变量模板：

```bash
cp .env.example .env
```

LLM 调用由两个变量共同决定：

| 变量 | 可选值 | 说明 |
| --- | --- | --- |
| `LLM_MODE` | `setup-token` / `api-key` | `setup-token` 走 CLIProxyAPI；`api-key` 直连 Provider |
| `LLM_PROVIDER` | `openai` / `deepseek` / `google` | 选择直连 Provider，也用于默认模型选择 |
| `LLM_MODEL` | 任意可用模型名 | `setup-token` 和 OpenAI 直连使用；必须与代理或 Provider 暴露的模型一致 |
| `DEEPSEEK_MODEL` / `GOOGLE_MODEL` | 任意可用模型名 | DeepSeek / Google 直连的可选模型覆盖 |
| `LLM_PROXY_URL` | URL | CLIProxyAPI 地址，默认 `http://localhost:8317` |

Provider Key：

| Provider | 必填 Key | 可选 Base URL |
| --- | --- | --- |
| OpenAI | `OPENAI_API_KEY` | `OPENAI_BASE_URL` |
| DeepSeek | `DEEPSEEK_API_KEY` | `DEEPSEEK_BASE_URL` |
| Google Gemini OpenAI-compatible | `GOOGLE_API_KEY` | `GOOGLE_BASE_URL` |

外部搜索 Key：

| 变量 | 是否必需 | 说明 |
| --- | --- | --- |
| `TAVILY_API_KEY` | 推荐 | 产品、网页论文线索、开源实现和社区资料搜索 |

不要把真实密钥写入 `config/default.yaml` 或文档。密钥只放在 `.env` 或系统环境变量中。

## 调研流水线

```text
标题 + 详细描述
  |
  v
IdeaDecomposer
  |
  v
ResearchPlanner
  |
  +--> Path 1: IndustryResearcher || AcademicResearcher || EngineeringAnalyst
  +--> Path 2: IndustryResearcher || AcademicResearcher || EngineeringAnalyst
  +--> Path N: IndustryResearcher || AcademicResearcher || EngineeringAnalyst
  |
  v
ReputationScorer + MaturityMapper
  |
  v
ReportGenerator
  |
  +--> MarkdownReporter
  +--> DocxReporter
```

当前不会引入 LangGraph、CrewAI、向量数据库或 embedding。项目保持 CLI、本地文件存储和自定义 Python Agent 类。

## 目录结构

```text
src/
  cli.py                  # Typer CLI 和 start 交互式向导
  orchestrator.py         # 异步流水线协调器
  logging_config.py       # 非轮转日志配置
  llm/client.py           # OpenAI-compatible/CLIProxyAPI LLM 客户端
  agents/                 # 调研子 Agent
  apis/                   # Tavily、arXiv、Web scraper，以及历史兼容客户端
  models/                 # Pydantic v2 数据模型
  storage/local_store.py  # 本地 JSON 会话存储
  reporters/              # Markdown/DOCX 输出
  utils/                  # 命名、JSON 修复、文本处理、重试工具
config/default.yaml       # 非密钥运行配置
docs/research/            # 当前调研和设计依据
docs/archive/             # 过时说明和历史调研资料
tests/                    # pytest 测试
```

## 输出和日志

默认输出到 `output/`，日志写入 `logs/`。二者都使用同一个 run name：`YYYYMMDD_HHMMSS_标题`。

示例：

```text
output/20260504_120000_实时视频翻译工具.md
output/20260504_120000_实时视频翻译工具.docx
logs/20260504_120000_实时视频翻译工具.log
data/research/20260504_120000_实时视频翻译工具/
```

日志文件不会按大小轮转；一次运行对应一个日志文件。

## 报告结构

报告包括 Executive Summary、Research Question and Method、Decision Matrix、Key Claims and Evidence、Technology Landscape、Maturity Map、Implementation Workflows、行业/学术/工程发现、Reputation、Feasibility、Confidence/Gaps/Assumptions、Recommendations 和 Sources。

## 测试

Windows 上推荐直接使用环境内 Python，避免 `conda run` 的控制台编码噪声：

```powershell
$env:PYTHONUTF8='1'
$env:PYTHONIOENCODING='utf-8'
C:\Path\To\anaconda3\envs\research_tools\python.exe -m pytest tests/ -v
```

常规命令：

```bash
conda run -n research_tools python -m pytest tests/ -v
conda run -n research_tools python -m pytest tests/ -v --cov=src --cov-report=term-missing
```

## License

MIT

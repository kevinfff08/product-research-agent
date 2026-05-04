# 产品技术调研 Agent

这是一个本地优先的命令行研究工具。它接收一个模糊的产品想法，自动拆解研究路径，并汇总产业产品、学术论文、开源实现、工程可行性、技术成熟度和来源信誉，最后生成 Markdown 或 DOCX 报告。

## 适用场景

- 判断一个产品想法是否已有成熟方案。
- 梳理某个技术方向的产业产品、论文和开源实现。
- 为原型开发选择技术路线、依赖栈和风险控制点。
- 生成可继续人工精修的技术 landscape 报告。

## 当前能力

- 想法拆解：把一句产品描述拆成多个可检索研究路径。
- 搜索规划：为 Web、GitHub、Semantic Scholar 和 arXiv 生成查询。
- 并行研究：同时运行产业、学术和工程分析。
- 信誉评分：综合公司、实验室、论文场所和社区信号。
- 成熟度映射：按 `early_prototype`、`development`、`mature`、`cutting_edge`、`academic_frontier` 归类。
- 报告输出：支持 `markdown`、`docx`、`both` 三种格式。

## 快速开始

Windows：

```bat
start.bat research "实时视频翻译工具" --depth comprehensive --format both
```

Linux/macOS：

```bash
chmod +x start.sh
./start.sh research "实时视频翻译工具" --depth comprehensive --format both
```

也可以直接运行 Python CLI：

```bash
python -m src research "AI 代码审查工具"
python -m src research "实时视频翻译工具" --depth deep --max-paths 3
python -m src list-sessions
python -m src show <session_id>
python -m src status <session_id>
```

## 环境要求

- Python 3.14+，推荐使用 conda 环境 `research_tools`。
- Codex 可用的 OpenAI-compatible LLM 通道。
- Tavily API key，用于产业和 Web 搜索。

可选：

- Semantic Scholar API key：提高学术搜索限额。
- GitHub token：提高 GitHub API 限额。
- `python-docx`：生成 DOCX 报告时需要。

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

## 配置

复制环境变量模板：

```bash
cp .env.example .env
```

常用变量：

| 变量 | 是否必需 | 说明 |
| --- | --- | --- |
| `LLM_MODE` | 是 | `setup-token` 或 `api-key` |
| `LLM_MODEL` | 否 | Codex 模型名，默认 `gpt-5.4` |
| `LLM_PROXY_URL` | setup-token 模式 | CLIProxyAPI 地址，默认 `http://localhost:8317` |
| `OPENAI_API_KEY` | api-key 模式 | 直连 OpenAI-compatible API 时使用 |
| `OPENAI_BASE_URL` | 否 | 自定义 OpenAI-compatible API 地址 |
| `TAVILY_API_KEY` | 是 | Tavily 搜索 API key |
| `SEMANTIC_SCHOLAR_API_KEY` | 否 | Semantic Scholar API key |
| `GITHUB_TOKEN` | 否 | GitHub personal access token |

研究参数在 `config/default.yaml` 中维护，包括最大路径数、API 限额、LLM token 数和温度设置。不要把密钥写入 YAML；密钥只放在 `.env` 或系统环境变量里。

## 架构

```text
用户输入
  |
  v
IdeaDecomposer
  |
  v
ResearchPlanner
  |
  +--> IndustryResearcher
  +--> AcademicResearcher
  +--> EngineeringAnalyst
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

核心目录：

```text
src/
  cli.py                  # Typer CLI
  orchestrator.py         # 异步流水线协调器
  llm/client.py           # Codex/OpenAI-compatible LLM 客户端
  agents/                 # 各研究 Agent
  apis/                   # 外部 API 客户端
  models/                 # Pydantic v2 模型
  storage/local_store.py  # 本地 JSON 会话存储
  reporters/              # Markdown/DOCX 输出
  utils/                  # JSON 修复、文本处理、重试工具
config/
  default.yaml            # 非密钥运行配置
docs/archive/             # 过时说明和历史调研资料
tests/                    # pytest 测试
```

## 输出

默认输出到 `output/`：

- `--format markdown`：生成 `output/<session_id>.md`
- `--format docx`：生成 `output/<session_id>.docx`
- `--format both`：同时生成两种格式

运行数据保存在 `data/`，日志保存在 `logs/`。这三个目录都是运行产物，不进入 Git。

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

## 维护规则

- LLM 只允许 Codex/OpenAI-compatible 路径，不引入其他 LLM SDK。
- `render_template()` 只渲染 prompt，不发起 LLM 调用。
- 结构化 LLM 输出必须走 `generate_json()`，并用 Pydantic 模型兜底验证。
- 新功能必须配对应测试。
- 根目录只保留活跃入口文件；历史说明和调研资料放入 `docs/archive/`。
- `.env.example` 只放占位符，不能写入真实密钥或本机绝对路径。

## 历史资料

旧的代理说明、早期架构调研文档和本地助手设置已归档到 `docs/archive/legacy-2026-05-04/`。这些文件仅作历史参考，不再代表当前运行方式。

## License

MIT

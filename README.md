# Career Readiness Agent｜求职准备智能体

一个本地优先、以真实证据为约束的求职准备 Agent。它不是只给出泛化建议的聊天壳：用户可以从简历和项目资料中建立可确认的能力档案，录入多个目标岗位 JD，由 Agent 分析岗位要求与证据缺口，生成简历优化建议和下一步准备任务。

项目提供两条独立入口：普通聊天围绕本地资料库问答并展示来源；求职 Agent 按“隐私设置 → 能力档案 → 岗位机会 → 准备任务 → 求职建议”递进，且可随时返回上一步修改。

## 核心能力

| 能力 | 当前实现 |
| --- | --- |
| 能力档案 | 上传 DOCX/PDF/TXT/MD 简历，自动提取为待确认草稿；支持项目、实习/工作、竞赛、课程与实践、校园经历的多条记录。 |
| 结构化提取 | 用户同意后调用兼容 OpenAI API 的模型提取简历；模型不可用时自动降级为本地规则解析。 |
| 岗位 JD 分析 | 粘贴 JD 后创建岗位草稿；确认岗位时自动建立要求矩阵，标记已匹配、证据不足或能力缺口。 |
| 多岗位规划 | 最多维护 3 个活跃岗位，可选主岗位；任务按缺口、复用价值、时间可行性与截止日期排序。 |
| 简历优化 | 只依据已确认经历和关联岗位要求生成可编辑建议，不写入未证实信息。 |
| 本地 RAG 与记忆 | Chroma 返回来源文件、路径和切片；`thread_id` + SQLite 保存会话上下文。 |
| 联网研究与隐私 | 本地处理、模型分析、联网研究分别授权且可撤回；研究只发送公开查询关键词，并标注来源等级。 |

## 架构

```text
Streamlit Web UI
  ├─ 普通聊天 ────────────────► FastAPI /chat ─► Agent + RAG + SQLite memory
  └─ 求职 Agent
       ├─ 能力档案 ───────────► Resume parser / structured model extraction
       ├─ 岗位机会 ───────────► JD analysis + evidence assessment
       ├─ 准备任务 ───────────► Multi-job priority planning
       └─ 求职建议 ───────────► Evidence-constrained resume suggestions

FastAPI ─► SQLite（档案、岗位、任务、授权、会话）
        └► Chroma（本地文档索引，带来源切片）
```

FastAPI 的响应模型约束 UI 输出字段；模型/API 异常会转为面向用户的友好提示，而不是直接泄露内部异常。

## 快速运行

环境要求：Python 3.11+。以下命令以 Windows PowerShell 为例。

```powershell
conda activate career-agent
pip install -e ".[dev]"
```

复制 `.env.example` 为 `.env`，填入 OpenAI 或任意 OpenAI 兼容服务的 Key、模型名和（如需）兼容 API 地址：

```powershell
Copy-Item .env.example .env
```

在第一个终端启动后端：

```powershell
conda activate career-agent
python -m uvicorn career_agent.api:app --reload
```

API 文档地址：[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

在第二个终端启动 Web UI：

```powershell
conda activate career-agent
python -m streamlit run streamlit_app.py
```

终端通常显示 [http://localhost:8501](http://localhost:8501)；端口冲突时以实际输出为准。未配置模型时，本地资料与档案流程仍可查看，模型功能会返回明确提示而不会伪造结果。

## 项目结构

```text
src/career_agent/   # API、Agent、档案、岗位、规划、RAG、研究与简历建议
streamlit_app.py    # 普通聊天 + 递进式求职工作台
scripts/            # CLI 演示和测试脚本
tests/              # API、RAG、档案、岗位、规划与 UI 自动化测试
```

## 数据与隐私边界

- `data/` 保存本地数据库、向量索引和上传文件，已被 `.gitignore` 排除，**不会上传 GitHub**。
- `.env` 包含 API Key，已忽略；仅提交 `.env.example`。
- 模型分析、联网研究均须在界面中单独授权；联网请求不携带个人简历或能力档案原文。

## 推荐体验流程

1. 打开 Web UI，选择“开始求职准备”，按需开启本地处理、模型分析和联网研究授权。
2. 上传简历，检查并确认自动提取的多条经历草稿；也可从空白档案开始。
3. 粘贴一个或多个岗位 JD，确认岗位后自动生成岗位要求与证据缺口。
4. 选择 1–3 个活跃岗位，设置主岗位（可选），创建或查看准备任务推荐。
5. 在“职业报告”查看证据、岗位要求矩阵、下一步任务和简历优化建议。

## API 概览

| 路径 | 作用 |
| --- | --- |
| `GET /health` | 检查后端状态。 |
| `POST /chat` | 返回 `thread_id`、回答和来源引用。 |
| `POST /profiles/resume` | 上传简历并生成待确认能力档案草稿。 |
| `POST /jobs/jd` | 从 JD 文本创建岗位草稿。 |
| `POST /jobs/{id}/analyze` | 基于能力档案自动分析岗位要求。 |
| `PUT /planning/active-opportunities` | 保存 0–3 个活跃岗位；空列表可清空选择。 |
| `GET /reports/career` | 生成证据可追溯的职业报告和下一步任务。 |
| `POST /research` | 在联网研究授权后检索公开资料并分级返回来源。 |

完整请求示例请访问 `/docs`。

## 测试

```powershell
conda activate career-agent
python -m pytest -q
```

如遇 Windows 上 pytest 默认临时目录权限问题，可使用基础端到端测试脚本：

```powershell
.\scripts\run_basic_test.ps1 -Python D:\Conda\envs\career-agent\python.exe
```

当前版本的全量验证结果为：**71 passed**。

## 技术栈

Python · FastAPI · Streamlit · SQLite · LangChain · Chroma · Pydantic · OpenAI 兼容模型 API · pytest

## 简历项目条目

可直接复用的中文简历描述见：[docs/resume/career-readiness-agent.md](docs/resume/career-readiness-agent.md)。

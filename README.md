# Career Knowledge Agent

一个可以写进简历、可在面试中拆解讲解的端到端 Agent 项目：它能回答个人简历/项目文档问题，并在需要时自主调用**知识库检索、当前时间、计算器**工具。

## 它解决什么问题

面试准备时，候选人的项目材料通常散落在简历、复盘和技术文档中。该 Agent 将这些资料建立为本地向量索引，让回答有依据、可追溯；同时通过工具调用处理“现在几点”“指标如何计算”等不应靠模型猜测的问题。

## 架构

```text
用户问题
  ↓
LangChain Agent（决策：需要资料、时间还是计算？）
  ├─ search_knowledge_base → Chroma → 文档切片 → 返回带来源的内容
  ├─ get_current_time → 系统时钟
  └─ calculator → AST 白名单计算器
  ↓
LLM 基于工具结果生成答案（不确定时拒答）
```

这就是 Agent 的“感知—决策—执行”闭环：输入和工具结果是感知，LLM 选择工具是决策，调用工具并根据结果回答是执行。

## RAG、工具调用、MCP：一句话区分

| 概念 | 解决的问题 | 在本项目中的对应物 |
| --- | --- | --- |
| RAG | 模型不知道私有/新资料时，先检索再回答 | 文档切片 + 哈希向量 + Chroma 检索 |
| Tool Use | 让模型调用确定性能力或外部 API | `calculator`、`get_current_time`、检索器 |
| MCP | 让不同 Agent 客户端用统一协议发现和调用工具 | 下一阶段可将这些工具封装为 MCP Server |

RAG 不等于“联网”：它是把**指定外部资料**在回答前送入上下文；联网搜索只是可被 Agent 调用的一种工具。

## 快速运行

环境要求：Python 3.11+。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
python scripts/ingest.py
python scripts/retrieval_demo.py
```

至此已经完整验证“准备文档 → 切片 → 向量索引 → 检索”。`HashEmbeddings` 是为了第一天离线跑通流程的教学实现，并不具备生产级语义检索能力。

启用真正的 Agent：复制 `.env.example` 为 `.env`，填入 OpenAI 或任意 OpenAI 兼容服务的 Key 与模型名，然后运行：

```powershell
python scripts/chat.py
```

建议演示提问：

```text
订单洞察平台使用了哪些技术？请给出来源。
把订单日报优化前后的耗时差换算成分钟。
现在上海几点？
```

## 项目结构

```text
data/documents/     # 可替换为你的简历、项目复盘（.md / .txt）
src/career_agent/   # RAG、工具与 Agent 编排
scripts/            # 建索引、离线检索、交互式聊天入口
tests/              # 工具安全性测试
```

## 面试讲解（90 秒）

“我实现了一个个人知识库问答 Agent。离线阶段会读取 Markdown 和文本资料，按 500 字符、80 字符重叠切片，生成 embedding 后存入 Chroma。提问时，Agent 不直接凭记忆回答简历信息，而是先把问题交给检索工具；工具返回原文和来源，模型只依据这些上下文生成答案。除 RAG 外，我还把时间和计算器封装为带 JSON Schema 的工具，模型自行判断何时调用。计算器没有执行任意代码，而是用 AST 白名单限制表达式。下一步我会接入语义 embedding、网页搜索和持久化对话记忆，并用评测集衡量检索命中率和答案正确性。”

## 迭代路线

1. **第 1 周**：替换自己的资料，跑通索引和检索；理解切片大小、重叠和 Top-K 对召回的影响。
2. **第 2 周**：配置模型，观察 Agent 的工具调用轨迹；新增一个天气或网页搜索工具。
3. **第 3 周**：用 OpenAI Embeddings 或 BGE 替换教学用哈希向量；增加会话记忆、来源引用和 10 条评测问题。
4. **第 4 周**：接入 MCP Server / Streamlit 页面，补充截图或 1 分钟 Demo，推送 GitHub。

## 学习资源

- [LangChain 官方：RAG、Agent 与教程入口](https://docs.langchain.com/oss/python/langchain/quickstart)
- [LangChain 官方：Tools 与工具调用](https://docs.langchain.com/oss/python/langchain/tools)
- [LangChain 官方：Agents 架构与调试](https://docs.langchain.com/oss/python/langchain/agents)
- [LangChain 官方：OpenAI 工具调用集成](https://docs.langchain.com/oss/python/integrations/chat/openai)
- [RAG 与 MCP 的关系说明](https://www.digitalocean.com/community/tutorials/engineers-guide-rag-vs-mcp-llms)

你提到的 `rag-tool-agent-course` 仓库名称无法在公开搜索中稳定定位；建议在 GitHub 用该关键词筛选近期活跃、含 README、许可证与可复现环境的仓库，再将其中一个实现作为对照，而不是直接照抄。

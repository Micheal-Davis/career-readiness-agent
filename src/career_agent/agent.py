"""把 RAG 检索器与工具注册为 LangChain Agent。"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from .memory import create_sqlite_checkpointer
from .rag import get_retriever
from .tools import make_tools

SYSTEM_PROMPT = """你是候选人的个人知识库助手。
回答经历、技能或项目事实前，必须调用 search_knowledge_base；只基于工具返回内容作答，并在回答末尾列出来源文件。
时间与计算问题优先调用对应工具。若资料不足，明确说“知识库中没有足够信息”，不要编造。"""


def create_career_agent(index_dir: Path, memory_path: Path):
    """Create an agent whose conversation state persists by thread ID."""
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("缺少 OPENAI_API_KEY。请复制 .env.example 为 .env 并填写 Key；可先运行 retrieval_demo.py 验证离线 RAG。")
    model_args = {"model": os.getenv("OPENAI_MODEL", "gpt-4.1-mini"), "temperature": 0}
    if base_url := os.getenv("OPENAI_BASE_URL"):
        model_args["base_url"] = base_url
    model = ChatOpenAI(**model_args)
    return create_agent(
        model=model,
        tools=make_tools(get_retriever(index_dir)),
        system_prompt=SYSTEM_PROMPT,
        checkpointer=create_sqlite_checkpointer(memory_path),
    )

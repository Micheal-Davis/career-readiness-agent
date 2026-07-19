"""不依赖 API Key，验证文档切片、向量写入和检索。"""
from pathlib import Path

from career_agent.rag import get_retriever

ROOT = Path(__file__).resolve().parents[1]
question = input("请输入问题：").strip()
for index, doc in enumerate(get_retriever(ROOT / "data" / "chroma").invoke(question), start=1):
    print(f"\n--- 命中 {index}｜{doc.metadata.get('source')} ---\n{doc.page_content}")


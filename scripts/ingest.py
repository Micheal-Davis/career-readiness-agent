from pathlib import Path

from career_agent.rag import build_index

ROOT = Path(__file__).resolve().parents[1]
count = build_index(ROOT / "data" / "documents", ROOT / "data" / "chroma")
print(f"索引完成：{count} 个文档切片已写入 data/chroma")


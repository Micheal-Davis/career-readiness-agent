"""可离线运行的确定性向量模型，仅用于学习与冒烟验证。

真实项目请使用 OpenAIEmbedding 或 BGE 等语义 embedding 模型。这里保留它，
让项目在没有下载大模型、没有 API Key 的第一天也能跑完整个索引流程。
"""
from __future__ import annotations

import hashlib
import math
import re

from langchain_core.embeddings import Embeddings


class HashEmbeddings(Embeddings):
    """基于词与中文 n-gram 的哈希向量，不具备真实语义理解能力。"""

    def __init__(self, dimensions: int = 384) -> None:
        self.dimensions = dimensions

    @staticmethod
    def _tokens(text: str) -> list[str]:
        ascii_tokens = re.findall(r"[A-Za-z0-9_]+", text.lower())
        chinese = "".join(re.findall(r"[\u4e00-\u9fff]", text))
        grams = [chinese[i : i + width] for width in (1, 2, 3) for i in range(max(0, len(chinese) - width + 1))]
        return ascii_tokens + grams

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in self._tokens(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "big")
            vector[value % self.dimensions] += 1.0 if value & 1 else -1.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


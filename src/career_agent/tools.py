"""Agent 可调用的受限工具。"""
from __future__ import annotations

import ast
import operator
from datetime import datetime
from zoneinfo import ZoneInfo

from langchain.tools import tool

_ALLOWED_OPERATORS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg,
}


def _calculate(expression: str) -> float:
    def visit(node: ast.AST) -> float:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPERATORS:
            return _ALLOWED_OPERATORS[type(node.op)](visit(node.left), visit(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPERATORS:
            return _ALLOWED_OPERATORS[type(node.op)](visit(node.operand))
        raise ValueError("只允许数字与 + - * / ** 括号")
    return visit(ast.parse(expression, mode="eval").body)


def make_tools(retriever):
    @tool
    def search_knowledge_base(query: str) -> str:
        """检索个人简历和项目资料。凡是涉及候选人经历、技能、项目数据的问题都必须先调用此工具。"""
        docs = retriever.invoke(query)
        if not docs:
            return "知识库中没有找到相关资料。"
        return "\n\n".join(
            "[来源：{source}｜路径：{path}｜切片：{chunk}]\n{content}".format(
                source=doc.metadata.get("source", "未知"),
                path=doc.metadata.get("source_path", "未知"),
                chunk=doc.metadata.get("chunk_index", "未知"),
                content=doc.page_content,
            )
            for doc in docs
        )

    @tool
    def get_current_time(timezone: str = "Asia/Shanghai") -> str:
        """查询指定 IANA 时区的当前时间，例如 Asia/Shanghai 或 America/New_York。"""
        return datetime.now(ZoneInfo(timezone)).isoformat(timespec="seconds")

    @tool
    def calculator(expression: str) -> str:
        """计算纯数学表达式，例如 (1024 / 8) + 7。不能处理变量、函数或单位。"""
        try:
            return str(_calculate(expression))
        except (SyntaxError, ValueError, ZeroDivisionError) as error:
            return f"计算失败：{error}"

    return [search_knowledge_base, get_current_time, calculator]

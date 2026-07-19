from langchain_core.documents import Document

from career_agent.tools import _calculate, make_tools


def test_calculator_allows_basic_math():
    assert _calculate("(1024 / 8) + 7") == 135


def test_calculator_blocks_code():
    try:
        _calculate("__import__('os').system('echo unsafe')")
    except ValueError:
        pass
    else:
        raise AssertionError("unsafe expression was accepted")


def test_search_tool_includes_traceable_source_metadata():
    class Retriever:
        def invoke(self, query):
            return [
                Document(
                    page_content="构建了知识库项目。",
                    metadata={
                        "source": "resume.md",
                        "source_path": "career/resume.md",
                        "chunk_index": 2,
                    },
                )
            ]

    tools = make_tools(Retriever())
    search_tool = next(tool for tool in tools if tool.name == "search_knowledge_base")

    assert search_tool.invoke({"query": "项目经验"}) == (
        "[来源：resume.md｜路径：career/resume.md｜切片：2]\n"
        "构建了知识库项目。"
    )

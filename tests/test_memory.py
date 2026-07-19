from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from career_agent.memory import create_sqlite_checkpointer


def test_same_thread_restores_prior_messages_from_sqlite(tmp_path):
    checkpointer = create_sqlite_checkpointer(tmp_path / "conversations.sqlite3")
    agent = create_agent(
        FakeMessagesListChatModel(
            responses=[AIMessage(content="第一轮回答"), AIMessage(content="第二轮回答")]
        ),
        checkpointer=checkpointer,
    )
    config = {"configurable": {"thread_id": "career-session"}}

    agent.invoke({"messages": [{"role": "user", "content": "介绍项目"}]}, config)
    agent.invoke({"messages": [{"role": "user", "content": "展开刚才的项目"}]}, config)

    messages = agent.get_state(config).values["messages"]
    contents = [message.content for message in messages]

    assert contents == ["介绍项目", "第一轮回答", "展开刚才的项目", "第二轮回答"]


def test_different_threads_do_not_share_messages(tmp_path):
    checkpointer = create_sqlite_checkpointer(tmp_path / "conversations.sqlite3")
    agent = create_agent(
        FakeMessagesListChatModel(responses=[AIMessage(content="回答")]),
        checkpointer=checkpointer,
    )
    first_thread = {"configurable": {"thread_id": "first"}}
    second_thread = {"configurable": {"thread_id": "second"}}

    agent.invoke({"messages": [{"role": "user", "content": "第一段对话"}]}, first_thread)

    assert agent.get_state(second_thread).values == {}

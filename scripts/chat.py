import argparse
from pathlib import Path
from uuid import uuid4

from career_agent.agent import create_career_agent
from career_agent.errors import friendly_error_message

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 Career Knowledge Agent")
    parser.add_argument(
        "--thread-id",
        help="复用已有会话；省略时会创建一个新会话。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    thread_id = args.thread_id or str(uuid4())

    try:
        agent = create_career_agent(
            ROOT / "data" / "chroma",
            ROOT / "data" / "conversations.sqlite3",
        )
    except Exception as error:
        print(f"启动失败：{friendly_error_message(error)}")
        return

    print("Career Knowledge Agent 已启动，输入 exit 退出。")
    print(f"会话 ID：{thread_id}")
    print(f"重启后使用 --thread-id {thread_id} 继续此会话。")
    while True:
        question = input("\n你：").strip()
        if question.lower() in {"exit", "quit"}:
            break

        try:
            result = agent.invoke(
                {"messages": [{"role": "user", "content": question}]},
                config={"configurable": {"thread_id": thread_id}},
            )
        except Exception as error:
            print(f"\n请求失败：{friendly_error_message(error)}")
            continue

        print("\nAgent：", result["messages"][-1].content)


if __name__ == "__main__":
    main()

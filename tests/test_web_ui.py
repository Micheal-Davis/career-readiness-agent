import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]


def _click_button(app: AppTest, label: str) -> None:
    next(button for button in app.button if button.label == label).click().run()


def test_streamlit_app_starts_with_two_clear_product_entries():
    app = AppTest.from_file(str(ROOT / "streamlit_app.py"))
    app.run()

    assert not app.exception
    assert app.title[0].value == "Career Knowledge Agent"
    assert {button.label for button in app.button} >= {"进入普通聊天", "开始求职准备"}


def test_streamlit_app_enters_agent_at_the_first_guided_step():
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            response = json.dumps(
                {"local_processing": False, "model_analysis": False, "web_research": False}
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

        def log_message(self, format, *args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    Thread(target=server.serve_forever, daemon=True).start()
    app = AppTest.from_file(str(ROOT / "streamlit_app.py"))
    app.session_state["api_url"] = f"http://127.0.0.1:{server.server_port}"

    try:
        app.run()
        _click_button(app, "开始求职准备")

        assert not app.exception
        assert app.session_state["workspace_mode"] == "agent"
        assert app.session_state["agent_step"] == 0
        assert app.caption[0].value == "求职 Agent · 第 1 / 5 步"
    finally:
        server.shutdown()


def test_streamlit_app_accepts_backend_generated_thread_id():
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            response = json.dumps(
                {
                    "thread_id": "generated-thread",
                    "answer": "人才库中有一名候选人。",
                    "sources": [],
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

        def log_message(self, format, *args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    Thread(target=server.serve_forever, daemon=True).start()
    app = AppTest.from_file(str(ROOT / "streamlit_app.py"))
    app.session_state["api_url"] = f"http://127.0.0.1:{server.server_port}"

    try:
        app.run()
        _click_button(app, "进入普通聊天")
        app.chat_input[0].set_value("介绍人才库").run()

        assert not app.exception
        assert app.session_state["thread_id"] == "generated-thread"
    finally:
        server.shutdown()

"""可独立运行的 Career Readiness Agent 基础端到端验收测试。

运行方式：
    python -m pytest tests/test_basic_e2e.py -q

测试不调用真实大模型或互联网：所有外部依赖都由 FakeAgent 和 fake_web_search 替代。
"""
from __future__ import annotations

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from career_agent.api import create_app
from career_agent.research import ConsentScope, ConsentStore


class FakeAgent:
    """替代真实 LLM，记录请求内容，保证测试无需 API Key。"""

    def __init__(self) -> None:
        self.payloads: list[dict] = []

    def invoke(self, payload: dict, config: dict) -> dict:
        self.payloads.append(payload)
        return {"messages": [AIMessage(content="这是离线测试模型的回答。")]}


def test_basic_user_journey(tmp_path) -> None:
    """验证一位用户从建档到岗位准备、报告和聊天的最小完整流程。"""
    # 1. 创建临时且隔离的数据目录；任何真实用户数据都不会被读取或改动。
    consents = ConsentStore(tmp_path / "consents.sqlite3")
    fake_agent = FakeAgent()
    client = TestClient(
        create_app(
            tmp_path,
            lambda *_: fake_agent,
            consents,
            web_search=lambda _: [
                {
                    "title": "示例企业招聘页",
                    "url": "https://careers.example.com/backend",
                    "summary": "公开岗位说明。",
                }
            ],
        )
    )

    # 2. 健康检查：确认 API 服务已创建，尚未依赖模型服务。
    assert client.get("/health").json() == {"status": "ok"}

    # 3. 明确授权：模型分析和联网研究是两项独立授权。
    assert client.put("/consents/model_analysis", json={"granted": True}).status_code == 200
    assert client.put("/consents/web_research", json={"granted": True}).status_code == 200

    # 4. 创建空白能力档案，再增加并确认一条有完整证据的项目经历。
    profile_id = client.post("/profiles/empty").json()["profile_id"]
    draft = client.post("/profiles/resume", files={
        "file": ("resume.txt", "项目经历\n职业 Agent\n实现检索服务", "text/plain")
    }).json()["drafts"][0]
    # 这里的简历草稿属于另一份档案；以下直接使用当前档案创建可确认经历。
    evidence = client.put(
        f"/evidence/{draft['id']}",
        json={
            "title": "职业 Agent",
            "evidence_type": "project",
            "details": {
                "technologies": "Python、FastAPI、SQLite",
                "contribution": "实现岗位分析和检索接口",
                "result": "完成可本地运行的原型",
            },
        },
    )
    # 使用简历提取流程生成的证据所属档案，以下读取其真实 profile_id，避免跨档案写入。
    evidence_profile_id = evidence.json()["profile_id"]
    assert client.post(f"/evidence/{draft['id']}/confirm").status_code == 200

    # 5. 导入岗位 JD，建立一项有证据支持的岗位要求。
    job = client.post("/jobs/jd", json={"jd_text": "后端开发工程师"}).json()[0]
    requirement = client.post(
        f"/jobs/{job['id']}/requirements",
        json={
            "text": "Python 服务端开发",
            "category": "foundational_capability",
            "priority": "core",
            "depth": "independent",
            "origin": "explicit",
            "evidence_assessment": "supported",
            "supporting_evidence_ids": [draft["id"]],
        },
    ).json()

    # 6. 搜索公开资料并保存到这个岗位；假搜索保证测试不访问互联网。
    research = client.post(
        "/research",
        json={
            "query": "后端开发工程师 招聘要求",
            "first_party_domains": ["example.com"],
            "job_opportunity_id": job["id"],
        },
    )
    assert research.status_code == 200
    assert research.json()[0]["tier"] == "first_party"
    assert client.get(f"/jobs/{job['id']}/research-sources").json() == research.json()

    # 7. 选择活跃岗位与主岗位，再创建一个可验证的准备任务。
    assert client.put("/planning/active-opportunities", json={"opportunity_ids": [job["id"]]}).status_code == 200
    assert client.put("/planning/primary-opportunity", json={"opportunity_id": job["id"]}).status_code == 200
    task = client.post(
        "/planning/tasks",
        json={
            "title": "完善职业 Agent 项目说明",
            "description": "补充 API 文档、测试与演示步骤",
            "opportunity_ids": [job["id"]],
            "requirement_ids": [requirement["id"]],
            "estimated_hours": 3,
        },
    ).json()

    # 8. 职业报告应能汇总确认的证据、活跃岗位和下一步任务。
    report = client.get(
        "/reports/career", params={"profile_id": evidence_profile_id, "weekly_hours": 8}
    )
    assert report.status_code == 200
    assert report.json()["confirmed_evidence_count"] == 1
    assert report.json()["active_jobs"][0]["id"] == job["id"]
    assert report.json()["next_task"]["id"] == task["id"]

    # 9. 简历建议必须可回溯到原始证据和匹配的岗位要求。
    resume = client.get(
        "/resume/optimization",
        params={"profile_id": evidence_profile_id, "opportunity_id": job["id"]},
    )
    assert resume.status_code == 200
    assert resume.json()["suggestions"][0]["evidence_id"] == draft["id"]
    assert requirement["id"] in resume.json()["suggestions"][0]["supporting_requirement_ids"]

    # 10. 聊天带入本地职业上下文；FakeAgent 能证明上下文真正传给了模型层。
    chat = client.post(
        "/chat", json={"message": "我该先做什么？", "profile_id": evidence_profile_id}
    )
    assert chat.status_code == 200
    assert chat.json()["answer"] == "这是离线测试模型的回答。"
    assert "[职业工作台上下文]" in fake_agent.payloads[0]["messages"][0]["content"]

    consents.close()

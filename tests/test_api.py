from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, ToolMessage

from career_agent.api import create_app
from career_agent.research import ConsentScope, ConsentStore


class FakeAgent:
    def __init__(self, result=None, error: Exception | None = None):
        self.result = result
        self.error = error
        self.configs = []
        self.payloads = []

    def invoke(self, payload, config):
        self.payloads.append(payload)
        self.configs.append(config)
        if self.error:
            raise self.error
        return self.result


def test_health_does_not_initialize_agent(tmp_path):
    def agent_factory(index_dir, memory_path):
        raise AssertionError("health should not create the agent")

    client = TestClient(create_app(tmp_path, agent_factory))

    assert client.get("/health").json() == {"status": "ok"}


def test_chat_returns_thread_and_structured_sources(tmp_path):
    agent = FakeAgent(
        {
            "messages": [
                ToolMessage(
                    name="search_knowledge_base",
                    tool_call_id="search-1",
                    content=(
                        "[来源：resume.md｜路径：career/resume.md｜切片：2]\n"
                        "构建了知识库项目。"
                    ),
                ),
                AIMessage(content="候选人构建了知识库项目。"),
            ]
        }
    )
    consents = ConsentStore(tmp_path / "consents.sqlite3")
    consents.set(ConsentScope.MODEL_ANALYSIS, granted=True)
    client = TestClient(create_app(tmp_path, lambda *_: agent, consents))

    response = client.post(
        "/chat",
        json={"message": "介绍项目", "thread_id": "career-1"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "thread_id": "career-1",
        "answer": "候选人构建了知识库项目。",
        "sources": [
            {
                "source": "resume.md",
                "source_path": "career/resume.md",
                "chunk_index": 2,
            }
        ],
    }
    assert agent.configs == [{"configurable": {"thread_id": "career-1"}}]
    consents.close()


def test_chat_converts_agent_errors_to_friendly_api_error(tmp_path):
    agent = FakeAgent(error=ConnectionError())
    consents = ConsentStore(tmp_path / "consents.sqlite3")
    consents.set(ConsentScope.MODEL_ANALYSIS, granted=True)
    client = TestClient(create_app(tmp_path, lambda *_: agent, consents))

    response = client.post("/chat", json={"message": "介绍项目"})

    assert response.status_code == 503
    assert response.json() == {
        "detail": "无法连接模型服务，请检查网络和 OPENAI_BASE_URL。"
    }
    consents.close()


def test_chat_refuses_model_analysis_without_consent(tmp_path):
    client = TestClient(create_app(tmp_path, lambda *_: FakeAgent()))

    response = client.post("/chat", json={"message": "介绍项目"})

    assert response.status_code == 403
    assert "模型分析" in response.json()["detail"]


def test_profile_initialization_and_draft_confirmation_over_api(tmp_path):
    client = TestClient(create_app(tmp_path, lambda *_: FakeAgent()))

    empty_response = client.post("/profiles/empty")
    resume_response = client.post(
        "/profiles/resume",
        files={
            "file": (
                "resume.txt",
                "项目经历\n职业 Agent\n实现检索服务",
                "text/plain",
            )
        },
    )

    assert empty_response.status_code == 200
    assert empty_response.json()["drafts"] == []
    assert resume_response.status_code == 200
    profile = resume_response.json()
    draft = profile["drafts"][0]
    assert draft["confirmation_status"] == "draft"

    updated = client.put(
        f"/evidence/{draft['id']}",
        json={
            "title": "职业 Agent",
            "evidence_type": "project",
            "details": {
                "technologies": "Python, FastAPI",
                "contribution": "实现检索接口",
                "result": "完成可演示原型",
            },
        },
    )
    confirmed = client.post(f"/evidence/{draft['id']}/confirm")

    assert updated.status_code == 200
    assert confirmed.status_code == 200
    assert confirmed.json()["confirmation_status"] == "confirmed"
    assert client.get(f"/profiles/{profile['profile_id']}/evidence/drafts").json() == []


def test_jd_import_and_job_confirmation_over_api(tmp_path):
    client = TestClient(create_app(tmp_path, lambda *_: FakeAgent()))

    imported = client.post(
        "/jobs/jd",
        json={"jd_text": "招聘方向：服务端工程师\n负责 API\n招聘方向：Agent 后端工程师\n负责执行环境"},
    )

    assert imported.status_code == 200
    drafts = imported.json()
    assert [item["title"] for item in drafts] == ["服务端工程师", "Agent 后端工程师"]
    confirmed = client.post(f"/jobs/{drafts[0]['id']}/confirm")
    assert confirmed.json()["confirmation_status"] == "confirmed"


def test_job_requirement_matrix_over_api_labels_inferred_gaps(tmp_path):
    client = TestClient(create_app(tmp_path, lambda *_: FakeAgent()))
    job = client.post("/jobs/jd", json={"jd_text": "服务端工程师"}).json()[0]

    created = client.post(
        f"/jobs/{job['id']}/requirements",
        json={
            "text": "能够进行并发服务性能分析",
            "category": "domain_capability",
            "priority": "core",
            "depth": "usable",
            "origin": "inferred",
            "evidence_assessment": "gap",
        },
    )
    matrix = client.get(f"/jobs/{job['id']}/requirements")

    assert created.status_code == 200
    assert matrix.json() == [
        {
            "id": created.json()["id"],
            "job_opportunity_id": job["id"],
            "text": "能够进行并发服务性能分析",
            "category": "domain_capability",
            "priority": "core",
            "depth": "usable",
            "origin": "inferred",
            "evidence_assessment": "gap",
            "supporting_evidence_ids": [],
        }
    ]


def test_active_opportunities_are_selected_over_api(tmp_path):
    client = TestClient(create_app(tmp_path, lambda *_: FakeAgent()))
    jobs = client.post(
        "/jobs/jd",
        json={"jd_text": "招聘方向：后端工程师\n职责\n招聘方向：数据工程师\n职责"},
    ).json()

    selected = client.put("/planning/active-opportunities", json={"opportunity_ids": [jobs[0]["id"], jobs[1]["id"]]})
    current = client.get("/planning/active-opportunities")

    assert selected.status_code == 200
    assert current.json() == {"opportunity_ids": [jobs[0]["id"], jobs[1]["id"]], "primary_opportunity_id": None}


def test_task_creation_and_recommendation_over_api(tmp_path):
    client = TestClient(create_app(tmp_path, lambda *_: FakeAgent()))
    job = client.post("/jobs/jd", json={"jd_text": "后端工程师"}).json()[0]
    requirement = client.post(
        f"/jobs/{job['id']}/requirements",
        json={"text": "Python", "category": "foundational_capability", "priority": "core", "depth": "independent", "origin": "explicit", "evidence_assessment": "gap"},
    ).json()
    client.put("/planning/active-opportunities", json={"opportunity_ids": [job["id"]]})

    created = client.post(
        "/planning/tasks",
        json={"title": "完成 FastAPI 项目", "description": "实现可演示服务", "opportunity_ids": [job["id"]], "requirement_ids": [requirement["id"]], "estimated_hours": 6},
    )
    recommendation = client.get("/planning/recommendation", params={"weekly_hours": 8})

    assert created.status_code == 200
    assert recommendation.status_code == 200
    assert recommendation.json()["task"]["id"] == created.json()["id"]
    assert recommendation.json()["gap_count"] == 1
    assert recommendation.json()["feasible_in_weekly_time"] is True


def test_task_completion_requires_evidence_then_user_confirmation_over_api(tmp_path):
    client = TestClient(create_app(tmp_path, lambda *_: FakeAgent()))
    job = client.post("/jobs/jd", json={"jd_text": "后端工程师"}).json()[0]
    requirement = client.post(
        f"/jobs/{job['id']}/requirements",
        json={"text": "Python", "category": "foundational_capability", "priority": "core", "depth": "independent", "origin": "explicit", "evidence_assessment": "gap"},
    ).json()
    client.put("/planning/active-opportunities", json={"opportunity_ids": [job["id"]]})
    task = client.post(
        "/planning/tasks",
        json={"title": "完成 FastAPI 项目", "description": "实现可演示服务", "opportunity_ids": [job["id"]], "requirement_ids": [requirement["id"]], "estimated_hours": 6},
    ).json()

    submitted = client.post(
        f"/planning/tasks/{task['id']}/completion",
        json={"project_evidence": {"title": "检索服务", "technologies": "FastAPI", "contribution": "实现接口", "result": "完成演示"}, "work_link": "https://github.com/example/project"},
    )
    confirmed = client.post(f"/planning/tasks/{task['id']}/completion/confirm")

    assert submitted.status_code == 200
    assert submitted.json()["status"] == "evidence_submitted"
    assert confirmed.status_code == 200
    assert confirmed.json()["status"] == "confirmed"


def test_confirmed_task_can_be_explicitly_promoted_to_profile_evidence(tmp_path):
    client = TestClient(create_app(tmp_path, lambda *_: FakeAgent()))
    profile_id = client.post("/profiles/empty").json()["profile_id"]
    job = client.post("/jobs/jd", json={"jd_text": "后端工程师"}).json()[0]
    requirement = client.post(
        f"/jobs/{job['id']}/requirements",
        json={"text": "Python", "category": "foundational_capability", "priority": "core", "depth": "independent", "origin": "explicit", "evidence_assessment": "gap"},
    ).json()
    client.put("/planning/active-opportunities", json={"opportunity_ids": [job["id"]]})
    task = client.post(
        "/planning/tasks",
        json={"title": "完成 FastAPI 项目", "description": "实现可演示服务", "opportunity_ids": [job["id"]], "requirement_ids": [requirement["id"]], "estimated_hours": 6},
    ).json()
    client.post(
        f"/planning/tasks/{task['id']}/completion",
        json={"project_evidence": {"title": "检索服务", "technologies": "FastAPI", "contribution": "实现接口", "result": "完成演示"}},
    )
    client.post(f"/planning/tasks/{task['id']}/completion/confirm")

    promoted = client.post(f"/planning/tasks/{task['id']}/promote-evidence", json={"profile_id": profile_id})

    assert promoted.status_code == 200
    assert promoted.json()["profile_id"] == profile_id
    assert promoted.json()["evidence_type"] == "project"
    assert promoted.json()["confirmation_status"] == "confirmed"


def test_document_library_upload_list_and_delete_over_api(tmp_path):
    client = TestClient(create_app(tmp_path, lambda *_: FakeAgent()))

    uploaded = client.post(
        "/documents",
        files={"file": ("portfolio.txt", "我的 FastAPI 项目说明", "text/plain")},
    )
    listed = client.get("/documents")

    assert uploaded.status_code == 200
    assert uploaded.json()["filename"] == "portfolio.txt"
    assert uploaded.json()["index_status"] == "ready"
    assert listed.json()[0]["id"] == uploaded.json()["id"]
    assert client.delete(f"/documents/{uploaded.json()['id']}").status_code == 204
    assert client.get("/documents").json() == []


def test_career_report_aggregates_confirmed_evidence_active_jobs_and_recommendation(tmp_path):
    client = TestClient(create_app(tmp_path, lambda *_: FakeAgent()))
    profile = client.post("/profiles/resume", files={"file": ("resume.txt", "项目经历\n职业 Agent\n实现检索服务", "text/plain")}).json()
    draft = profile["drafts"][0]
    client.put(f"/evidence/{draft['id']}", json={"title": "职业 Agent", "evidence_type": "project", "details": {"technologies": "Python", "contribution": "实现 API", "result": "完成原型"}})
    client.post(f"/evidence/{draft['id']}/confirm")
    job = client.post("/jobs/jd", json={"jd_text": "后端工程师"}).json()[0]
    requirement = client.post(f"/jobs/{job['id']}/requirements", json={"text": "Python", "category": "foundational_capability", "priority": "core", "depth": "independent", "origin": "explicit", "evidence_assessment": "supported", "supporting_evidence_ids": [draft["id"]]}).json()
    client.put("/planning/active-opportunities", json={"opportunity_ids": [job["id"]]})
    client.post("/planning/tasks", json={"title": "完成项目文档", "description": "补充使用说明", "opportunity_ids": [job["id"]], "requirement_ids": [requirement["id"]], "estimated_hours": 2})

    report = client.get("/reports/career", params={"profile_id": profile["profile_id"], "weekly_hours": 8})

    assert report.status_code == 200
    assert report.json()["confirmed_evidence_count"] == 1
    assert report.json()["active_jobs"][0]["id"] == job["id"]
    assert report.json()["active_jobs"][0]["requirements"][0]["evidence_assessment"] == "supported"
    assert report.json()["next_task"]["title"] == "完成项目文档"


def test_primary_opportunity_can_only_be_selected_from_active_jobs(tmp_path):
    client = TestClient(create_app(tmp_path, lambda *_: FakeAgent()))
    jobs = client.post(
        "/jobs/jd", json={"jd_text": "岗位方向：岗位 A\n职责\n岗位方向：岗位 B\n职责"}
    ).json()
    client.put("/planning/active-opportunities", json={"opportunity_ids": [jobs[0]["id"]]})

    selected = client.put(
        "/planning/primary-opportunity", json={"opportunity_id": jobs[0]["id"]}
    )
    rejected = client.put(
        "/planning/primary-opportunity", json={"opportunity_id": jobs[1]["id"]}
    )

    assert selected.status_code == 200
    assert selected.json()["primary_opportunity_id"] == jobs[0]["id"]
    assert rejected.status_code == 400


def test_confirmed_task_can_only_be_promoted_once_per_profile(tmp_path):
    client = TestClient(create_app(tmp_path, lambda *_: FakeAgent()))
    profile_id = client.post("/profiles/empty").json()["profile_id"]
    job = client.post("/jobs/jd", json={"jd_text": "后端工程师"}).json()[0]
    requirement = client.post(
        f"/jobs/{job['id']}/requirements",
        json={"text": "Python", "category": "foundational_capability", "priority": "core", "depth": "usable", "origin": "explicit", "evidence_assessment": "gap"},
    ).json()
    client.put("/planning/active-opportunities", json={"opportunity_ids": [job["id"]]})
    task = client.post(
        "/planning/tasks",
        json={"title": "完成项目", "description": "实现服务", "opportunity_ids": [job["id"]], "requirement_ids": [requirement["id"]], "estimated_hours": 2},
    ).json()
    client.post(f"/planning/tasks/{task['id']}/completion", json={"project_evidence": {"title": "服务", "technologies": "Python", "contribution": "实现", "result": "可运行"}})
    client.post(f"/planning/tasks/{task['id']}/completion/confirm")

    first = client.post(f"/planning/tasks/{task['id']}/promote-evidence", json={"profile_id": profile_id})
    repeated = client.post(f"/planning/tasks/{task['id']}/promote-evidence", json={"profile_id": profile_id})

    assert first.status_code == 200
    assert repeated.status_code == 409


def test_managed_document_can_be_replaced_without_changing_its_id(tmp_path):
    client = TestClient(create_app(tmp_path, lambda *_: FakeAgent()))
    uploaded = client.post("/documents", files={"file": ("old.txt", "旧内容", "text/plain")}).json()

    replaced = client.put(
        f"/documents/{uploaded['id']}",
        files={"file": ("new.txt", "新内容", "text/plain")},
    )

    assert replaced.status_code == 200
    assert replaced.json()["id"] == uploaded["id"]
    assert replaced.json()["filename"] == "new.txt"
    assert replaced.json()["index_status"] == "ready"


def test_resume_optimization_returns_editable_bullets_bound_to_evidence(tmp_path):
    client = TestClient(create_app(tmp_path, lambda *_: FakeAgent()))
    profile = client.post("/profiles/resume", files={"file": ("resume.txt", "项目经历\n职业 Agent\n实现检索服务", "text/plain")}).json()
    draft = profile["drafts"][0]
    client.put(f"/evidence/{draft['id']}", json={"title": "职业 Agent", "evidence_type": "project", "details": {"technologies": "Python", "contribution": "实现检索 API", "result": "完成原型"}})
    client.post(f"/evidence/{draft['id']}/confirm")
    job = client.post("/jobs/jd", json={"jd_text": "后端工程师"}).json()[0]
    requirement = client.post(f"/jobs/{job['id']}/requirements", json={"text": "Python", "category": "foundational_capability", "priority": "core", "depth": "independent", "origin": "explicit", "evidence_assessment": "supported", "supporting_evidence_ids": [draft["id"]]}).json()

    response = client.get("/resume/optimization", params={"profile_id": profile["profile_id"], "opportunity_id": job["id"]})

    assert response.status_code == 200
    suggestion = response.json()["suggestions"][0]
    assert suggestion["evidence_id"] == draft["id"]
    assert suggestion["supporting_requirement_ids"] == [requirement["id"]]
    assert "实现检索 API" in suggestion["suggested_bullet"]


def test_chat_includes_local_career_context_when_a_profile_is_selected(tmp_path):
    agent = FakeAgent({"messages": [AIMessage(content="已结合档案回答。")]})
    consents = ConsentStore(tmp_path / "consents.sqlite3")
    consents.set(ConsentScope.MODEL_ANALYSIS, granted=True)
    client = TestClient(create_app(tmp_path, lambda *_: agent, consents))
    profile_id = client.post("/profiles/empty").json()["profile_id"]
    job = client.post("/jobs/jd", json={"jd_text": "后端工程师"}).json()[0]
    client.put("/planning/active-opportunities", json={"opportunity_ids": [job["id"]]})

    response = client.post("/chat", json={"message": "我该怎么准备？", "profile_id": profile_id})

    assert response.status_code == 200
    assert "[职业工作台上下文]" in agent.payloads[0]["messages"][0]["content"]
    consents.close()


def test_profile_supports_multiple_manual_evidence_records_of_the_same_type(tmp_path):
    client = TestClient(create_app(tmp_path, lambda *_: FakeAgent()))
    profile_id = client.post("/profiles/empty").json()["profile_id"]
    payload = {
        "evidence_type": "project",
        "title": "项目 A",
        "details": {"technologies": "Python", "contribution": "实现 API", "result": "完成原型"},
    }

    first = client.post(f"/profiles/{profile_id}/evidence", json=payload)
    second = client.post(
        f"/profiles/{profile_id}/evidence",
        json={**payload, "title": "项目 B"},
    )
    client.post(f"/evidence/{first.json()['id']}/confirm")
    client.post(f"/evidence/{second.json()['id']}/confirm")
    removed = client.delete(f"/evidence/{first.json()['id']}")

    assert first.status_code == 200
    assert second.status_code == 200
    assert removed.status_code == 204
    report = client.get("/reports/career", params={"profile_id": profile_id})
    assert report.json()["confirmed_evidence_count"] == 1


def test_resume_upload_splits_multiple_projects_and_competitions_and_prefills_fields(tmp_path):
    client = TestClient(create_app(tmp_path, lambda *_: FakeAgent()))
    resume = """项目经历
职业 Agent
技术：Python、FastAPI
职责：实现岗位分析接口
成果：完成本地可运行原型

数据看板
技术：Python、SQL
职责：搭建数据查询页面
成果：减少人工整理

竞赛经历
数学建模竞赛
主办方：全国大学生数学建模竞赛
奖项：省级二等奖
主题：物流优化
个人贡献：负责建模与论文撰写

算法竞赛
主办方：ICPC
奖项：区域赛铜奖
主题：程序设计
个人贡献：完成图论题目
"""

    response = client.post(
        "/profiles/resume", files={"file": ("resume.txt", resume, "text/plain")}
    )

    assert response.status_code == 200
    drafts = response.json()["drafts"]
    assert [(item["evidence_type"], item["title"]) for item in drafts] == [
        ("project", "职业 Agent"),
        ("project", "数据看板"),
        ("competition", "数学建模竞赛"),
        ("competition", "算法竞赛"),
    ]
    assert drafts[0]["details"]["technologies"] == "Python、FastAPI"
    assert drafts[0]["details"]["contribution"] == "实现岗位分析接口"
    assert drafts[2]["details"]["outcome"] == "省级二等奖"


def test_resume_upload_uses_model_structured_extraction_after_model_consent(tmp_path):
    consents = ConsentStore(tmp_path / "consents.sqlite3")
    consents.set(ConsentScope.MODEL_ANALYSIS, granted=True)
    seen_texts = []
    client = TestClient(
        create_app(
            tmp_path,
            lambda *_: FakeAgent(),
            consents,
            resume_extractor=lambda text: seen_texts.append(text) or [
                {"evidence_type": "project", "title": "模型拆分项目 A", "details": {"technologies": "Python", "contribution": "实现接口", "result": "完成原型"}},
                {"evidence_type": "project", "title": "模型拆分项目 B", "details": {"technologies": "SQL", "contribution": "分析数据", "result": "产出报表"}},
            ],
        )
    )

    response = client.post(
        "/profiles/resume",
        files={"file": ("resume.txt", "一份格式不规整的简历", "text/plain")},
    )

    assert response.status_code == 200
    assert [draft["title"] for draft in response.json()["drafts"]] == ["模型拆分项目 A", "模型拆分项目 B"]
    assert seen_texts == ["一份格式不规整的简历"]
    assert all(draft["confirmation_status"] == "draft" for draft in response.json()["drafts"])
    consents.close()

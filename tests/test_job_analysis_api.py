from fastapi.testclient import TestClient

from career_agent.api import create_app
from career_agent.jobs import (
    EvidenceAssessment,
    RequirementCategory,
    RequirementDepth,
    RequirementOrigin,
    RequirementPriority,
)
from career_agent.research import ConsentScope, ConsentStore


class FakeAgent:
    def invoke(self, *_args, **_kwargs):
        raise AssertionError("chat agent is not used for JD analysis")


def test_model_analysis_creates_a_labelled_requirement_matrix_from_confirmed_evidence(tmp_path):
    observed = {}

    def analyzer(jd_text, evidence):
        observed["jd_text"] = jd_text
        observed["evidence_ids"] = [item.id for item in evidence]
        return {
            "responsibilities": ["负责高并发 API 服务"],
            "requirements": [
                {
                    "text": "熟练使用 Python",
                    "category": RequirementCategory.FOUNDATIONAL_CAPABILITY,
                    "priority": RequirementPriority.CORE,
                    "depth": RequirementDepth.INDEPENDENT,
                    "origin": RequirementOrigin.EXPLICIT,
                    "evidence_assessment": EvidenceAssessment.SUPPORTED,
                    "supporting_evidence_ids": [evidence[0].id],
                },
                {
                    "text": "能够进行并发服务性能分析",
                    "category": RequirementCategory.DOMAIN_CAPABILITY,
                    "priority": RequirementPriority.CORE,
                    "depth": RequirementDepth.USABLE,
                    "origin": RequirementOrigin.INFERRED,
                    "evidence_assessment": EvidenceAssessment.GAP,
                    "supporting_evidence_ids": [],
                },
            ],
        }

    consents = ConsentStore(tmp_path / "consents.sqlite3")
    consents.set(ConsentScope.MODEL_ANALYSIS, granted=True)
    client = TestClient(create_app(tmp_path, lambda *_: FakeAgent(), consents, analyzer))
    initialization = client.post(
        "/profiles/resume",
        files={"file": ("resume.txt", "项目经历\n职业 Agent\n实现检索服务", "text/plain")},
    ).json()
    profile_id = initialization["profile_id"]
    draft = initialization["drafts"][0]
    client.put(
        f"/evidence/{draft['id']}",
        json={"title": "职业 Agent", "evidence_type": "project", "details": {"technologies": "Python", "contribution": "实现 API", "result": "完成原型"}},
    )
    client.post(f"/evidence/{draft['id']}/confirm")
    job = client.post("/jobs/jd", json={"jd_text": "后端工程师\n负责高并发 API 服务"}).json()[0]

    response = client.post(f"/jobs/{job['id']}/analyze", json={"profile_id": profile_id})

    assert response.status_code == 200
    assert observed["jd_text"] == job["source_jd_text"]
    assert len(observed["evidence_ids"]) == 1
    assert response.json()["responsibilities"] == ["负责高并发 API 服务"]
    assert response.json()["requirements"][1]["origin"] == "inferred"
    assert response.json()["requirements"][1]["evidence_assessment"] == "gap"
    consents.close()

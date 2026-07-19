import pytest

from career_agent.jobs import (
    EvidenceAssessment,
    JobConfirmationStatus,
    JobOpportunityStore,
    RequirementCategory,
    RequirementDepth,
    RequirementOrigin,
    RequirementPriority,
)


@pytest.fixture
def store(tmp_path):
    database = JobOpportunityStore(tmp_path / "jobs.sqlite3")
    yield database
    database.close()


def test_multi_role_jd_becomes_independent_drafts(store):
    jd = """招聘方向：服务端开发工程师
岗位职责：负责 API 服务与数据库优化
岗位要求：熟悉 Python

招聘方向：Agent 后端研发工程师
岗位职责：负责 Agent 执行环境
岗位要求：了解容器技术
"""

    drafts = store.import_jd_document(jd)

    assert [draft.title for draft in drafts] == ["服务端开发工程师", "Agent 后端研发工程师"]
    assert all(draft.confirmation_status == JobConfirmationStatus.DRAFT for draft in drafts)
    assert "数据库优化" in drafts[0].source_jd_text
    assert "容器技术" in drafts[1].source_jd_text


def test_single_jd_remains_one_draft_and_can_be_confirmed(store):
    draft = store.import_jd_document("负责检索服务开发，要求掌握 Python")[0]

    confirmed = store.confirm_job_opportunity(draft.id)

    assert confirmed.confirmation_status == JobConfirmationStatus.CONFIRMED
    with pytest.raises(ValueError, match="Only draft"):
        store.update_draft_job_opportunity(confirmed.id, title="另一个职位")


def test_requirement_matrix_keeps_responsibilities_origin_and_evidence_state(store):
    opportunity = store.import_jd_document("招聘方向：后端工程师\n岗位职责：负责高并发 API 服务")[0]
    responsibility = store.add_responsibility(opportunity.id, text="负责高并发 API 服务")
    explicit = store.add_requirement(
        opportunity.id,
        text="熟练掌握 Python",
        category=RequirementCategory.FOUNDATIONAL_CAPABILITY,
        priority=RequirementPriority.CORE,
        depth=RequirementDepth.INDEPENDENT,
        origin=RequirementOrigin.EXPLICIT,
        evidence_assessment=EvidenceAssessment.SUPPORTED,
        supporting_evidence_ids=["evidence-python-project"],
    )
    inferred = store.add_requirement(
        opportunity.id,
        text="能够进行并发服务的性能分析",
        category=RequirementCategory.DOMAIN_CAPABILITY,
        priority=RequirementPriority.CORE,
        depth=RequirementDepth.USABLE,
        origin=RequirementOrigin.INFERRED,
        evidence_assessment=EvidenceAssessment.GAP,
    )

    matrix = store.requirement_matrix(opportunity.id)

    assert responsibility.text == "负责高并发 API 服务"
    assert matrix == [explicit, inferred]
    assert matrix[0].supporting_evidence_ids == ("evidence-python-project",)
    assert matrix[1].origin == RequirementOrigin.INFERRED
    assert matrix[1].evidence_assessment == EvidenceAssessment.GAP


@pytest.mark.parametrize(
    "assessment",
    [
        EvidenceAssessment.EVIDENCE_INSUFFICIENT,
        EvidenceAssessment.GAP,
        EvidenceAssessment.NOT_APPLICABLE,
    ],
)
def test_matrix_supports_all_non_supported_evidence_states(store, assessment):
    opportunity = store.import_jd_document("数据工程师")[0]

    requirement = store.add_requirement(
        opportunity.id,
        text="SQL",
        category=RequirementCategory.DOMAIN_CAPABILITY,
        priority=RequirementPriority.PREFERRED,
        depth=RequirementDepth.USABLE,
        origin=RequirementOrigin.EXPLICIT,
        evidence_assessment=assessment,
    )

    assert requirement.evidence_assessment == assessment


def test_supported_requirement_requires_evidence_link(store):
    opportunity = store.import_jd_document("数据工程师")[0]

    with pytest.raises(ValueError, match="linked evidence"):
        store.add_requirement(
            opportunity.id,
            text="SQL",
            category=RequirementCategory.DOMAIN_CAPABILITY,
            priority=RequirementPriority.CORE,
            depth=RequirementDepth.USABLE,
            origin=RequirementOrigin.EXPLICIT,
            evidence_assessment=EvidenceAssessment.SUPPORTED,
        )

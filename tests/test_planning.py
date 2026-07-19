from datetime import date, timedelta

import pytest

from career_agent.jobs import (
    EvidenceAssessment,
    JobRequirement,
    RequirementCategory,
    RequirementDepth,
    RequirementOrigin,
    RequirementPriority,
)
from career_agent.planning import PreparationTaskStore, TaskStatus


def requirement(
    identifier: str, opportunity_id: str, text: str, assessment: EvidenceAssessment
) -> JobRequirement:
    return JobRequirement(
        id=identifier,
        job_opportunity_id=opportunity_id,
        text=text,
        category=RequirementCategory.DOMAIN_CAPABILITY,
        priority=RequirementPriority.CORE,
        depth=RequirementDepth.USABLE,
        origin=RequirementOrigin.EXPLICIT,
        evidence_assessment=assessment,
        supporting_evidence_ids=(),
    )


@pytest.fixture
def store(tmp_path):
    database = PreparationTaskStore(tmp_path / "planning.sqlite3")
    yield database
    database.close()


def test_active_opportunities_are_limited_to_three(store):
    assert store.set_active_opportunities(["job-a", "job-b", "job-c"]) == ("job-a", "job-b", "job-c")

    with pytest.raises(ValueError, match="At most three"):
        store.set_active_opportunities(["job-a", "job-b", "job-c", "job-d"])


def test_divergent_core_requirements_require_a_primary_selection(store):
    store.set_active_opportunities(["backend", "product"])
    requirements = [
        requirement("r1", "backend", "Python 服务开发", EvidenceAssessment.GAP),
        requirement("r2", "product", "用户研究", EvidenceAssessment.GAP),
    ]

    assert store.primary_selection_required(requirements)
    store.select_primary_opportunity("backend")
    assert not store.primary_selection_required(requirements)


def test_recommendation_explains_gaps_reuse_feasibility_and_primary_priority(store):
    store.set_active_opportunities(["backend", "agent"])
    store.select_primary_opportunity("backend")
    requirements = [
        requirement("r1", "backend", "Python", EvidenceAssessment.GAP),
        requirement("r2", "agent", "Python", EvidenceAssessment.EVIDENCE_INSUFFICIENT),
    ]
    shared_task = store.add_task(
        title="完成 FastAPI 检索服务项目", description="实现可演示 API", opportunity_ids=["backend", "agent"],
        requirement_ids=["r1", "r2"], estimated_hours=6,
        deadline=date(2026, 7, 20),
    )
    store.add_task(
        title="阅读文章", description="了解概念", opportunity_ids=["agent"],
        requirement_ids=["r2"], estimated_hours=1,
    )

    recommendation = store.recommend_next_task(requirements, weekly_hours=8, today=date(2026, 7, 16))

    assert recommendation is not None
    assert recommendation.task == shared_task
    assert recommendation.gap_count == 2
    assert recommendation.shared_opportunity_count == 2
    assert recommendation.primary_opportunity_bonus
    assert recommendation.feasible_in_weekly_time
    assert recommendation.deadline_urgency == "urgent"
    assert any("活跃岗位" in item for item in recommendation.rationale)


def test_task_completion_requires_structured_evidence_then_user_confirmation(store):
    store.set_active_opportunities(["backend"])
    task = store.add_task(
        title="完成服务端项目", description="交付一个 API", opportunity_ids=["backend"],
        requirement_ids=["r1"], estimated_hours=4,
    )

    with pytest.raises(ValueError, match="technologies"):
        store.submit_completion_evidence(task.id, project_evidence={"title": "API 项目"})

    submitted = store.submit_completion_evidence(
        task.id,
        project_evidence={
            "title": "API 项目", "technologies": "FastAPI", "contribution": "实现检索接口", "result": "完成部署",
        },
        work_link="https://github.com/example/project",
    )
    assert submitted.status == TaskStatus.EVIDENCE_SUBMITTED
    assert store.confirm_completion(task.id).status == TaskStatus.CONFIRMED


def test_shared_task_must_only_target_active_opportunities(store):
    store.set_active_opportunities(["backend"])

    with pytest.raises(ValueError, match="active opportunities"):
        store.add_task(
            title="无效任务", description="不应保存", opportunity_ids=["backend", "product"],
            requirement_ids=["r1"], estimated_hours=2,
        )

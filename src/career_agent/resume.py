"""Evidence-bound resume suggestions that never invent candidate claims."""
from __future__ import annotations

from dataclasses import dataclass

from .jobs import JobRequirement
from .profile import EvidenceRecord


@dataclass(frozen=True)
class ResumeSuggestion:
    section: str
    evidence_id: str
    evidence_title: str
    suggested_bullet: str
    supporting_requirement_ids: tuple[str, ...]


def build_resume_suggestions(
    evidence_records: list[EvidenceRecord], requirements: list[JobRequirement]
) -> list[ResumeSuggestion]:
    """Render only user-confirmed details into editable resume bullets."""
    suggestions: list[ResumeSuggestion] = []
    for evidence in evidence_records:
        details = evidence.details
        contribution = details.get("contribution") or details.get("responsibilities")
        result = details.get("result") or details.get("outcome")
        technologies = details.get("technologies")
        if not contribution or not result:
            continue
        parts = [f"{evidence.title}：{contribution}"]
        if technologies:
            parts.append(f"使用 {technologies}")
        parts.append(f"结果：{result}")
        linked_ids = tuple(
            item.id for item in requirements if evidence.id in item.supporting_evidence_ids
        )
        suggestions.append(
            ResumeSuggestion(
                section=_section_name(evidence.evidence_type.value),
                evidence_id=evidence.id,
                evidence_title=evidence.title,
                suggested_bullet="；".join(parts) + "。",
                supporting_requirement_ids=linked_ids,
            )
        )
    return suggestions


def _section_name(evidence_type: str) -> str:
    return {
        "project": "项目经历",
        "work": "实习/工作经历",
        "competition": "竞赛经历",
        "course": "课程与实践",
        "campus": "校园经历",
    }.get(evidence_type, "其他经历")

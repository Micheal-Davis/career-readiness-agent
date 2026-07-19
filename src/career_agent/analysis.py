"""Model-backed, evidence-constrained analysis of one job opportunity."""
from __future__ import annotations

import os
from collections.abc import Callable

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from .jobs import EvidenceAssessment, RequirementCategory, RequirementDepth, RequirementOrigin, RequirementPriority
from .profile import EvidenceRecord, EvidenceType


class AnalysedRequirement(BaseModel):
    text: str
    category: RequirementCategory
    priority: RequirementPriority
    depth: RequirementDepth
    origin: RequirementOrigin
    evidence_assessment: EvidenceAssessment
    supporting_evidence_ids: list[str] = Field(default_factory=list)


class JobAnalysis(BaseModel):
    responsibilities: list[str] = Field(default_factory=list)
    requirements: list[AnalysedRequirement] = Field(default_factory=list)


JobAnalyzer = Callable[[str, list[EvidenceRecord]], JobAnalysis | dict]


class ResumeEvidenceCandidate(BaseModel):
    evidence_type: EvidenceType
    title: str
    details: dict[str, str]


class ResumeExtraction(BaseModel):
    evidence: list[ResumeEvidenceCandidate] = Field(default_factory=list)


ResumeExtractor = Callable[[str], list[dict] | ResumeExtraction]


def create_model_job_analyzer() -> JobAnalyzer:
    """Create the default structured analyzer only when an endpoint needs it."""
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("缺少 OPENAI_API_KEY，无法执行模型分析。")
    model_args = {"model": os.getenv("OPENAI_MODEL", "gpt-4.1-mini"), "temperature": 0}
    if base_url := os.getenv("OPENAI_BASE_URL"):
        model_args["base_url"] = base_url
    model = ChatOpenAI(**model_args).with_structured_output(JobAnalysis)

    def analyze(jd_text: str, evidence: list[EvidenceRecord]) -> JobAnalysis:
        evidence_summary = [
            {"id": item.id, "title": item.title, "type": item.evidence_type.value, "details": item.details}
            for item in evidence
        ]
        result = model.invoke(
            "分析以下岗位 JD。职责必须与要求分开；只有从职责合理推导出的要求才标记 inferred。"
            "证据状态只能基于给定的已确认经历；supported 时必须引用对应 evidence ID，否则用 evidence_insufficient、gap 或 not_applicable。"
            f"\n\nJD:\n{jd_text}\n\n已确认经历：\n{evidence_summary}"
        )
        return JobAnalysis.model_validate(result)

    return analyze


def create_model_resume_extractor() -> ResumeExtractor:
    """Create a consent-gated structured extractor for semi-structured resume text."""
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("缺少 OPENAI_API_KEY，无法执行模型分析。")
    model_args = {"model": os.getenv("OPENAI_MODEL", "gpt-4.1-mini"), "temperature": 0}
    if base_url := os.getenv("OPENAI_BASE_URL"):
        model_args["base_url"] = base_url
    model = ChatOpenAI(**model_args).with_structured_output(ResumeExtraction)

    def extract(resume_text: str) -> ResumeExtraction:
        return ResumeExtraction.model_validate(model.invoke(
            "将以下简历拆成多条待确认经历。不得编造内容，标题删除日期、角色和奖项等附加信息。"
            "每一个独立项目标题、日期区间或技术栈都必须对应独立项目，绝不可合并两个项目；保留原文中的结果与个人贡献。"
            "字段严格限定为：project(technologies, contribution, result)、"
            "competition(outcome, contribution)、work(responsibilities, work_content)、"
            "campus(responsibilities, developed_capabilities)、"
            "course(course_or_activity, outcome, related_work)。"
            "每一条经历独立返回；无法从原文支持的字段留空。"
            f"\n\n简历原文：\n{resume_text}"
        ))

    return extract

"""Persistent, independently confirmable job opportunities and requirements."""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from uuid import uuid4


class JobConfirmationStatus(StrEnum):
    DRAFT = "draft"
    CONFIRMED = "confirmed"


class RequirementCategory(StrEnum):
    ELIGIBILITY = "eligibility"
    FOUNDATIONAL_CAPABILITY = "foundational_capability"
    DOMAIN_CAPABILITY = "domain_capability"
    DEMONSTRATED_ACHIEVEMENT_PREFERENCE = "demonstrated_achievement_preference"
    WORK_STYLE_EXPECTATION = "work_style_expectation"


class RequirementOrigin(StrEnum):
    EXPLICIT = "explicit"
    INFERRED = "inferred"


class RequirementPriority(StrEnum):
    ELIGIBILITY = "eligibility"
    CORE = "core"
    PREFERRED = "preferred"


class RequirementDepth(StrEnum):
    AWARE = "aware"
    USABLE = "usable"
    INDEPENDENT = "independent"
    DEEP = "deep"


class EvidenceAssessment(StrEnum):
    SUPPORTED = "supported"
    EVIDENCE_INSUFFICIENT = "evidence_insufficient"
    GAP = "gap"
    NOT_APPLICABLE = "not_applicable"


@dataclass(frozen=True)
class JobOpportunity:
    id: str
    title: str
    source_jd_text: str
    confirmation_status: JobConfirmationStatus


@dataclass(frozen=True)
class JobResponsibility:
    id: str
    job_opportunity_id: str
    text: str


@dataclass(frozen=True)
class JobRequirement:
    id: str
    job_opportunity_id: str
    text: str
    category: RequirementCategory
    priority: RequirementPriority
    depth: RequirementDepth
    origin: RequirementOrigin
    evidence_assessment: EvidenceAssessment
    supporting_evidence_ids: tuple[str, ...]


@dataclass(frozen=True)
class JobResearchSource:
    id: str
    job_opportunity_id: str
    title: str
    url: str
    summary: str
    tier: str


class JobOpportunityStore:
    """SQLite repository for job opportunities, responsibilities, and requirement matrices."""

    def __init__(self, database_path: Path) -> None:
        database_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(database_path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._setup()

    def close(self) -> None:
        self._connection.close()

    def import_jd_document(self, jd_text: str) -> list[JobOpportunity]:
        """Split a multi-role JD into independently editable draft opportunities."""
        if not jd_text.strip():
            raise ValueError("JD text cannot be blank.")
        opportunities = []
        for title, section_text in _split_jd_roles(jd_text):
            opportunity = JobOpportunity(
                id=str(uuid4()),
                title=title,
                source_jd_text=section_text,
                confirmation_status=JobConfirmationStatus.DRAFT,
            )
            self._connection.execute(
                """
                INSERT INTO job_opportunities (id, title, source_jd_text, confirmation_status)
                VALUES (?, ?, ?, ?)
                """,
                (
                    opportunity.id,
                    opportunity.title,
                    opportunity.source_jd_text,
                    opportunity.confirmation_status.value,
                ),
            )
            opportunities.append(opportunity)
        self._connection.commit()
        return opportunities

    def confirm_job_opportunity(self, opportunity_id: str) -> JobOpportunity:
        self._get_opportunity(opportunity_id)
        self._connection.execute(
            "UPDATE job_opportunities SET confirmation_status = ? WHERE id = ?",
            (JobConfirmationStatus.CONFIRMED.value, opportunity_id),
        )
        self._connection.commit()
        return self._get_opportunity(opportunity_id)

    def get_opportunity(self, opportunity_id: str) -> JobOpportunity:
        return self._get_opportunity(opportunity_id)

    def list_opportunities(self) -> list[JobOpportunity]:
        rows = self._connection.execute("SELECT * FROM job_opportunities ORDER BY rowid").fetchall()
        return [
            JobOpportunity(
                id=row["id"],
                title=row["title"],
                source_jd_text=row["source_jd_text"],
                confirmation_status=JobConfirmationStatus(row["confirmation_status"]),
            )
            for row in rows
        ]

    def update_draft_job_opportunity(self, opportunity_id: str, *, title: str) -> JobOpportunity:
        opportunity = self._get_opportunity(opportunity_id)
        if opportunity.confirmation_status != JobConfirmationStatus.DRAFT:
            raise ValueError("Only draft job opportunities can be edited.")
        if not title.strip():
            raise ValueError("Job opportunity title cannot be blank.")
        self._connection.execute(
            "UPDATE job_opportunities SET title = ? WHERE id = ?",
            (title.strip(), opportunity_id),
        )
        self._connection.commit()
        return self._get_opportunity(opportunity_id)

    def discard_draft_job_opportunity(self, opportunity_id: str) -> None:
        opportunity = self._get_opportunity(opportunity_id)
        if opportunity.confirmation_status != JobConfirmationStatus.DRAFT:
            raise ValueError("Only draft job opportunities can be discarded.")
        self._connection.execute("DELETE FROM job_opportunities WHERE id = ?", (opportunity_id,))
        self._connection.commit()

    def add_responsibility(self, opportunity_id: str, *, text: str) -> JobResponsibility:
        self._get_opportunity(opportunity_id)
        if not text.strip():
            raise ValueError("Responsibility text cannot be blank.")
        responsibility = JobResponsibility(id=str(uuid4()), job_opportunity_id=opportunity_id, text=text.strip())
        self._connection.execute(
            "INSERT INTO job_responsibilities (id, job_opportunity_id, text) VALUES (?, ?, ?)",
            (responsibility.id, responsibility.job_opportunity_id, responsibility.text),
        )
        self._connection.commit()
        return responsibility

    def add_requirement(
        self,
        opportunity_id: str,
        *,
        text: str,
        category: RequirementCategory,
        priority: RequirementPriority,
        depth: RequirementDepth,
        origin: RequirementOrigin,
        evidence_assessment: EvidenceAssessment,
        supporting_evidence_ids: list[str] | None = None,
    ) -> JobRequirement:
        self._get_opportunity(opportunity_id)
        if not text.strip():
            raise ValueError("Requirement text cannot be blank.")
        evidence_ids = tuple(dict.fromkeys(supporting_evidence_ids or []))
        if evidence_assessment == EvidenceAssessment.SUPPORTED and not evidence_ids:
            raise ValueError("Supported requirements need linked evidence.")
        if evidence_assessment != EvidenceAssessment.SUPPORTED and evidence_ids:
            raise ValueError("Only supported requirements may link supporting evidence.")
        requirement = JobRequirement(
            id=str(uuid4()),
            job_opportunity_id=opportunity_id,
            text=text.strip(),
            category=category,
            priority=priority,
            depth=depth,
            origin=origin,
            evidence_assessment=evidence_assessment,
            supporting_evidence_ids=evidence_ids,
        )
        self._connection.execute(
            """
            INSERT INTO job_requirements
                (id, job_opportunity_id, text, category, priority, depth, origin, evidence_assessment)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                requirement.id,
                requirement.job_opportunity_id,
                requirement.text,
                requirement.category.value,
                requirement.priority.value,
                requirement.depth.value,
                requirement.origin.value,
                requirement.evidence_assessment.value,
            ),
        )
        self._connection.executemany(
            "INSERT INTO job_requirement_evidence (requirement_id, evidence_id) VALUES (?, ?)",
            [(requirement.id, evidence_id) for evidence_id in evidence_ids],
        )
        self._connection.commit()
        return requirement

    def requirement_matrix(self, opportunity_id: str) -> list[JobRequirement]:
        self._get_opportunity(opportunity_id)
        rows = self._connection.execute(
            "SELECT * FROM job_requirements WHERE job_opportunity_id = ? ORDER BY rowid",
            (opportunity_id,),
        ).fetchall()
        return [self._row_to_requirement(row) for row in rows]

    def requirements_for_opportunities(self, opportunity_ids: tuple[str, ...]) -> list[JobRequirement]:
        return [
            requirement
            for opportunity_id in opportunity_ids
            for requirement in self.requirement_matrix(opportunity_id)
        ]

    def save_research_sources(
        self,
        opportunity_id: str,
        *,
        sources: list[tuple[str, str, str, str]],
    ) -> list[JobResearchSource]:
        """Store user-requested public research sources, de-duplicated by URL per job."""
        self._get_opportunity(opportunity_id)
        for title, url, summary, tier in sources:
            if not url.startswith(("https://", "http://")):
                raise ValueError("Research source URL must use http or https.")
            self._connection.execute(
                """
                INSERT OR IGNORE INTO job_research_sources
                    (id, job_opportunity_id, title, url, summary, tier)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (str(uuid4()), opportunity_id, title, url, summary, tier),
            )
        self._connection.commit()
        return self.list_research_sources(opportunity_id)

    def list_research_sources(self, opportunity_id: str) -> list[JobResearchSource]:
        self._get_opportunity(opportunity_id)
        rows = self._connection.execute(
            "SELECT * FROM job_research_sources WHERE job_opportunity_id = ? ORDER BY rowid",
            (opportunity_id,),
        ).fetchall()
        return [
            JobResearchSource(
                id=row["id"], job_opportunity_id=row["job_opportunity_id"],
                title=row["title"], url=row["url"], summary=row["summary"], tier=row["tier"],
            )
            for row in rows
        ]

    def _setup(self) -> None:
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS job_opportunities (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                source_jd_text TEXT NOT NULL,
                confirmation_status TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS job_responsibilities (
                id TEXT PRIMARY KEY,
                job_opportunity_id TEXT NOT NULL REFERENCES job_opportunities(id) ON DELETE CASCADE,
                text TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS job_requirements (
                id TEXT PRIMARY KEY,
                job_opportunity_id TEXT NOT NULL REFERENCES job_opportunities(id) ON DELETE CASCADE,
                text TEXT NOT NULL,
                category TEXT NOT NULL,
                priority TEXT NOT NULL,
                depth TEXT NOT NULL,
                origin TEXT NOT NULL,
                evidence_assessment TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS job_requirement_evidence (
                requirement_id TEXT NOT NULL REFERENCES job_requirements(id) ON DELETE CASCADE,
                evidence_id TEXT NOT NULL,
                PRIMARY KEY (requirement_id, evidence_id)
            );
            CREATE TABLE IF NOT EXISTS job_research_sources (
                id TEXT PRIMARY KEY,
                job_opportunity_id TEXT NOT NULL REFERENCES job_opportunities(id) ON DELETE CASCADE,
                title TEXT NOT NULL,
                url TEXT NOT NULL,
                summary TEXT NOT NULL,
                tier TEXT NOT NULL,
                UNIQUE(job_opportunity_id, url)
            );
            """
        )
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.commit()

    def _get_opportunity(self, opportunity_id: str) -> JobOpportunity:
        row = self._connection.execute(
            "SELECT * FROM job_opportunities WHERE id = ?", (opportunity_id,)
        ).fetchone()
        if row is None:
            raise ValueError("Job opportunity does not exist.")
        return JobOpportunity(
            id=row["id"],
            title=row["title"],
            source_jd_text=row["source_jd_text"],
            confirmation_status=JobConfirmationStatus(row["confirmation_status"]),
        )

    def _row_to_requirement(self, row: sqlite3.Row) -> JobRequirement:
        evidence_rows = self._connection.execute(
            "SELECT evidence_id FROM job_requirement_evidence WHERE requirement_id = ? ORDER BY rowid",
            (row["id"],),
        ).fetchall()
        return JobRequirement(
            id=row["id"],
            job_opportunity_id=row["job_opportunity_id"],
            text=row["text"],
            category=RequirementCategory(row["category"]),
            priority=RequirementPriority(row["priority"]),
            depth=RequirementDepth(row["depth"]),
            origin=RequirementOrigin(row["origin"]),
            evidence_assessment=EvidenceAssessment(row["evidence_assessment"]),
            supporting_evidence_ids=tuple(item["evidence_id"] for item in evidence_rows),
        )


def _split_jd_roles(jd_text: str) -> list[tuple[str, str]]:
    """Use short role headings to split one JD document; otherwise retain one draft."""
    sections: list[tuple[str, list[str]]] = []
    current: tuple[str, list[str]] | None = None
    for raw_line in jd_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if _is_role_heading(line):
            if current is not None:
                sections.append(current)
            current = (_clean_role_heading(line), [line])
        elif current is not None:
            current[1].append(line)

    if current is not None:
        sections.append(current)
    if not sections:
        return [("待确认岗位", jd_text.strip())]
    return [(title, "\n".join(lines)) for title, lines in sections]


def _is_role_heading(line: str) -> bool:
    normalized = line.replace(" ", "")
    markers = ("招聘方向", "岗位方向", "岗位：", "岗位:", "方向：", "方向:")
    return len(normalized) <= 60 and any(marker in normalized for marker in markers)


def _clean_role_heading(line: str) -> str:
    for marker in ("招聘方向", "岗位方向", "岗位：", "岗位:", "方向：", "方向:"):
        if marker in line:
            title = line.split(marker, maxsplit=1)[-1].strip(" ：:")
            return title or "待确认岗位"
    return line

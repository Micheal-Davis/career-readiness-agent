"""Transparent preparation-task planning for up to three active opportunities."""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from pathlib import Path
from uuid import uuid4

from .jobs import EvidenceAssessment, JobRequirement, RequirementPriority


class TaskStatus(StrEnum):
    PROPOSED = "proposed"
    EVIDENCE_SUBMITTED = "evidence_submitted"
    CONFIRMED = "confirmed"


@dataclass(frozen=True)
class PreparationTask:
    id: str
    title: str
    description: str
    opportunity_ids: tuple[str, ...]
    requirement_ids: tuple[str, ...]
    estimated_hours: int
    deadline: date | None
    status: TaskStatus


@dataclass(frozen=True)
class TaskRecommendation:
    task: PreparationTask
    gap_count: int
    shared_opportunity_count: int
    primary_opportunity_bonus: bool
    feasible_in_weekly_time: bool
    deadline_urgency: str
    rationale: tuple[str, ...]


class PreparationTaskStore:
    """SQLite repository for active targets, preparation tasks, and completion review."""

    def __init__(self, database_path: Path) -> None:
        database_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(database_path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._setup()

    def close(self) -> None:
        self._connection.close()

    def set_active_opportunities(self, opportunity_ids: list[str]) -> tuple[str, ...]:
        unique_ids = tuple(dict.fromkeys(opportunity_ids))
        if len(unique_ids) > 3:
            raise ValueError("At most three opportunities can be active at once.")
        self._connection.execute("DELETE FROM active_opportunities")
        self._connection.executemany(
            "INSERT INTO active_opportunities (opportunity_id) VALUES (?)",
            [(opportunity_id,) for opportunity_id in unique_ids],
        )
        self._connection.execute("DELETE FROM planning_settings WHERE setting = 'primary_opportunity_id'")
        self._connection.commit()
        return unique_ids

    def select_primary_opportunity(self, opportunity_id: str) -> None:
        if opportunity_id not in self.active_opportunity_ids():
            raise ValueError("Primary opportunity must be active.")
        self._connection.execute(
            """
            INSERT INTO planning_settings (setting, value) VALUES ('primary_opportunity_id', ?)
            ON CONFLICT(setting) DO UPDATE SET value = excluded.value
            """,
            (opportunity_id,),
        )
        self._connection.commit()

    def active_opportunity_ids(self) -> tuple[str, ...]:
        rows = self._connection.execute(
            "SELECT opportunity_id FROM active_opportunities ORDER BY rowid"
        ).fetchall()
        return tuple(row["opportunity_id"] for row in rows)

    def primary_opportunity_id(self) -> str | None:
        row = self._connection.execute(
            "SELECT value FROM planning_settings WHERE setting = 'primary_opportunity_id'"
        ).fetchone()
        return row["value"] if row else None

    def primary_selection_required(self, requirements: list[JobRequirement]) -> bool:
        """Ask for a primary only when active opportunities share no core requirement text."""
        active_ids = self.active_opportunity_ids()
        if len(active_ids) < 2 or self.primary_opportunity_id() is not None:
            return False
        core_texts_by_opportunity: dict[str, set[str]] = {}
        for opportunity_id in active_ids:
            core_texts_by_opportunity[opportunity_id] = {
                _normalize_requirement(requirement.text)
                for requirement in requirements
                if requirement.job_opportunity_id == opportunity_id
                and requirement.priority == RequirementPriority.CORE
            }
        sets = list(core_texts_by_opportunity.values())
        return not set.intersection(*sets) if sets else False

    def add_task(
        self,
        *,
        title: str,
        description: str,
        opportunity_ids: list[str],
        requirement_ids: list[str],
        estimated_hours: int,
        deadline: date | None = None,
    ) -> PreparationTask:
        active_ids = set(self.active_opportunity_ids())
        task_opportunity_ids = tuple(dict.fromkeys(opportunity_ids))
        if not title.strip() or not description.strip():
            raise ValueError("Task title and description cannot be blank.")
        if not task_opportunity_ids or not set(task_opportunity_ids).issubset(active_ids):
            raise ValueError("A task must target one or more active opportunities.")
        if not requirement_ids:
            raise ValueError("A task must link to at least one job requirement.")
        if estimated_hours <= 0:
            raise ValueError("Estimated hours must be positive.")
        task = PreparationTask(
            id=str(uuid4()),
            title=title.strip(),
            description=description.strip(),
            opportunity_ids=task_opportunity_ids,
            requirement_ids=tuple(dict.fromkeys(requirement_ids)),
            estimated_hours=estimated_hours,
            deadline=deadline,
            status=TaskStatus.PROPOSED,
        )
        self._connection.execute(
            """
            INSERT INTO preparation_tasks
                (id, title, description, estimated_hours, deadline, status)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (task.id, task.title, task.description, task.estimated_hours, _date_value(task.deadline), task.status.value),
        )
        self._connection.executemany(
            "INSERT INTO task_opportunities (task_id, opportunity_id) VALUES (?, ?)",
            [(task.id, opportunity_id) for opportunity_id in task.opportunity_ids],
        )
        self._connection.executemany(
            "INSERT INTO task_requirements (task_id, requirement_id) VALUES (?, ?)",
            [(task.id, requirement_id) for requirement_id in task.requirement_ids],
        )
        self._connection.commit()
        return task

    def submit_completion_evidence(
        self, task_id: str, *, project_evidence: dict[str, str], work_link: str | None = None
    ) -> PreparationTask:
        task = self._get_task(task_id)
        missing = [field for field in ("title", "technologies", "contribution", "result") if not project_evidence.get(field, "").strip()]
        if missing:
            raise ValueError(f"Completion evidence is missing required fields: {', '.join(missing)}.")
        if task.status != TaskStatus.PROPOSED:
            raise ValueError("Completion evidence can only be submitted once for a proposed task.")
        self._connection.execute(
            """
            INSERT INTO task_completion_evidence (task_id, project_evidence_json, work_link)
            VALUES (?, ?, ?)
            """,
            (task_id, json.dumps(project_evidence, ensure_ascii=False), work_link.strip() if work_link else None),
        )
        self._connection.execute(
            "UPDATE preparation_tasks SET status = ? WHERE id = ?",
            (TaskStatus.EVIDENCE_SUBMITTED.value, task_id),
        )
        self._connection.commit()
        return self._get_task(task_id)

    def confirm_completion(self, task_id: str) -> PreparationTask:
        task = self._get_task(task_id)
        if task.status != TaskStatus.EVIDENCE_SUBMITTED:
            raise ValueError("User confirmation requires submitted completion evidence.")
        self._connection.execute(
            "UPDATE preparation_tasks SET status = ? WHERE id = ?",
            (TaskStatus.CONFIRMED.value, task_id),
        )
        self._connection.commit()
        return self._get_task(task_id)

    def confirmed_completion_evidence(self, task_id: str) -> tuple[dict[str, str], str | None]:
        """Return completion evidence only after the user has confirmed the task."""
        task = self._get_task(task_id)
        if task.status != TaskStatus.CONFIRMED:
            raise ValueError("Only confirmed tasks can be promoted to profile evidence.")
        row = self._connection.execute(
            "SELECT project_evidence_json, work_link FROM task_completion_evidence WHERE task_id = ?",
            (task_id,),
        ).fetchone()
        if row is None:
            raise ValueError("Confirmed task has no completion evidence.")
        return json.loads(row["project_evidence_json"]), row["work_link"]

    def promoted_evidence_id(self, task_id: str, profile_id: str) -> str | None:
        row = self._connection.execute(
            "SELECT evidence_id FROM task_profile_promotions WHERE task_id = ? AND profile_id = ?",
            (task_id, profile_id),
        ).fetchone()
        return row["evidence_id"] if row else None

    def record_evidence_promotion(self, task_id: str, profile_id: str, evidence_id: str) -> None:
        try:
            self._connection.execute(
                "INSERT INTO task_profile_promotions (task_id, profile_id, evidence_id) VALUES (?, ?, ?)",
                (task_id, profile_id, evidence_id),
            )
            self._connection.commit()
        except sqlite3.IntegrityError as error:
            raise ValueError("This task has already been promoted to this profile.") from error

    def recommend_next_task(
        self, requirements: list[JobRequirement], *, weekly_hours: int, today: date | None = None
    ) -> TaskRecommendation | None:
        if weekly_hours <= 0:
            raise ValueError("Weekly hours must be positive.")
        requirement_by_id = {requirement.id: requirement for requirement in requirements}
        recommendations = [
            self._recommendation_for(task, requirement_by_id, weekly_hours, today or date.today())
            for task in self._list_tasks(status=TaskStatus.PROPOSED)
        ]
        if not recommendations:
            return None
        return max(recommendations, key=self._ranking_key)

    def _recommendation_for(
        self,
        task: PreparationTask,
        requirement_by_id: dict[str, JobRequirement],
        weekly_hours: int,
        today: date,
    ) -> TaskRecommendation:
        linked_requirements = [
            requirement_by_id[requirement_id]
            for requirement_id in task.requirement_ids
            if requirement_id in requirement_by_id
        ]
        gap_count = sum(
            requirement.evidence_assessment in {EvidenceAssessment.GAP, EvidenceAssessment.EVIDENCE_INSUFFICIENT}
            for requirement in linked_requirements
        )
        primary_bonus = self.primary_opportunity_id() in task.opportunity_ids
        feasible = task.estimated_hours <= weekly_hours
        urgency = _deadline_urgency(task.deadline, today)
        rationale = [f"关联 {len(linked_requirements)} 项岗位要求，其中 {gap_count} 项存在证据缺口。"]
        if len(task.opportunity_ids) > 1:
            rationale.append(f"可同时服务 {len(task.opportunity_ids)} 个活跃岗位。")
        if primary_bonus:
            rationale.append("直接服务当前主岗位。")
        rationale.append("可在本周完成。" if feasible else "预计超过本周可用时间。")
        if urgency != "none":
            rationale.append(f"截止日期紧迫度：{urgency}。")
        return TaskRecommendation(
            task=task,
            gap_count=gap_count,
            shared_opportunity_count=len(task.opportunity_ids),
            primary_opportunity_bonus=primary_bonus,
            feasible_in_weekly_time=feasible,
            deadline_urgency=urgency,
            rationale=tuple(rationale),
        )

    @staticmethod
    def _ranking_key(recommendation: TaskRecommendation) -> tuple[int, int, int, int, int]:
        urgency = {"none": 0, "soon": 1, "urgent": 2}[recommendation.deadline_urgency]
        return (
            recommendation.gap_count,
            recommendation.shared_opportunity_count,
            int(recommendation.primary_opportunity_bonus),
            int(recommendation.feasible_in_weekly_time),
            urgency,
        )

    def _list_tasks(self, *, status: TaskStatus) -> list[PreparationTask]:
        rows = self._connection.execute(
            "SELECT * FROM preparation_tasks WHERE status = ? ORDER BY rowid", (status.value,)
        ).fetchall()
        return [self._row_to_task(row) for row in rows]

    def _get_task(self, task_id: str) -> PreparationTask:
        row = self._connection.execute("SELECT * FROM preparation_tasks WHERE id = ?", (task_id,)).fetchone()
        if row is None:
            raise ValueError("Preparation task does not exist.")
        return self._row_to_task(row)

    def _row_to_task(self, row: sqlite3.Row) -> PreparationTask:
        opportunity_rows = self._connection.execute(
            "SELECT opportunity_id FROM task_opportunities WHERE task_id = ? ORDER BY rowid", (row["id"],)
        ).fetchall()
        requirement_rows = self._connection.execute(
            "SELECT requirement_id FROM task_requirements WHERE task_id = ? ORDER BY rowid", (row["id"],)
        ).fetchall()
        return PreparationTask(
            id=row["id"], title=row["title"], description=row["description"],
            opportunity_ids=tuple(item["opportunity_id"] for item in opportunity_rows),
            requirement_ids=tuple(item["requirement_id"] for item in requirement_rows),
            estimated_hours=row["estimated_hours"], deadline=date.fromisoformat(row["deadline"]) if row["deadline"] else None,
            status=TaskStatus(row["status"]),
        )

    def _setup(self) -> None:
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS active_opportunities (opportunity_id TEXT PRIMARY KEY);
            CREATE TABLE IF NOT EXISTS planning_settings (setting TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS preparation_tasks (
                id TEXT PRIMARY KEY, title TEXT NOT NULL, description TEXT NOT NULL,
                estimated_hours INTEGER NOT NULL, deadline TEXT, status TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS task_opportunities (
                task_id TEXT NOT NULL REFERENCES preparation_tasks(id) ON DELETE CASCADE,
                opportunity_id TEXT NOT NULL, PRIMARY KEY (task_id, opportunity_id)
            );
            CREATE TABLE IF NOT EXISTS task_requirements (
                task_id TEXT NOT NULL REFERENCES preparation_tasks(id) ON DELETE CASCADE,
                requirement_id TEXT NOT NULL, PRIMARY KEY (task_id, requirement_id)
            );
            CREATE TABLE IF NOT EXISTS task_completion_evidence (
                task_id TEXT PRIMARY KEY REFERENCES preparation_tasks(id) ON DELETE CASCADE,
                project_evidence_json TEXT NOT NULL, work_link TEXT
            );
            CREATE TABLE IF NOT EXISTS task_profile_promotions (
                task_id TEXT NOT NULL REFERENCES preparation_tasks(id) ON DELETE CASCADE,
                profile_id TEXT NOT NULL,
                evidence_id TEXT NOT NULL,
                PRIMARY KEY (task_id, profile_id)
            );
            """
        )
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.commit()


def _normalize_requirement(text: str) -> str:
    return "".join(text.lower().split())


def _date_value(value: date | None) -> str | None:
    return value.isoformat() if value else None


def _deadline_urgency(deadline: date | None, today: date) -> str:
    if deadline is None:
        return "none"
    days = (deadline - today).days
    if days <= 7:
        return "urgent"
    if days <= 21:
        return "soon"
    return "none"

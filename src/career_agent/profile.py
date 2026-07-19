"""Local, evidence-backed persistence for a user's capability profile."""
from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from uuid import uuid4


class EvidenceType(StrEnum):
    """The five kinds of experience that can support a capability claim."""

    PROJECT = "project"
    WORK = "work"
    COMPETITION = "competition"
    COURSE = "course"
    CAMPUS = "campus"


class ConfirmationStatus(StrEnum):
    """Whether a user has checked an extracted or entered record."""

    DRAFT = "draft"
    CONFIRMED = "confirmed"


_REQUIRED_DETAILS: dict[EvidenceType, tuple[str, ...]] = {
    EvidenceType.PROJECT: ("technologies", "contribution", "result"),
    EvidenceType.WORK: ("responsibilities", "work_content"),
    EvidenceType.COMPETITION: ("outcome", "contribution"),
    EvidenceType.COURSE: ("course_or_activity", "outcome", "related_work"),
    EvidenceType.CAMPUS: ("responsibilities", "developed_capabilities"),
}


@dataclass(frozen=True)
class CapabilityProfile:
    id: str


@dataclass(frozen=True)
class SourceDocument:
    id: str
    profile_id: str
    filename: str
    source_path: str


@dataclass(frozen=True)
class EvidenceRecord:
    id: str
    profile_id: str
    evidence_type: EvidenceType
    title: str
    details: dict[str, str]
    confirmation_status: ConfirmationStatus
    source_document_id: str | None


@dataclass(frozen=True)
class CapabilityClaim:
    id: str
    profile_id: str
    statement: str
    evidence_record_ids: tuple[str, ...]


class CapabilityProfileStore:
    """A small SQLite repository that preserves the profile evidence boundary."""

    def __init__(
        self,
        database_path: Path,
        on_evidence_confirmed: Callable[[EvidenceRecord], None] | None = None,
        on_evidence_deleted: Callable[[EvidenceRecord], None] | None = None,
    ) -> None:
        database_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(database_path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._on_evidence_confirmed = on_evidence_confirmed
        self._on_evidence_deleted = on_evidence_deleted
        self._setup()

    def close(self) -> None:
        self._connection.close()

    def create_profile(self) -> CapabilityProfile:
        profile = CapabilityProfile(id=str(uuid4()))
        self._connection.execute("INSERT INTO capability_profiles (id) VALUES (?)", (profile.id,))
        self._connection.commit()
        return profile

    def add_source_document(
        self, profile_id: str, *, filename: str, source_path: str
    ) -> SourceDocument:
        """Record an original file reference; source documents deliberately have no update API."""
        self._require_profile(profile_id)
        if not filename.strip() or not source_path.strip():
            raise ValueError("Source document filename and path cannot be blank.")
        document = SourceDocument(
            id=str(uuid4()),
            profile_id=profile_id,
            filename=filename.strip(),
            source_path=source_path.strip(),
        )
        self._connection.execute(
            """
            INSERT INTO source_documents (id, profile_id, filename, source_path)
            VALUES (?, ?, ?, ?)
            """,
            (document.id, document.profile_id, document.filename, document.source_path),
        )
        self._connection.commit()
        return document

    def add_evidence_record(
        self,
        profile_id: str,
        *,
        evidence_type: EvidenceType,
        title: str,
        details: dict[str, str],
        confirmation_status: ConfirmationStatus = ConfirmationStatus.DRAFT,
        source_document_id: str | None = None,
    ) -> EvidenceRecord:
        self._require_profile(profile_id)
        if not title.strip():
            raise ValueError("Evidence title cannot be blank.")
        if confirmation_status == ConfirmationStatus.CONFIRMED:
            self._validate_details(evidence_type, details)
        if source_document_id is not None:
            source_document = self._get_source_document(source_document_id)
            if source_document.profile_id != profile_id:
                raise ValueError("An evidence record must reference its own profile's source document.")

        record = EvidenceRecord(
            id=str(uuid4()),
            profile_id=profile_id,
            evidence_type=evidence_type,
            title=title.strip(),
            details=dict(details),
            confirmation_status=confirmation_status,
            source_document_id=source_document_id,
        )
        self._connection.execute(
            """
            INSERT INTO evidence_records
                (id, profile_id, evidence_type, title, details_json, confirmation_status, source_document_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.id,
                record.profile_id,
                record.evidence_type.value,
                record.title,
                json.dumps(record.details, ensure_ascii=False),
                record.confirmation_status.value,
                record.source_document_id,
            ),
        )
        self._connection.commit()
        if confirmation_status == ConfirmationStatus.CONFIRMED and self._on_evidence_confirmed is not None:
            self._on_evidence_confirmed(record)
        return record

    def confirm_evidence_record(self, evidence_record_id: str) -> EvidenceRecord:
        record = self._get_evidence_record(evidence_record_id)
        self._validate_details(record.evidence_type, record.details)
        self._connection.execute(
            "UPDATE evidence_records SET confirmation_status = ? WHERE id = ?",
            (ConfirmationStatus.CONFIRMED.value, evidence_record_id),
        )
        self._connection.commit()
        confirmed = self._get_evidence_record(evidence_record_id)
        if self._on_evidence_confirmed is not None:
            self._on_evidence_confirmed(confirmed)
        return confirmed

    def update_draft_evidence_record(
        self,
        evidence_record_id: str,
        *,
        title: str,
        details: dict[str, str],
        evidence_type: EvidenceType | None = None,
    ) -> EvidenceRecord:
        """Edit an extracted draft before the user confirms it."""
        record = self._get_evidence_record(evidence_record_id)
        if record.confirmation_status != ConfirmationStatus.DRAFT:
            raise ValueError("Only draft evidence records can be edited in initialization.")
        if not title.strip():
            raise ValueError("Evidence title cannot be blank.")
        selected_type = evidence_type or record.evidence_type
        self._connection.execute(
            """
            UPDATE evidence_records
            SET evidence_type = ?, title = ?, details_json = ?
            WHERE id = ?
            """,
            (
                selected_type.value,
                title.strip(),
                json.dumps(details, ensure_ascii=False),
                evidence_record_id,
            ),
        )
        self._connection.commit()
        return self._get_evidence_record(evidence_record_id)

    def discard_draft_evidence_record(self, evidence_record_id: str) -> None:
        """Remove an extracted draft the user decides not to retain."""
        record = self._get_evidence_record(evidence_record_id)
        if record.confirmation_status != ConfirmationStatus.DRAFT:
            raise ValueError("Only draft evidence records can be discarded.")
        self._connection.execute("DELETE FROM evidence_records WHERE id = ?", (evidence_record_id,))
        self._connection.commit()

    def delete_evidence_record(self, evidence_record_id: str) -> None:
        """Delete a user-selected record; confirmed records also leave the local index."""
        record = self._get_evidence_record(evidence_record_id)
        if record.confirmation_status == ConfirmationStatus.CONFIRMED and self._on_evidence_deleted is not None:
            self._on_evidence_deleted(record)
        self._connection.execute("DELETE FROM evidence_records WHERE id = ?", (evidence_record_id,))
        self._connection.commit()

    def list_draft_evidence(self, profile_id: str) -> list[EvidenceRecord]:
        self._require_profile(profile_id)
        rows = self._connection.execute(
            """
            SELECT * FROM evidence_records
            WHERE profile_id = ? AND confirmation_status = ?
            ORDER BY rowid
            """,
            (profile_id, ConfirmationStatus.DRAFT.value),
        ).fetchall()
        return [self._row_to_evidence(row) for row in rows]

    def list_confirmed_evidence(self, profile_id: str) -> list[EvidenceRecord]:
        self._require_profile(profile_id)
        rows = self._connection.execute(
            """
            SELECT * FROM evidence_records
            WHERE profile_id = ? AND confirmation_status = ?
            ORDER BY rowid
            """,
            (profile_id, ConfirmationStatus.CONFIRMED.value),
        ).fetchall()
        return [self._row_to_evidence(row) for row in rows]

    def create_capability_claim(
        self, profile_id: str, *, statement: str, evidence_record_ids: list[str]
    ) -> CapabilityClaim:
        self._require_profile(profile_id)
        if not statement.strip():
            raise ValueError("Capability claim statement cannot be blank.")
        if not evidence_record_ids:
            raise ValueError("A capability claim needs at least one evidence record.")

        unique_evidence_ids = tuple(dict.fromkeys(evidence_record_ids))
        for evidence_id in unique_evidence_ids:
            record = self._get_evidence_record(evidence_id)
            if record.profile_id != profile_id:
                raise ValueError("Capability claim evidence must belong to the same profile.")
            if record.confirmation_status != ConfirmationStatus.CONFIRMED:
                raise ValueError("Only confirmed evidence can support a capability claim.")

        claim = CapabilityClaim(
            id=str(uuid4()),
            profile_id=profile_id,
            statement=statement.strip(),
            evidence_record_ids=unique_evidence_ids,
        )
        self._connection.execute(
            "INSERT INTO capability_claims (id, profile_id, statement) VALUES (?, ?, ?)",
            (claim.id, claim.profile_id, claim.statement),
        )
        self._connection.executemany(
            """
            INSERT INTO capability_claim_evidence (claim_id, evidence_record_id)
            VALUES (?, ?)
            """,
            [(claim.id, evidence_id) for evidence_id in claim.evidence_record_ids],
        )
        self._connection.commit()
        return claim

    def _setup(self) -> None:
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS capability_profiles (
                id TEXT PRIMARY KEY
            );
            CREATE TABLE IF NOT EXISTS source_documents (
                id TEXT PRIMARY KEY,
                profile_id TEXT NOT NULL REFERENCES capability_profiles(id),
                filename TEXT NOT NULL,
                source_path TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS evidence_records (
                id TEXT PRIMARY KEY,
                profile_id TEXT NOT NULL REFERENCES capability_profiles(id),
                evidence_type TEXT NOT NULL,
                title TEXT NOT NULL,
                details_json TEXT NOT NULL,
                confirmation_status TEXT NOT NULL,
                source_document_id TEXT REFERENCES source_documents(id)
            );
            CREATE TABLE IF NOT EXISTS capability_claims (
                id TEXT PRIMARY KEY,
                profile_id TEXT NOT NULL REFERENCES capability_profiles(id),
                statement TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS capability_claim_evidence (
                claim_id TEXT NOT NULL REFERENCES capability_claims(id),
                evidence_record_id TEXT NOT NULL REFERENCES evidence_records(id),
                PRIMARY KEY (claim_id, evidence_record_id)
            );
            """
        )
        self._connection.commit()

    def _require_profile(self, profile_id: str) -> None:
        exists = self._connection.execute(
            "SELECT 1 FROM capability_profiles WHERE id = ?", (profile_id,)
        ).fetchone()
        if exists is None:
            raise ValueError("Capability profile does not exist.")

    def _get_source_document(self, source_document_id: str) -> SourceDocument:
        row = self._connection.execute(
            "SELECT * FROM source_documents WHERE id = ?", (source_document_id,)
        ).fetchone()
        if row is None:
            raise ValueError("Source document does not exist.")
        return SourceDocument(
            id=row["id"],
            profile_id=row["profile_id"],
            filename=row["filename"],
            source_path=row["source_path"],
        )

    def _get_evidence_record(self, evidence_record_id: str) -> EvidenceRecord:
        row = self._connection.execute(
            "SELECT * FROM evidence_records WHERE id = ?", (evidence_record_id,)
        ).fetchone()
        if row is None:
            raise ValueError("Evidence record does not exist.")
        return self._row_to_evidence(row)

    @staticmethod
    def _validate_details(evidence_type: EvidenceType, details: dict[str, str]) -> None:
        missing = [
            field for field in _REQUIRED_DETAILS[evidence_type] if not details.get(field, "").strip()
        ]
        if missing:
            raise ValueError(
                f"{evidence_type.value} evidence is missing required fields: {', '.join(missing)}."
            )

    @staticmethod
    def _row_to_evidence(row: sqlite3.Row) -> EvidenceRecord:
        return EvidenceRecord(
            id=row["id"],
            profile_id=row["profile_id"],
            evidence_type=EvidenceType(row["evidence_type"]),
            title=row["title"],
            details=json.loads(row["details_json"]),
            confirmation_status=ConfirmationStatus(row["confirmation_status"]),
            source_document_id=row["source_document_id"],
        )

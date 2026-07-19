"""Managed documents and safe automatic rebuilding of the search index."""
from __future__ import annotations

import json
import shutil
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from uuid import uuid4

from .profile import CapabilityProfileStore, EvidenceRecord
from .rag import build_index, clear_active_index

IndexBuilder = Callable[[Path, Path], int]


class IndexStatus(StrEnum):
    READY = "ready"
    FAILED = "failed"


@dataclass(frozen=True)
class ManagedDocument:
    id: str
    filename: str
    managed_path: str
    index_status: IndexStatus
    last_error: str | None


class DocumentManager:
    """Own documents and switch indexes only after a replacement has been built."""

    def __init__(
        self, database_path: Path, documents_dir: Path, index_dir: Path, index_builder: IndexBuilder = build_index
    ) -> None:
        database_path.parent.mkdir(parents=True, exist_ok=True)
        documents_dir.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(database_path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self.documents_dir = documents_dir
        self.index_dir = index_dir
        self._index_builder = index_builder
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS managed_documents (
                id TEXT PRIMARY KEY, filename TEXT NOT NULL, managed_path TEXT NOT NULL,
                index_status TEXT NOT NULL, last_error TEXT
            )
            """
        )
        self._connection.commit()

    def close(self) -> None:
        self._connection.close()

    def upload(self, source_path: Path) -> ManagedDocument:
        if not source_path.is_file():
            raise ValueError("Uploaded document does not exist.")
        document_id = str(uuid4())
        destination = self.documents_dir / document_id / source_path.name
        destination.parent.mkdir(parents=True)
        shutil.copy2(source_path, destination)
        document = ManagedDocument(document_id, source_path.name, destination.as_posix(), IndexStatus.READY, None)
        self._insert(document)
        return self._rebuild_and_store_status(document)

    def replace(self, document_id: str, source_path: Path) -> ManagedDocument:
        document = self.get(document_id)
        if not source_path.is_file():
            raise ValueError("Replacement document does not exist.")
        destination = Path(document.managed_path)
        backup = destination.with_name(f"{destination.name}.backup-{uuid4().hex}")
        destination.rename(backup)
        try:
            shutil.copy2(source_path, destination)
            success, error = self._rebuild()
            if not success:
                destination.unlink(missing_ok=True)
                backup.rename(destination)
                return self._set_status(document, IndexStatus.FAILED, error)
            backup.unlink(missing_ok=True)
            replacement = ManagedDocument(document.id, source_path.name, destination.as_posix(), IndexStatus.READY, None)
            self._connection.execute("UPDATE managed_documents SET filename = ?, managed_path = ?, index_status = ?, last_error = NULL WHERE id = ?", (replacement.filename, replacement.managed_path, IndexStatus.READY.value, document_id))
            self._connection.commit()
            return replacement
        except Exception:
            if not destination.exists() and backup.exists():
                backup.rename(destination)
            raise

    def delete(self, document_id: str) -> None:
        document = self.get(document_id)
        path = Path(document.managed_path)
        backup = path.with_name(f"{path.name}.backup-{uuid4().hex}")
        path.rename(backup)
        if not _has_indexable_documents(self.documents_dir):
            clear_active_index(self.index_dir)
            backup.unlink(missing_ok=True)
            backup.parent.rmdir()
            self._connection.execute("DELETE FROM managed_documents WHERE id = ?", (document_id,))
            self._connection.commit()
            return
        success, error = self._rebuild()
        if not success:
            backup.rename(path)
            self._set_status(document, IndexStatus.FAILED, error)
            raise RuntimeError(error)
        backup.unlink(missing_ok=True)
        backup.parent.rmdir()
        self._connection.execute("DELETE FROM managed_documents WHERE id = ?", (document_id,))
        self._connection.commit()

    def get(self, document_id: str) -> ManagedDocument:
        row = self._connection.execute("SELECT * FROM managed_documents WHERE id = ?", (document_id,)).fetchone()
        if row is None:
            raise ValueError("Managed document does not exist.")
        return self._row_to_document(row)

    def list_documents(self) -> list[ManagedDocument]:
        rows = self._connection.execute("SELECT * FROM managed_documents ORDER BY rowid").fetchall()
        return [self._row_to_document(row) for row in rows]

    def confirmed_evidence_callback(self, evidence_dir: Path) -> Callable[[EvidenceRecord], None]:
        """Callback for CapabilityProfileStore: make each confirmed record searchable."""
        def index_confirmed_evidence(record: EvidenceRecord) -> None:
            evidence_path = evidence_dir / record.profile_id / f"{record.id}.md"
            evidence_path.parent.mkdir(parents=True, exist_ok=True)
            evidence_path.write_text(
                "\n".join([f"# {record.title}", f"Type: {record.evidence_type.value}", json.dumps(record.details, ensure_ascii=False)]),
                encoding="utf-8",
            )
            success, error = self._rebuild()
            if not success:
                raise RuntimeError(f"Evidence was confirmed but index rebuild failed: {error}")
        return index_confirmed_evidence

    def deleted_evidence_callback(self, evidence_dir: Path) -> Callable[[EvidenceRecord], None]:
        """Remove a confirmed evidence document before rebuilding the active index."""
        def remove_confirmed_evidence(record: EvidenceRecord) -> None:
            evidence_path = evidence_dir / record.profile_id / f"{record.id}.md"
            evidence_path.unlink(missing_ok=True)
            success, error = self._rebuild()
            if not success:
                raise RuntimeError(f"Evidence was deleted but index rebuild failed: {error}")
        return remove_confirmed_evidence

    def _rebuild_and_store_status(self, document: ManagedDocument) -> ManagedDocument:
        success, error = self._rebuild()
        return self._set_status(document, IndexStatus.READY if success else IndexStatus.FAILED, error)

    def _rebuild(self) -> tuple[bool, str | None]:
        try:
            self._index_builder(self.documents_dir, self.index_dir)
        except Exception as error:
            return False, str(error)
        return True, None

    def _insert(self, document: ManagedDocument) -> None:
        self._connection.execute("INSERT INTO managed_documents (id, filename, managed_path, index_status, last_error) VALUES (?, ?, ?, ?, ?)", (document.id, document.filename, document.managed_path, document.index_status.value, document.last_error))
        self._connection.commit()

    def _set_status(self, document: ManagedDocument, status: IndexStatus, error: str | None) -> ManagedDocument:
        self._connection.execute("UPDATE managed_documents SET index_status = ?, last_error = ? WHERE id = ?", (status.value, error, document.id))
        self._connection.commit()
        return ManagedDocument(document.id, document.filename, document.managed_path, status, error)

    @staticmethod
    def _row_to_document(row: sqlite3.Row) -> ManagedDocument:
        return ManagedDocument(row["id"], row["filename"], row["managed_path"], IndexStatus(row["index_status"]), row["last_error"])


def create_profile_store_with_auto_index(database_path: Path, manager: DocumentManager) -> CapabilityProfileStore:
    """Wire profile confirmation to the same managed-document index."""
    evidence_dir = manager.documents_dir / "confirmed-evidence"
    return CapabilityProfileStore(
        database_path,
        on_evidence_confirmed=manager.confirmed_evidence_callback(evidence_dir),
        on_evidence_deleted=manager.deleted_evidence_callback(evidence_dir),
    )


def _has_indexable_documents(documents_dir: Path) -> bool:
    return any(
        path.is_file() and path.suffix.lower() in {".md", ".txt", ".docx"}
        for path in documents_dir.rglob("*")
    )

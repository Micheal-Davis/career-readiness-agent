import pytest

from career_agent.documents import DocumentManager, IndexStatus, create_profile_store_with_auto_index
from career_agent.profile import EvidenceType


def recording_builder(calls):
    def build(documents_dir, index_dir):
        calls.append(
            sorted(
                path.relative_to(documents_dir).as_posix()
                for path in documents_dir.rglob("*")
                if path.is_file()
            )
        )
        index_dir.mkdir(parents=True, exist_ok=True)
        (index_dir / "marker.txt").write_text("rebuilt", encoding="utf-8")
        return 1
    return build


@pytest.fixture
def paths(tmp_path):
    return tmp_path / "documents.sqlite3", tmp_path / "documents", tmp_path / "chroma", tmp_path


def test_upload_replace_and_delete_rebuild_automatically(paths):
    database_path, documents_dir, index_dir, tmp_path = paths
    calls = []
    manager = DocumentManager(database_path, documents_dir, index_dir, recording_builder(calls))
    first = tmp_path / "first.txt"
    first.write_text("first", encoding="utf-8")
    second = tmp_path / "second.txt"
    second.write_text("second", encoding="utf-8")

    uploaded = manager.upload(first)
    replaced = manager.replace(uploaded.id, second)
    manager.delete(replaced.id)

    assert uploaded.index_status == IndexStatus.READY
    assert replaced.filename == "second.txt"
    # Deleting the final document safely clears the active index rather than
    # attempting to build an invalid empty Chroma collection.
    assert len(calls) == 2
    assert manager.list_documents() == []
    manager.close()


def test_failed_rebuild_preserves_current_index_and_reports_status(paths):
    database_path, documents_dir, index_dir, tmp_path = paths
    index_dir.mkdir()
    marker = index_dir / "current-index.txt"
    marker.write_text("still usable", encoding="utf-8")
    source = tmp_path / "resume.txt"
    source.write_text("resume", encoding="utf-8")
    def fail(*_):
        raise RuntimeError("index unavailable")
    manager = DocumentManager(database_path, documents_dir, index_dir, fail)

    uploaded = manager.upload(source)

    assert uploaded.index_status == IndexStatus.FAILED
    assert uploaded.last_error == "index unavailable"
    assert marker.read_text(encoding="utf-8") == "still usable"
    manager.close()


def test_confirmed_evidence_rebuilds_without_manual_ingest(paths):
    database_path, documents_dir, index_dir, _ = paths
    calls = []
    manager = DocumentManager(database_path, documents_dir, index_dir, recording_builder(calls))
    store = create_profile_store_with_auto_index(database_path.with_name("profiles.sqlite3"), manager)
    profile = store.create_profile()
    draft = store.add_evidence_record(
        profile.id,
        evidence_type=EvidenceType.PROJECT,
        title="Career Agent",
        details={"technologies": "Python", "contribution": "Built workflow", "result": "Working prototype"},
    )

    store.confirm_evidence_record(draft.id)

    assert calls == [[f"confirmed-evidence/{profile.id}/{draft.id}.md"]]
    store.close()
    manager.close()

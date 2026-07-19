from langchain_chroma import Chroma

from career_agent.embeddings import HashEmbeddings
from career_agent.rag import active_index_directory, build_index
import pytest


def test_build_index_replaces_removed_document_chunks(tmp_path):
    documents_dir = tmp_path / "documents"
    persist_dir = tmp_path / "chroma"
    documents_dir.mkdir()

    old_document = documents_dir / "old.txt"
    old_document.write_text("obsolete career detail", encoding="utf-8")
    build_index(documents_dir, persist_dir)

    old_document.unlink()
    (documents_dir / "current.txt").write_text(
        "current career detail",
        encoding="utf-8",
    )
    build_index(documents_dir, persist_dir)

    store = Chroma(
        collection_name="career_knowledge",
        persist_directory=str(active_index_directory(persist_dir)),
        embedding_function=HashEmbeddings(),
    )
    indexed_documents = store.get(include=["documents"])["documents"]

    assert indexed_documents == ["current career detail"]


def test_build_index_records_source_path_and_chunk_number(tmp_path):
    documents_dir = tmp_path / "documents"
    persist_dir = tmp_path / "chroma"
    nested_dir = documents_dir / "projects"
    nested_dir.mkdir(parents=True)
    (nested_dir / "resume.txt").write_text("career experience", encoding="utf-8")

    build_index(documents_dir, persist_dir)

    store = Chroma(
        collection_name="career_knowledge",
        persist_directory=str(active_index_directory(persist_dir)),
        embedding_function=HashEmbeddings(),
    )
    metadata = store.get(include=["metadatas"])["metadatas"]

    assert metadata == [
        {
            "source": "resume.txt",
            "source_path": "projects/resume.txt",
            "chunk_index": 1,
        }
    ]


def test_failed_rebuild_keeps_previous_index(tmp_path, monkeypatch):
    documents_dir = tmp_path / "documents"
    persist_dir = tmp_path / "chroma"
    documents_dir.mkdir()
    (documents_dir / "old.txt").write_text("old index", encoding="utf-8")
    build_index(documents_dir, persist_dir)

    def fail(*args, **kwargs):
        raise RuntimeError("cannot create replacement")

    monkeypatch.setattr("career_agent.rag.Chroma.from_documents", fail)
    (documents_dir / "new.txt").write_text("new index", encoding="utf-8")

    with pytest.raises(RuntimeError, match="cannot create replacement"):
        build_index(documents_dir, persist_dir)

    store = Chroma(
        collection_name="career_knowledge",
        persist_directory=str(active_index_directory(persist_dir)),
        embedding_function=HashEmbeddings(),
    )
    assert store.get(include=["documents"])["documents"] == ["old index"]

import pytest

from career_agent.profile import (
    CapabilityProfileStore,
    ConfirmationStatus,
    EvidenceType,
)


DETAILS_BY_TYPE = {
    EvidenceType.PROJECT: {
        "technologies": "Python, FastAPI",
        "contribution": "Designed the API",
        "result": "Shipped a working prototype",
    },
    EvidenceType.WORK: {
        "responsibilities": "Built data pipeline",
        "work_content": "Reduced manual work",
    },
    EvidenceType.COMPETITION: {
        "organizer": "Mathematical Contest in Modeling",
        "outcome": "Second prize",
        "topic": "Operations research",
        "contribution": "Built the model",
    },
    EvidenceType.COURSE: {
        "course_or_activity": "Database systems",
        "outcome": "Completed database design project",
        "related_work": "Designed a schema and queries",
    },
    EvidenceType.CAMPUS: {
        "responsibilities": "Organized activities",
        "developed_capabilities": "Communication and coordination",
    },
}


@pytest.fixture
def store(tmp_path):
    database = CapabilityProfileStore(tmp_path / "profiles.sqlite3")
    yield database
    database.close()


@pytest.mark.parametrize("evidence_type", list(EvidenceType))
def test_all_evidence_types_keep_their_required_details(store, evidence_type):
    profile = store.create_profile()

    record = store.add_evidence_record(
        profile.id,
        evidence_type=evidence_type,
        title=f"My {evidence_type.value} experience",
        details=DETAILS_BY_TYPE[evidence_type],
        confirmation_status=ConfirmationStatus.CONFIRMED,
    )

    assert store.list_confirmed_evidence(profile.id) == [record]


def test_type_specific_required_fields_are_validated(store):
    profile = store.create_profile()
    incomplete_details = dict(DETAILS_BY_TYPE[EvidenceType.COMPETITION])
    incomplete_details.pop("outcome")

    with pytest.raises(ValueError, match="outcome"):
        store.add_evidence_record(
            profile.id,
            evidence_type=EvidenceType.COMPETITION,
            title="Modeling contest",
            details=incomplete_details,
            confirmation_status=ConfirmationStatus.CONFIRMED,
        )


def test_source_document_is_referenced_without_being_mutated(store):
    profile = store.create_profile()
    document = store.add_source_document(
        profile.id,
        filename="original-resume.pdf",
        source_path="uploads/original-resume.pdf",
    )

    record = store.add_evidence_record(
        profile.id,
        evidence_type=EvidenceType.PROJECT,
        title="Career agent",
        details=DETAILS_BY_TYPE[EvidenceType.PROJECT],
        source_document_id=document.id,
    )

    assert record.source_document_id == document.id
    assert not hasattr(store, "update_source_document")


def test_unconfirmed_evidence_is_excluded_and_cannot_support_claims(store):
    profile = store.create_profile()
    draft = store.add_evidence_record(
        profile.id,
        evidence_type=EvidenceType.PROJECT,
        title="Imported project draft",
        details=DETAILS_BY_TYPE[EvidenceType.PROJECT],
    )

    assert store.list_confirmed_evidence(profile.id) == []
    with pytest.raises(ValueError, match="Only confirmed evidence"):
        store.create_capability_claim(
            profile.id,
            statement="Can design an API",
            evidence_record_ids=[draft.id],
        )

    confirmed = store.confirm_evidence_record(draft.id)
    claim = store.create_capability_claim(
        profile.id,
        statement="Can design an API",
        evidence_record_ids=[confirmed.id],
    )

    assert claim.evidence_record_ids == (confirmed.id,)


def test_claim_rejects_evidence_from_another_profile(store):
    first_profile = store.create_profile()
    second_profile = store.create_profile()
    evidence = store.add_evidence_record(
        first_profile.id,
        evidence_type=EvidenceType.PROJECT,
        title="Career agent",
        details=DETAILS_BY_TYPE[EvidenceType.PROJECT],
        confirmation_status=ConfirmationStatus.CONFIRMED,
    )

    with pytest.raises(ValueError, match="same profile"):
        store.create_capability_claim(
            second_profile.id,
            statement="Can design an API",
            evidence_record_ids=[evidence.id],
        )

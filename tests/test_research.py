import pytest

from career_agent.research import (
    ConsentScope,
    ConsentStore,
    ControlledResearchService,
    SourceTier,
)


@pytest.fixture
def consents(tmp_path):
    store = ConsentStore(tmp_path / "consents.sqlite3")
    yield store
    store.close()


def test_model_and_web_consent_are_separate_and_revocable(consents):
    assert not consents.granted(ConsentScope.MODEL_ANALYSIS)
    assert not consents.granted(ConsentScope.WEB_RESEARCH)

    consents.set(ConsentScope.MODEL_ANALYSIS, granted=True)
    consents.set(ConsentScope.WEB_RESEARCH, granted=True)
    consents.set(ConsentScope.MODEL_ANALYSIS, granted=False)

    assert not consents.granted(ConsentScope.MODEL_ANALYSIS)
    assert consents.granted(ConsentScope.WEB_RESEARCH)


def test_web_research_requires_consent_and_labels_source_tiers(consents):
    received_queries = []
    service = ControlledResearchService(
        consents,
        lambda query: received_queries.append(query) or [
            {"title": "Company careers", "url": "https://careers.example.com/jobs", "summary": "Official role"},
            {"title": "Community analysis", "url": "https://forum.example.net/post", "summary": "Discussion"},
            {"title": "Official GitHub", "url": "https://github.com/example-org/project", "summary": "Repository"},
        ],
    )

    with pytest.raises(PermissionError, match="web_research"):
        service.research("backend role")

    consents.set(ConsentScope.WEB_RESEARCH, granted=True)
    sources = service.research(
        "backend role",
        first_party_domains=["example.com"],
        first_party_url_prefixes=["https://github.com/example-org/"],
    )

    assert received_queries == ["backend role"]
    assert [source.tier for source in sources] == [SourceTier.FIRST_PARTY, SourceTier.SUPPLEMENTARY, SourceTier.FIRST_PARTY]
    assert all(source.url.startswith("http") for source in sources)


def test_research_service_only_exposes_query_to_search_provider(consents):
    received = []
    consents.set(ConsentScope.WEB_RESEARCH, granted=True)
    service = ControlledResearchService(consents, lambda query: received.append(query) or [])

    service.research("DeepSeek 后端岗位研究")

    assert received == ["DeepSeek 后端岗位研究"]


def test_research_endpoint_requires_consent_and_returns_labelled_sources(tmp_path):
    from fastapi.testclient import TestClient
    from career_agent.api import create_app

    consents = ConsentStore(tmp_path / "consents.sqlite3")
    client = TestClient(
        create_app(
            tmp_path,
            lambda *_: None,
            consents,
            web_search=lambda query: [
                {"title": "Official careers", "url": "https://careers.example.com/role", "summary": "Official role"},
                {"title": "Community", "url": "https://forum.example.net/post", "summary": "Discussion"},
            ],
        )
    )

    denied = client.post("/research", json={"query": "backend role", "first_party_domains": ["example.com"]})
    consents.set(ConsentScope.WEB_RESEARCH, granted=True)
    allowed = client.post("/research", json={"query": "backend role", "first_party_domains": ["example.com"]})

    assert denied.status_code == 403
    assert allowed.status_code == 200
    assert [item["tier"] for item in allowed.json()] == ["first_party", "supplementary"]
    consents.close()


def test_research_sources_can_be_saved_to_and_read_from_a_job(tmp_path):
    from fastapi.testclient import TestClient
    from career_agent.api import create_app

    consents = ConsentStore(tmp_path / "consents.sqlite3")
    consents.set(ConsentScope.WEB_RESEARCH, granted=True)
    client = TestClient(
        create_app(
            tmp_path,
            lambda *_: None,
            consents,
            web_search=lambda _: [
                {"title": "Official careers", "url": "https://careers.example.com/role", "summary": "Official role"},
            ],
        )
    )
    opportunity = client.post("/jobs/jd", json={"jd_text": "后端工程师"}).json()[0]

    saved = client.post(
        "/research",
        json={
            "query": "后端工程师",
            "first_party_domains": ["example.com"],
            "job_opportunity_id": opportunity["id"],
        },
    )
    stored = client.get(f"/jobs/{opportunity['id']}/research-sources")

    assert saved.status_code == 200
    assert stored.status_code == 200
    assert stored.json() == saved.json()
    consents.close()

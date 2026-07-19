"""Explicit consent controls and source-labelled web research boundaries."""
from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from urllib.parse import urlparse

import requests


class ConsentScope(StrEnum):
    LOCAL_PROCESSING = "local_processing"
    MODEL_ANALYSIS = "model_analysis"
    WEB_RESEARCH = "web_research"


class SourceTier(StrEnum):
    FIRST_PARTY = "first_party"
    SUPPLEMENTARY = "supplementary"


@dataclass(frozen=True)
class ResearchSource:
    title: str
    url: str
    summary: str
    tier: SourceTier


class ConsentStore:
    """Persist separately revocable consent choices for the local user."""

    def __init__(self, database_path: Path) -> None:
        database_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(database_path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute(
            "CREATE TABLE IF NOT EXISTS consents (scope TEXT PRIMARY KEY, granted INTEGER NOT NULL)"
        )
        self._connection.commit()

    def close(self) -> None:
        self._connection.close()

    def set(self, scope: ConsentScope, *, granted: bool) -> None:
        self._connection.execute(
            """
            INSERT INTO consents (scope, granted) VALUES (?, ?)
            ON CONFLICT(scope) DO UPDATE SET granted = excluded.granted
            """,
            (scope.value, int(granted)),
        )
        self._connection.commit()

    def granted(self, scope: ConsentScope) -> bool:
        row = self._connection.execute("SELECT granted FROM consents WHERE scope = ?", (scope.value,)).fetchone()
        return bool(row["granted"]) if row else False

    def all(self) -> dict[ConsentScope, bool]:
        return {scope: self.granted(scope) for scope in ConsentScope}

    def require(self, scope: ConsentScope) -> None:
        if not self.granted(scope):
            raise PermissionError(f"Consent required for {scope.value}.")


class ControlledResearchService:
    """Run query-only web research after consent, returning labelled public sources."""

    def __init__(self, consent_store: ConsentStore, web_search: Callable[[str], Iterable[dict[str, str]]]) -> None:
        self._consent_store = consent_store
        self._web_search = web_search

    def research(
        self,
        query: str,
        *,
        first_party_domains: Iterable[str] = (),
        first_party_url_prefixes: Iterable[str] = (),
    ) -> list[ResearchSource]:
        self._consent_store.require(ConsentScope.WEB_RESEARCH)
        if not query.strip():
            raise ValueError("Research query cannot be blank.")
        domains = {domain.lower().lstrip(".") for domain in first_party_domains}
        prefixes = tuple(prefix.lower() for prefix in first_party_url_prefixes)
        sources = []
        for result in self._web_search(query.strip()):
            url = result["url"]
            sources.append(
                ResearchSource(
                    title=result["title"],
                    url=url,
                    summary=result.get("summary", ""),
                    tier=_source_tier(url, domains, prefixes),
                )
            )
        return sources


def public_web_search(query: str) -> list[dict[str, str]]:
    """Run a query-only public search; callers never pass user evidence here."""
    try:
        response = requests.get(
            "https://api.duckduckgo.com/",
            params={"q": query, "format": "json", "no_html": "1", "no_redirect": "1"},
            timeout=15,
        )
        response.raise_for_status()
    except requests.RequestException as error:
        raise RuntimeError("联网研究暂时不可用，请检查网络后重试。") from error
    payload = response.json()
    results = []
    if payload.get("AbstractURL"):
        results.append({"title": payload.get("Heading") or query, "url": payload["AbstractURL"], "summary": payload.get("AbstractText", "")})
    for item in payload.get("RelatedTopics", []):
        if "FirstURL" in item:
            results.append({"title": item.get("Text", query), "url": item["FirstURL"], "summary": item.get("Text", "")})
    return results[:8]


def _source_tier(url: str, first_party_domains: set[str], first_party_url_prefixes: tuple[str, ...]) -> SourceTier:
    normalized_url = url.lower()
    hostname = (urlparse(url).hostname or "").lower()
    if normalized_url.startswith(first_party_url_prefixes):
        return SourceTier.FIRST_PARTY
    if any(hostname == domain or hostname.endswith(f".{domain}") for domain in first_party_domains):
        return SourceTier.FIRST_PARTY
    return SourceTier.SUPPLEMENTARY

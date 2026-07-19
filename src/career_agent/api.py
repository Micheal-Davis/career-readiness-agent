"""FastAPI application exposing the Career Knowledge Agent."""
from __future__ import annotations

import re
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field, field_validator

from .agent import create_career_agent
from .analysis import JobAnalysis, JobAnalyzer, ResumeExtractor, create_model_job_analyzer, create_model_resume_extractor
from .documents import DocumentManager, IndexStatus, ManagedDocument, create_profile_store_with_auto_index
from .errors import friendly_error_message
from .initialization import initialize_empty_profile, initialize_profile_from_resume
from .jobs import (
    EvidenceAssessment,
    JobConfirmationStatus,
    JobOpportunity,
    JobOpportunityStore,
    JobResearchSource,
    JobRequirement,
    RequirementCategory,
    RequirementDepth,
    RequirementOrigin,
    RequirementPriority,
)
from .profile import CapabilityProfileStore, ConfirmationStatus, EvidenceRecord, EvidenceType
from .planning import PreparationTask, PreparationTaskStore, TaskRecommendation, TaskStatus
from .research import ControlledResearchService, ConsentScope, ConsentStore, ResearchSource, public_web_search
from .resume import build_resume_suggestions

_SOURCE_PATTERN = re.compile(
    r"^\[来源：(.*?)｜路径：(.*?)｜切片：(\d+)\]$",
    re.MULTILINE,
)
AgentFactory = Callable[[Path, Path], Any]


class ChatRequest(BaseModel):
    """A single message sent to a conversation thread."""

    message: str = Field(max_length=10_000)
    thread_id: str | None = Field(default=None, max_length=128)
    profile_id: str | None = Field(default=None, max_length=128)

    @field_validator("message")
    @classmethod
    def message_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("message 不能为空")
        return value


class SourceCitation(BaseModel):
    """A source chunk used by the knowledge-base tool."""

    source: str
    source_path: str
    chunk_index: int


class ChatResponse(BaseModel):
    thread_id: str
    answer: str
    sources: list[SourceCitation]


class ErrorResponse(BaseModel):
    detail: str


class ConsentRequest(BaseModel):
    granted: bool


class ConsentResponse(BaseModel):
    scope: str
    granted: bool


class EvidenceDraftRequest(BaseModel):
    title: str = Field(min_length=1, max_length=300)
    evidence_type: EvidenceType
    details: dict[str, str]


class EvidenceResponse(BaseModel):
    id: str
    profile_id: str
    evidence_type: EvidenceType
    title: str
    details: dict[str, str]
    confirmation_status: ConfirmationStatus
    source_document_id: str | None


class ProfileInitializationResponse(BaseModel):
    profile_id: str
    drafts: list[EvidenceResponse]


class JDRequest(BaseModel):
    jd_text: str = Field(min_length=1, max_length=50_000)


class JobTitleRequest(BaseModel):
    title: str = Field(min_length=1, max_length=300)


class JobOpportunityResponse(BaseModel):
    id: str
    title: str
    source_jd_text: str
    confirmation_status: JobConfirmationStatus


class JobRequirementRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2_000)
    category: RequirementCategory
    priority: RequirementPriority
    depth: RequirementDepth
    origin: RequirementOrigin
    evidence_assessment: EvidenceAssessment
    supporting_evidence_ids: list[str] = Field(default_factory=list)


class JobRequirementResponse(BaseModel):
    id: str
    job_opportunity_id: str
    text: str
    category: RequirementCategory
    priority: RequirementPriority
    depth: RequirementDepth
    origin: RequirementOrigin
    evidence_assessment: EvidenceAssessment
    supporting_evidence_ids: list[str]


class ActiveOpportunitiesRequest(BaseModel):
    opportunity_ids: list[str] = Field(max_length=3)


class ActiveOpportunitiesResponse(BaseModel):
    opportunity_ids: list[str]
    primary_opportunity_id: str | None


class PrimaryOpportunityRequest(BaseModel):
    opportunity_id: str = Field(min_length=1, max_length=128)


class PreparationTaskRequest(BaseModel):
    title: str = Field(min_length=1, max_length=300)
    description: str = Field(min_length=1, max_length=3_000)
    opportunity_ids: list[str] = Field(min_length=1, max_length=3)
    requirement_ids: list[str] = Field(min_length=1)
    estimated_hours: int = Field(gt=0, le=200)
    deadline: date | None = None


class PreparationTaskResponse(BaseModel):
    id: str
    title: str
    description: str
    opportunity_ids: list[str]
    requirement_ids: list[str]
    estimated_hours: int
    deadline: date | None
    status: TaskStatus


class TaskRecommendationResponse(BaseModel):
    task: PreparationTaskResponse
    gap_count: int
    shared_opportunity_count: int
    primary_opportunity_bonus: bool
    feasible_in_weekly_time: bool
    deadline_urgency: str
    rationale: list[str]


class CareerReportJobResponse(BaseModel):
    """A currently active role and its evidence-aware requirement matrix."""

    id: str
    title: str
    requirements: list[JobRequirementResponse]


class CareerReportNextTaskResponse(PreparationTaskResponse):
    """The actionable recommendation in a report, with display fields at top level."""

    gap_count: int
    shared_opportunity_count: int
    primary_opportunity_bonus: bool
    feasible_in_weekly_time: bool
    deadline_urgency: str
    rationale: list[str]


class CareerReportResponse(BaseModel):
    """A transparent snapshot of the user's current job-search preparation."""

    profile_id: str
    confirmed_evidence_count: int
    active_jobs: list[CareerReportJobResponse]
    next_task: CareerReportNextTaskResponse | None


class ResumeSuggestionResponse(BaseModel):
    section: str
    evidence_id: str
    evidence_title: str
    suggested_bullet: str
    supporting_requirement_ids: list[str]


class ResumeOptimizationResponse(BaseModel):
    profile_id: str
    opportunity_id: str
    suggestions: list[ResumeSuggestionResponse]


class TaskCompletionRequest(BaseModel):
    project_evidence: dict[str, str]
    work_link: str | None = Field(default=None, max_length=2_000)


class PromoteEvidenceRequest(BaseModel):
    profile_id: str = Field(min_length=1, max_length=128)


class ManagedDocumentResponse(BaseModel):
    id: str
    filename: str
    managed_path: str
    index_status: IndexStatus
    last_error: str | None


class AnalyzeJobRequest(BaseModel):
    profile_id: str = Field(min_length=1, max_length=128)


class JobAnalysisResponse(BaseModel):
    responsibilities: list[str]
    requirements: list[JobRequirementResponse]


class ResearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2_000)
    first_party_domains: list[str] = Field(default_factory=list)
    first_party_url_prefixes: list[str] = Field(default_factory=list)
    job_opportunity_id: str | None = Field(default=None, max_length=128)


class ResearchSourceResponse(BaseModel):
    title: str
    url: str
    summary: str
    tier: str


def extract_sources(messages: list[Any]) -> list[SourceCitation]:
    """Extract de-duplicated source metadata from search tool responses."""
    sources: list[SourceCitation] = []
    seen: set[tuple[str, str, int]] = set()

    for message in messages:
        if getattr(message, "name", None) != "search_knowledge_base":
            continue

        for source, source_path, chunk_index in _SOURCE_PATTERN.findall(
            str(message.content)
        ):
            key = (source, source_path, int(chunk_index))
            if key not in seen:
                seen.add(key)
                sources.append(
                    SourceCitation(
                        source=source,
                        source_path=source_path,
                        chunk_index=chunk_index,
                    )
                )

    return sources


def create_app(
    root: Path | None = None,
    agent_factory: AgentFactory = create_career_agent,
    consent_store: ConsentStore | None = None,
    job_analyzer: JobAnalyzer | None = None,
    web_search: Callable[[str], list[dict[str, str]]] | None = None,
    resume_extractor: ResumeExtractor | None = None,
) -> FastAPI:
    """Create the HTTP API without initializing the model eagerly."""
    project_root = root or Path(__file__).resolve().parents[2]
    consents = consent_store
    agent: Any | None = None
    document_manager: DocumentManager | None = None
    profile_store: CapabilityProfileStore | None = None
    job_store: JobOpportunityStore | None = None
    planning_store: PreparationTaskStore | None = None
    analyzer = job_analyzer
    extractor = resume_extractor
    search_provider = web_search or public_web_search

    def get_consent_store() -> ConsentStore:
        nonlocal consents
        if consents is None:
            consents = ConsentStore(project_root / "data" / "consents.sqlite3")
        return consents

    def get_agent() -> Any:
        nonlocal agent
        if agent is None:
            agent = agent_factory(
                project_root / "data" / "chroma",
                project_root / "data" / "conversations.sqlite3",
            )
        return agent

    def get_document_manager() -> DocumentManager:
        nonlocal document_manager
        if document_manager is None:
            document_manager = DocumentManager(
                project_root / "data" / "documents.sqlite3",
                project_root / "data" / "knowledge_documents",
                project_root / "data" / "chroma",
            )
        return document_manager

    def get_profile_store() -> CapabilityProfileStore:
        nonlocal profile_store
        if profile_store is None:
            profile_store = create_profile_store_with_auto_index(
                project_root / "data" / "capability_profiles.sqlite3",
                get_document_manager(),
            )
        return profile_store

    def get_job_store() -> JobOpportunityStore:
        nonlocal job_store
        if job_store is None:
            job_store = JobOpportunityStore(project_root / "data" / "job_opportunities.sqlite3")
        return job_store

    def get_planning_store() -> PreparationTaskStore:
        nonlocal planning_store
        if planning_store is None:
            planning_store = PreparationTaskStore(project_root / "data" / "planning.sqlite3")
        return planning_store

    def get_job_analyzer() -> JobAnalyzer:
        nonlocal analyzer
        if analyzer is None:
            analyzer = create_model_job_analyzer()
        return analyzer

    def get_resume_extractor() -> ResumeExtractor:
        nonlocal extractor
        if extractor is None:
            extractor = create_model_resume_extractor()
        return extractor

    app = FastAPI(title="Career Knowledge Agent API", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/consents")
    def get_consents() -> dict[str, bool]:
        return {scope.value: granted for scope, granted in get_consent_store().all().items()}

    @app.put("/consents/{scope}")
    def set_consent(scope: ConsentScope, request: ConsentRequest) -> ConsentResponse:
        get_consent_store().set(scope, granted=request.granted)
        return ConsentResponse(scope=scope.value, granted=request.granted)

    @app.post("/research", response_model=list[ResearchSourceResponse])
    def research(request: ResearchRequest) -> list[ResearchSourceResponse]:
        try:
            sources = ControlledResearchService(get_consent_store(), search_provider).research(
                request.query,
                first_party_domains=request.first_party_domains,
                first_party_url_prefixes=request.first_party_url_prefixes,
            )
        except PermissionError as error:
            raise HTTPException(status_code=403, detail="请先同意联网研究，才能执行公开搜索。") from error
        except RuntimeError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        if request.job_opportunity_id:
            try:
                get_job_store().save_research_sources(
                    request.job_opportunity_id,
                    sources=[(item.title, item.url, item.summary, item.tier.value) for item in sources],
                )
            except ValueError as error:
                raise HTTPException(status_code=404, detail=str(error)) from error
        return [_research_source_response(source) for source in sources]

    @app.post("/documents", response_model=ManagedDocumentResponse)
    def upload_document(file: UploadFile = File(...)) -> ManagedDocumentResponse:
        if not file.filename:
            raise HTTPException(status_code=400, detail="请选择资料文件。")
        staging_dir = project_root / "data" / "upload_staging"
        staging_dir.mkdir(parents=True, exist_ok=True)
        staging_path = staging_dir / str(uuid4()) / Path(file.filename).name
        staging_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            staging_path.write_bytes(file.file.read())
            document = get_document_manager().upload(staging_path)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        finally:
            staging_path.unlink(missing_ok=True)
            staging_path.parent.rmdir()
        return _managed_document_response(document)

    @app.get("/documents", response_model=list[ManagedDocumentResponse])
    def list_documents() -> list[ManagedDocumentResponse]:
        return [_managed_document_response(item) for item in get_document_manager().list_documents()]

    @app.put("/documents/{document_id}", response_model=ManagedDocumentResponse)
    def replace_document(
        document_id: str, file: UploadFile = File(...)
    ) -> ManagedDocumentResponse:
        if not file.filename:
            raise HTTPException(status_code=400, detail="请选择替换资料文件。")
        staging_dir = project_root / "data" / "upload_staging" / str(uuid4())
        staging_dir.mkdir(parents=True, exist_ok=True)
        staging_path = staging_dir / Path(file.filename).name
        try:
            staging_path.write_bytes(file.file.read())
            document = get_document_manager().replace(document_id, staging_path)
        except ValueError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        finally:
            staging_path.unlink(missing_ok=True)
            staging_dir.rmdir()
        return _managed_document_response(document)

    @app.delete("/documents/{document_id}", status_code=204)
    def delete_document(document_id: str) -> None:
        try:
            get_document_manager().delete(document_id)
        except ValueError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except RuntimeError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @app.post("/profiles/empty", response_model=ProfileInitializationResponse)
    def create_empty_profile() -> ProfileInitializationResponse:
        result = initialize_empty_profile(get_profile_store())
        return _profile_initialization_response(result.profile.id, result.drafts)

    @app.post("/profiles/resume", response_model=ProfileInitializationResponse)
    def create_profile_from_resume(file: UploadFile = File(...)) -> ProfileInitializationResponse:
        if not file.filename:
            raise HTTPException(status_code=400, detail="请选择一份简历文件。")
        staging_dir = project_root / "data" / "upload_staging"
        staging_dir.mkdir(parents=True, exist_ok=True)
        staging_path = staging_dir / f"{uuid4()}-{Path(file.filename).name}"
        try:
            staging_path.write_bytes(file.file.read())
            model_extractor = None
            if get_consent_store().granted(ConsentScope.MODEL_ANALYSIS):
                try:
                    model_extractor = get_resume_extractor()
                except RuntimeError:
                    model_extractor = None
            try:
                result = initialize_profile_from_resume(
                get_profile_store(),
                resume_path=staging_path,
                source_documents_dir=project_root / "data" / "source_documents",
                resume_extractor=model_extractor,
                )
            except Exception:
                # A malformed or unavailable model must not block a local-only upload.
                result = initialize_profile_from_resume(
                    get_profile_store(),
                    resume_path=staging_path,
                    source_documents_dir=project_root / "data" / "source_documents",
                )
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        finally:
            staging_path.unlink(missing_ok=True)
        return _profile_initialization_response(result.profile.id, result.drafts)

    @app.get("/profiles/{profile_id}/evidence/drafts", response_model=list[EvidenceResponse])
    def list_evidence_drafts(profile_id: str) -> list[EvidenceResponse]:
        try:
            return [_evidence_response(item) for item in get_profile_store().list_draft_evidence(profile_id)]
        except ValueError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @app.get("/profiles/{profile_id}/evidence", response_model=list[EvidenceResponse])
    def list_confirmed_evidence(profile_id: str) -> list[EvidenceResponse]:
        try:
            return [
                _evidence_response(item)
                for item in get_profile_store().list_confirmed_evidence(profile_id)
            ]
        except ValueError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @app.post("/profiles/{profile_id}/evidence", response_model=EvidenceResponse)
    def add_evidence_draft(profile_id: str, request: EvidenceDraftRequest) -> EvidenceResponse:
        try:
            record = get_profile_store().add_evidence_record(
                profile_id,
                evidence_type=request.evidence_type,
                title=request.title,
                details=request.details,
            )
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return _evidence_response(record)

    @app.put("/evidence/{evidence_id}", response_model=EvidenceResponse)
    def update_evidence_draft(evidence_id: str, request: EvidenceDraftRequest) -> EvidenceResponse:
        try:
            record = get_profile_store().update_draft_evidence_record(
                evidence_id,
                title=request.title,
                details=request.details,
                evidence_type=request.evidence_type,
            )
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return _evidence_response(record)

    @app.post("/evidence/{evidence_id}/confirm", response_model=EvidenceResponse)
    def confirm_evidence_draft(evidence_id: str) -> EvidenceResponse:
        try:
            return _evidence_response(get_profile_store().confirm_evidence_record(evidence_id))
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.delete("/evidence/{evidence_id}", status_code=204)
    def discard_evidence_draft(evidence_id: str) -> None:
        try:
            get_profile_store().delete_evidence_record(evidence_id)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.post("/jobs/jd", response_model=list[JobOpportunityResponse])
    def import_job_description(request: JDRequest) -> list[JobOpportunityResponse]:
        try:
            return [_job_response(item) for item in get_job_store().import_jd_document(request.jd_text)]
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.get("/jobs", response_model=list[JobOpportunityResponse])
    def list_job_opportunities() -> list[JobOpportunityResponse]:
        return [_job_response(item) for item in get_job_store().list_opportunities()]

    @app.put("/jobs/{opportunity_id}", response_model=JobOpportunityResponse)
    def update_job_opportunity(opportunity_id: str, request: JobTitleRequest) -> JobOpportunityResponse:
        try:
            return _job_response(get_job_store().update_draft_job_opportunity(opportunity_id, title=request.title))
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.post("/jobs/{opportunity_id}/confirm", response_model=JobOpportunityResponse)
    def confirm_job_opportunity(opportunity_id: str) -> JobOpportunityResponse:
        try:
            return _job_response(get_job_store().confirm_job_opportunity(opportunity_id))
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.post("/jobs/{opportunity_id}/requirements", response_model=JobRequirementResponse)
    def add_job_requirement(
        opportunity_id: str, request: JobRequirementRequest
    ) -> JobRequirementResponse:
        try:
            requirement = get_job_store().add_requirement(
                opportunity_id,
                text=request.text,
                category=request.category,
                priority=request.priority,
                depth=request.depth,
                origin=request.origin,
                evidence_assessment=request.evidence_assessment,
                supporting_evidence_ids=request.supporting_evidence_ids,
            )
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return _job_requirement_response(requirement)

    @app.get("/jobs/{opportunity_id}/requirements", response_model=list[JobRequirementResponse])
    def get_job_requirement_matrix(opportunity_id: str) -> list[JobRequirementResponse]:
        try:
            return [_job_requirement_response(item) for item in get_job_store().requirement_matrix(opportunity_id)]
        except ValueError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @app.get(
        "/jobs/{opportunity_id}/research-sources",
        response_model=list[ResearchSourceResponse],
    )
    def get_job_research_sources(opportunity_id: str) -> list[ResearchSourceResponse]:
        try:
            return [
                _job_research_source_response(item)
                for item in get_job_store().list_research_sources(opportunity_id)
            ]
        except ValueError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @app.post("/jobs/{opportunity_id}/analyze", response_model=JobAnalysisResponse)
    def analyze_job_opportunity(opportunity_id: str, request: AnalyzeJobRequest) -> JobAnalysisResponse:
        if not get_consent_store().granted(ConsentScope.MODEL_ANALYSIS):
            raise HTTPException(status_code=403, detail="请先同意模型分析，才能分析岗位 JD。")
        try:
            opportunity = get_job_store().get_opportunity(opportunity_id)
            evidence = get_profile_store().list_confirmed_evidence(request.profile_id)
            analysis = JobAnalysis.model_validate(get_job_analyzer()(opportunity.source_jd_text, evidence))
            for responsibility in analysis.responsibilities:
                get_job_store().add_responsibility(opportunity_id, text=responsibility)
            requirements = [
                get_job_store().add_requirement(
                    opportunity_id, text=item.text, category=item.category,
                    priority=item.priority, depth=item.depth, origin=item.origin,
                    evidence_assessment=item.evidence_assessment,
                    supporting_evidence_ids=item.supporting_evidence_ids,
                )
                for item in analysis.requirements
            ]
        except (RuntimeError, ValueError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return JobAnalysisResponse(
            responsibilities=analysis.responsibilities,
            requirements=[_job_requirement_response(item) for item in requirements],
        )

    @app.put("/planning/active-opportunities", response_model=ActiveOpportunitiesResponse)
    def set_active_opportunities(request: ActiveOpportunitiesRequest) -> ActiveOpportunitiesResponse:
        try:
            opportunity_ids = get_planning_store().set_active_opportunities(request.opportunity_ids)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return ActiveOpportunitiesResponse(
            opportunity_ids=list(opportunity_ids), primary_opportunity_id=None
        )

    @app.get("/planning/active-opportunities", response_model=ActiveOpportunitiesResponse)
    def get_active_opportunities() -> ActiveOpportunitiesResponse:
        store = get_planning_store()
        return ActiveOpportunitiesResponse(
            opportunity_ids=list(store.active_opportunity_ids()),
            primary_opportunity_id=store.primary_opportunity_id(),
        )

    @app.put("/planning/primary-opportunity", response_model=ActiveOpportunitiesResponse)
    def select_primary_opportunity(
        request: PrimaryOpportunityRequest,
    ) -> ActiveOpportunitiesResponse:
        store = get_planning_store()
        try:
            store.select_primary_opportunity(request.opportunity_id)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return ActiveOpportunitiesResponse(
            opportunity_ids=list(store.active_opportunity_ids()),
            primary_opportunity_id=store.primary_opportunity_id(),
        )

    @app.post("/planning/tasks", response_model=PreparationTaskResponse)
    def create_preparation_task(request: PreparationTaskRequest) -> PreparationTaskResponse:
        try:
            task = get_planning_store().add_task(
                title=request.title,
                description=request.description,
                opportunity_ids=request.opportunity_ids,
                requirement_ids=request.requirement_ids,
                estimated_hours=request.estimated_hours,
                deadline=request.deadline,
            )
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return _task_response(task)

    @app.get("/planning/recommendation", response_model=TaskRecommendationResponse | None)
    def get_next_task_recommendation(weekly_hours: int = 8) -> TaskRecommendationResponse | None:
        store = get_planning_store()
        recommendation = store.recommend_next_task(
            get_job_store().requirements_for_opportunities(store.active_opportunity_ids()),
            weekly_hours=weekly_hours,
        )
        return _recommendation_response(recommendation) if recommendation else None

    @app.get("/reports/career", response_model=CareerReportResponse)
    def get_career_report(
        profile_id: str, weekly_hours: int = 8
    ) -> CareerReportResponse:
        """Return facts already confirmed by the user; no opaque fit score is inferred."""
        try:
            confirmed_evidence = get_profile_store().list_confirmed_evidence(profile_id)
        except ValueError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

        planning = get_planning_store()
        active_ids = planning.active_opportunity_ids()
        active_jobs: list[CareerReportJobResponse] = []
        for opportunity_id in active_ids:
            try:
                opportunity = get_job_store().get_opportunity(opportunity_id)
            except ValueError:
                # An opportunity may have been removed after it was activated.
                # Keeping the report available is more useful than failing all output.
                continue
            requirements = get_job_store().requirement_matrix(opportunity_id)
            active_jobs.append(
                CareerReportJobResponse(
                    id=opportunity.id,
                    title=opportunity.title,
                    requirements=[_job_requirement_response(item) for item in requirements],
                )
            )

        recommendation = planning.recommend_next_task(
            get_job_store().requirements_for_opportunities(active_ids),
            weekly_hours=weekly_hours,
        )
        return CareerReportResponse(
            profile_id=profile_id,
            confirmed_evidence_count=len(confirmed_evidence),
            active_jobs=active_jobs,
            next_task=_career_report_next_task_response(recommendation) if recommendation else None,
        )

    @app.get("/resume/optimization", response_model=ResumeOptimizationResponse)
    def get_resume_optimization(
        profile_id: str, opportunity_id: str
    ) -> ResumeOptimizationResponse:
        try:
            evidence = get_profile_store().list_confirmed_evidence(profile_id)
            requirements = get_job_store().requirement_matrix(opportunity_id)
        except ValueError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        suggestions = build_resume_suggestions(evidence, requirements)
        return ResumeOptimizationResponse(
            profile_id=profile_id,
            opportunity_id=opportunity_id,
            suggestions=[
                ResumeSuggestionResponse(
                    section=item.section,
                    evidence_id=item.evidence_id,
                    evidence_title=item.evidence_title,
                    suggested_bullet=item.suggested_bullet,
                    supporting_requirement_ids=list(item.supporting_requirement_ids),
                )
                for item in suggestions
            ],
        )

    @app.post("/planning/tasks/{task_id}/completion", response_model=PreparationTaskResponse)
    def submit_task_completion(task_id: str, request: TaskCompletionRequest) -> PreparationTaskResponse:
        try:
            task = get_planning_store().submit_completion_evidence(
                task_id,
                project_evidence=request.project_evidence,
                work_link=request.work_link,
            )
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return _task_response(task)

    @app.post("/planning/tasks/{task_id}/completion/confirm", response_model=PreparationTaskResponse)
    def confirm_task_completion(task_id: str) -> PreparationTaskResponse:
        try:
            return _task_response(get_planning_store().confirm_completion(task_id))
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.post("/planning/tasks/{task_id}/promote-evidence", response_model=EvidenceResponse)
    def promote_task_evidence(task_id: str, request: PromoteEvidenceRequest) -> EvidenceResponse:
        planning = get_planning_store()
        if planning.promoted_evidence_id(task_id, request.profile_id):
            raise HTTPException(
                status_code=409,
                detail="该任务已经写入此能力档案，无需重复提交。",
            )
        try:
            details, work_link = planning.confirmed_completion_evidence(task_id)
            if work_link:
                details = {**details, "work_link": work_link}
            record = get_profile_store().add_evidence_record(
                request.profile_id,
                evidence_type=EvidenceType.PROJECT,
                title=details["title"],
                details={key: value for key, value in details.items() if key != "title"},
                confirmation_status=ConfirmationStatus.CONFIRMED,
            )
            planning.record_evidence_promotion(task_id, request.profile_id, record.id)
        except (KeyError, ValueError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return _evidence_response(record)

    @app.post(
        "/chat",
        response_model=ChatResponse,
        responses={503: {"model": ErrorResponse}},
    )
    def chat(request: ChatRequest) -> ChatResponse:
        thread_id = request.thread_id or str(uuid4())
        if not get_consent_store().granted(ConsentScope.MODEL_ANALYSIS):
            raise HTTPException(
                status_code=403,
                detail="请先明确同意“模型分析”，系统才会向模型服务发送聊天内容。",
            )
        try:
            message = request.message
            if request.profile_id:
                message = f"{request.message}\n\n{_chat_career_context(request.profile_id, get_profile_store(), get_job_store(), get_planning_store())}"
            result = get_agent().invoke(
                {"messages": [{"role": "user", "content": message}]},
                config={"configurable": {"thread_id": thread_id}},
            )
        except Exception as error:
            raise HTTPException(
                status_code=503,
                detail=friendly_error_message(error),
            ) from error

        messages = result["messages"]
        return ChatResponse(
            thread_id=thread_id,
            answer=str(messages[-1].content),
            sources=extract_sources(messages),
        )

    return app


app = create_app()


def _evidence_response(record: EvidenceRecord) -> EvidenceResponse:
    return EvidenceResponse(
        id=record.id,
        profile_id=record.profile_id,
        evidence_type=record.evidence_type,
        title=record.title,
        details=record.details,
        confirmation_status=record.confirmation_status,
        source_document_id=record.source_document_id,
    )


def _profile_initialization_response(
    profile_id: str, drafts: tuple[EvidenceRecord, ...]
) -> ProfileInitializationResponse:
    return ProfileInitializationResponse(
        profile_id=profile_id,
        drafts=[_evidence_response(draft) for draft in drafts],
    )


def _job_response(opportunity: JobOpportunity) -> JobOpportunityResponse:
    return JobOpportunityResponse(
        id=opportunity.id,
        title=opportunity.title,
        source_jd_text=opportunity.source_jd_text,
        confirmation_status=opportunity.confirmation_status,
    )


def _job_requirement_response(requirement: JobRequirement) -> JobRequirementResponse:
    return JobRequirementResponse(
        id=requirement.id,
        job_opportunity_id=requirement.job_opportunity_id,
        text=requirement.text,
        category=requirement.category,
        priority=requirement.priority,
        depth=requirement.depth,
        origin=requirement.origin,
        evidence_assessment=requirement.evidence_assessment,
        supporting_evidence_ids=list(requirement.supporting_evidence_ids),
    )


def _task_response(task: PreparationTask) -> PreparationTaskResponse:
    return PreparationTaskResponse(
        id=task.id, title=task.title, description=task.description,
        opportunity_ids=list(task.opportunity_ids), requirement_ids=list(task.requirement_ids),
        estimated_hours=task.estimated_hours, deadline=task.deadline, status=task.status,
    )


def _recommendation_response(recommendation: TaskRecommendation) -> TaskRecommendationResponse:
    return TaskRecommendationResponse(
        task=_task_response(recommendation.task), gap_count=recommendation.gap_count,
        shared_opportunity_count=recommendation.shared_opportunity_count,
        primary_opportunity_bonus=recommendation.primary_opportunity_bonus,
        feasible_in_weekly_time=recommendation.feasible_in_weekly_time,
        deadline_urgency=recommendation.deadline_urgency, rationale=list(recommendation.rationale),
    )


def _career_report_next_task_response(
    recommendation: TaskRecommendation,
) -> CareerReportNextTaskResponse:
    task = _task_response(recommendation.task)
    return CareerReportNextTaskResponse(
        **task.model_dump(),
        gap_count=recommendation.gap_count,
        shared_opportunity_count=recommendation.shared_opportunity_count,
        primary_opportunity_bonus=recommendation.primary_opportunity_bonus,
        feasible_in_weekly_time=recommendation.feasible_in_weekly_time,
        deadline_urgency=recommendation.deadline_urgency,
        rationale=list(recommendation.rationale),
    )


def _job_research_source_response(source: JobResearchSource) -> ResearchSourceResponse:
    return ResearchSourceResponse(
        title=source.title, url=source.url, summary=source.summary, tier=source.tier,
    )


def _chat_career_context(
    profile_id: str,
    profile_store: CapabilityProfileStore,
    job_store: JobOpportunityStore,
    planning_store: PreparationTaskStore,
) -> str:
    """Append concise, local-only facts so the model can ground its chat response."""
    evidence = profile_store.list_confirmed_evidence(profile_id)
    jobs = []
    for opportunity_id in planning_store.active_opportunity_ids():
        try:
            jobs.append(job_store.get_opportunity(opportunity_id).title)
        except ValueError:
            continue
    evidence_titles = "、".join(item.title for item in evidence) or "无"
    active_jobs = "、".join(jobs) or "无"
    return (
        "[职业工作台上下文]\n"
        f"已确认能力证据（{len(evidence)}）：{evidence_titles}\n"
        f"当前活跃岗位：{active_jobs}\n"
        "请仅将这些内容视作本地已确认事实；证据不足时明确说明，不要补造经历。"
    )


def _managed_document_response(document: ManagedDocument) -> ManagedDocumentResponse:
    return ManagedDocumentResponse(
        id=document.id,
        filename=document.filename,
        managed_path=document.managed_path,
        index_status=document.index_status,
        last_error=document.last_error,
    )


def _research_source_response(source: ResearchSource) -> ResearchSourceResponse:
    return ResearchSourceResponse(title=source.title, url=source.url, summary=source.summary, tier=source.tier.value)

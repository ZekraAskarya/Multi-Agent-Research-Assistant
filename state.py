from typing import TypedDict, List, Optional, Annotated
import operator
from pydantic import BaseModel, Field


# ---------- Structured outputs ----------
class SubQuestions(BaseModel):
    questions: List[str] = Field(
        description="2-4 focused, non-overlapping sub-questions that together answer the main query."
    )


class Fact(BaseModel):
    claim: str = Field(description="A concise, verifiable factual claim.")
    evidence: str = Field(
        description="Short supporting excerpt from the source.")
    source_url: str = Field(description="URL where the evidence was found.")
    relevance: float = Field(
        ge=0.0, le=1.0, description="Relevance to the query (0-1).")


class AnalyzedFacts(BaseModel):
    facts: List[Fact] = Field(default_factory=list)
    is_sufficient: bool = Field(
        description="True if the gathered facts are enough to write a high-quality report."
    )
    missing_aspects: List[str] = Field(
        default_factory=list,
        description="If insufficient, up to 3 NEW search queries that would fill the gaps.",
    )
    summary: str = Field(description="Brief synthesis of the key findings.")


class Critique(BaseModel):
    needs_revision: bool
    feedback: str = Field(description="Specific, actionable feedback.")
    quality_score: float = Field(ge=0.0, le=10.0)


# ---------- Graph state ----------
class ResearchState(TypedDict):
    query: str
    sub_questions: List[str]
    next_queries: List[str]
    # accumulates across rounds
    raw_findings: Annotated[List[dict], operator.add]
    analyzed: Optional[dict]
    report: str
    critique: Optional[dict]
    research_iteration: int
    max_research_iterations: int
    revision_count: int
    max_revisions: int

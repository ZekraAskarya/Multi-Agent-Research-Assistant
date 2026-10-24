from langgraph.graph import StateGraph, START, END
from langgraph.types import RetryPolicy

from state import ResearchState
from agents import (
    planner_node, researcher_node, analyzer_node, writer_node, reviewer_node,
)


# ---------- Retry policy for transient 503s / 5xx ----------
LLM_RETRY = RetryPolicy(
    max_attempts=5,
    initial_interval=2.0,
    backoff_factor=2.0,
    max_interval=30.0,
    jitter=True,
)


# ---------- Routing functions (conditional edges) ----------
def route_after_analysis(state: ResearchState) -> str:
    """Loop back to Researcher if the Analyzer says we're not done yet."""
    a = state["analyzed"]
    if (
        not a["is_sufficient"]
        and state["research_iteration"] < state["max_research_iterations"]
        and state["next_queries"]
    ):
        return "researcher"
    return "writer"


def route_after_review(state: ResearchState) -> str:
    """Loop back to Writer if the Reviewer demands a revision."""
    c = state["critique"]
    if c["needs_revision"] and state["revision_count"] < state["max_revisions"]:
        return "writer"
    return END


# ---------- Graph construction ----------
def build_graph():
    g = StateGraph(ResearchState)

    g.add_node("planner", planner_node, retry_policy=LLM_RETRY)
    g.add_node("researcher", researcher_node, retry_policy=LLM_RETRY)
    g.add_node("analyzer", analyzer_node, retry_policy=LLM_RETRY)
    g.add_node("writer", writer_node, retry_policy=LLM_RETRY)
    g.add_node("reviewer", reviewer_node, retry_policy=LLM_RETRY)

    g.add_edge(START, "planner")
    g.add_edge("planner", "researcher")
    g.add_edge("researcher", "analyzer")

    g.add_conditional_edges(
        "analyzer",
        route_after_analysis,
        {"researcher": "researcher", "writer": "writer"},
    )

    g.add_edge("writer", "reviewer")
    g.add_conditional_edges(
        "reviewer",
        route_after_review,
        {"writer": "writer", END: END},
    )

    return g.compile()

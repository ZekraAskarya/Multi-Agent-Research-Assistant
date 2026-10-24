from utils import extract_text
from state import ResearchState, SubQuestions, AnalyzedFacts, Critique
from tavily import AsyncTavilyClient
from langchain_google_genai import ChatGoogleGenerativeAI
import os
import asyncio
import logging
from dotenv import load_dotenv

load_dotenv()


logger = logging.getLogger("agents")

# ---------- Model definitions (primary + fallbacks) ----------
_primary = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    max_retries=3,
    api_key=os.getenv("GOOGLE_API_KEY"),
)
_fallback_1 = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    max_retries=2,
    api_key=os.getenv("GOOGLE_API_KEY"),
)
_fallback_2 = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    max_retries=2,
    api_key=os.getenv("GOOGLE_API_KEY"),
)

# Plain-text LLM with fallback — used by writer_node
llm = _primary.with_fallbacks([_fallback_1, _fallback_2])

# Structured-output LLMs with fallback — used by planner/analyzer/reviewer


def _structured_with_fallbacks(schema):
    return _primary.with_structured_output(schema).with_fallbacks([
        _fallback_1.with_structured_output(schema),
        _fallback_2.with_structured_output(schema),
    ])


planner_llm = _structured_with_fallbacks(SubQuestions)
analyzer_llm = _structured_with_fallbacks(AnalyzedFacts)
reviewer_llm = _structured_with_fallbacks(Critique)

# Web search
tavily = AsyncTavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

# ---------- 1. PLANNER ----------


async def planner_node(state: ResearchState) -> dict:
    logger.info("🧭 [PLANNER] Decomposing query")
    prompt = f"""You are a research planner. Decompose this query into 2-4 focused,
non-overlapping sub-questions that together fully answer it.

Query: {state['query']}
"""
    result: SubQuestions = await planner_llm.ainvoke(prompt)
    logger.info(f"   → {len(result.questions)} sub-questions")
    return {
        "sub_questions": result.questions,
        "next_queries": result.questions,
        "research_iteration": 0,
        "revision_count": 0,
    }


# ---------- 2. RESEARCHER ----------
async def researcher_node(state: ResearchState) -> dict:
    queries = state.get("next_queries") or state["sub_questions"]
    logger.info(f"🔍 [RESEARCHER] {len(queries)} parallel searches")

    async def safe_search(q: str) -> list[dict]:
        try:
            resp = await tavily.search(query=q, max_results=3, search_depth="advanced")
            return [
                {
                    "query": q,
                    "title": r.get("title", ""),
                    "content": r.get("content", ""),
                    "url": r.get("url", ""),
                    "score": r.get("score", 0.0),
                }
                for r in resp.get("results", [])
            ]
        except Exception as e:
            logger.warning(f"   ! Search failed for '{q}': {e}")
            return []

    batches = await asyncio.gather(*(safe_search(q) for q in queries))
    findings = [item for batch in batches for item in batch]
    logger.info(f"   → Collected {len(findings)} raw results")
    return {"raw_findings": findings, "next_queries": []}
# ---------- 3. ANALYZER ----------


async def analyzer_node(state: ResearchState) -> dict:
    logger.info(f"🧪 [ANALYZER] Filtering {len(state['raw_findings'])} results")

    corpus = "\n\n".join(
        f"[{i}] ({r['url']})\n{r['title']}\n{r['content'][:800]}"
        for i, r in enumerate(state["raw_findings"])
    )

    prompt = f"""You are a rigorous research analyst.

User query: {state['query']}
Research round: {state['research_iteration'] + 1} of {state['max_research_iterations']}

From the raw search results below, extract the most relevant factual claims.
Rules:
- Only include claims directly supported by the provided text.
- Deduplicate: if multiple sources say the same thing, keep the best-evidenced one.
- Score relevance to the original query (0-1) and drop anything below 0.4.
- Decide if these facts are sufficient for a high-quality report.
- If NOT sufficient, propose up to 3 specific NEW search queries to fill the gaps.

Raw results:
{corpus}
"""
    analysis: AnalyzedFacts = await analyzer_llm.ainvoke(prompt)
    logger.info(
        f"   → {len(analysis.facts)} facts | sufficient={analysis.is_sufficient}")

    return {
        "analyzed": analysis.model_dump(),
        "next_queries": analysis.missing_aspects,
        "research_iteration": state["research_iteration"] + 1,
    }


# ---------- 4. WRITER ----------
async def writer_node(state: ResearchState) -> dict:
    logger.info(f"✍️  [WRITER] Drafting (revision #{state['revision_count']})")
    analyzed = state["analyzed"]
    facts_block = "\n".join(
        f"- {f['claim']}\n  Evidence: {f['evidence'][:300]}\n  Source: {f['source_url']}"
        for f in analyzed["facts"]
    )
    feedback = ""
    if state.get("critique") and state["critique"].get("needs_revision"):
        feedback = f"\n\nReviewer feedback to address:\n{state['critique']['feedback']}"

    prompt = f"""You are a professional research writer. Produce a Markdown report.

Original query: {state['query']}

Key synthesis:
{analyzed.get('summary', '')}

Verified facts:
{facts_block}
{feedback}

Requirements:
- H1 title + Executive summary.
- Clear H2 sections.
- Inline citations [1], [2]... matching the fact order, plus a final "## Sources" section listing URLs.
- Never invent facts beyond those provided.
"""
    response = await llm.ainvoke(prompt)
    return {"report": extract_text(response.content)}


# ---------- 5. REVIEWER ----------
async def reviewer_node(state: ResearchState) -> dict:
    logger.info("🧐 [REVIEWER] Critiquing")
    prompt = f"""You are a demanding editor. Evaluate this report.

Query: {state['query']}
Report:
{state['report']}

Score coverage, structure, clarity, citation quality, factual grounding (0-10).
Set needs_revision=True ONLY for material problems (missing key aspect, unsupported claim, structural issue). Do not nitpick style.
"""
    critique: Critique = await reviewer_llm.ainvoke(prompt)
    logger.info(
        f"   → score={critique.quality_score}/10 revision={critique.needs_revision}")
    return {
        "critique": critique.model_dump(),
        "revision_count": state["revision_count"] + 1,
    }

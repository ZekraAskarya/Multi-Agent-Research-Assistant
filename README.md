# Multi-Agent Research Assistant

A five-agent LangGraph pipeline that researches a topic and writes a cited Markdown report — with two bounded feedback loops for self-correction.

## Architecture

```
START → planner → researcher → analyzer ─┬→ writer → reviewer ─┬→ END
              ▲                          │                    │
              └──── (insufficient) ──────┘                    │
                                    ▲                          │
                                    └──── (needs_revision) ────┘
```

Both loops are bounded by explicit counters in state (`max_research_iterations`, `max_revisions`) so the graph cannot run forever.

## Agents

| Agent | Role |
|---|---|
| **Planner** | Decomposes the user query into 2–4 focused sub-questions |
| **Researcher** | Runs parallel Tavily searches for each sub-question |
| **Analyzer** | Extracts facts, dedupes, scores relevance, and decides if more research is needed |
| **Writer** | Drafts a Markdown report with inline citations |
| **Reviewer** | Critiques the draft (0–10 score) and can request a revision |

### Feedback loops

- **Research loop** — if the Analyzer finds the facts insufficient, it proposes refined queries and sends the graph back to the Researcher.
- **Revision loop** — if the Reviewer returns `needs_revision=True`, the graph sends the report back to the Writer with specific feedback.

## Sample output

See [`examples/sample-report.md`](examples/sample-report.md) for a real run on *"The impact of AI on renewable energy in 2025"*:

```
📊 6 facts | 2 research rounds | 1 revision rounds | quality=9.2/10
```

## Stack

- **Python 3.11+**
- [LangGraph](https://github.com/langchain-ai/langgraph) — graph orchestration, conditional edges, retry policy
- [Gemini 2.5 Flash Lite](https://ai.google.dev/) — LLM (with fallback chain to `2.5-flash` and `3.6-flash`)
- [Tavily](https://tavily.com) — web search API

## Setup

```bash
git clone https://github.com/YOUR-USERNAME/multi-agent-research-assistant.git
cd multi-agent-research-assistant

python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
```

Create `.env` in the project root:

```env
GOOGLE_API_KEY=your_gemini_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

- Get a free Gemini key at [aistudio.google.com](https://aistudio.google.com)
- Get a free Tavily key at [tavily.com](https://tavily.com) (1,000 searches/month, no credit card)

## Run

```bash
python main.py
```

Enter a topic when prompted. The script streams each node's progress, prints the final report to stdout, and saves a copy to `examples/sample-report.md`.

## Files

| File | Purpose |
|---|---|
| `main.py` | Async entry point — streams node updates, prints the report, saves a sample |
| `graph.py` | StateGraph wiring, conditional edges, retry policy |
| `agents.py` | Five agent nodes, LLM fallback chain, Tavily client |
| `state.py` | `ResearchState` TypedDict + Pydantic schemas for structured outputs |
| `utils.py` | `extract_text()` helper for flattening Gemini content blocks |
| `examples/sample-report.md` | Reference output from a successful run |

## Tuning

Edit the defaults in `main.py`:

```python
asyncio.run(run(topic, max_research_iterations=2, max_revisions=1))
```

- `max_research_iterations` — how many extra research rounds the Analyzer can trigger (default 2)
- `max_revisions` — how many Writer revisions the Reviewer can request (default 1)

## Notes on free-tier rate limits

The free tier of Gemini varies significantly by model:

| Model | RPM | RPD |
|---|---|---|
| `gemini-2.5-flash-lite` | 15 | 1,000 |
| `gemini-2.5-flash` | 10 | 250 |
| `gemini-3.6-flash` | 5 | 20 |

A single run of this pipeline makes 4–6 LLM calls. `gemini-3.6-flash` (5 RPM, 20 RPD) is too restrictive — it fails on most runs. The project defaults to `gemini-2.5-flash-lite` with fallbacks to `2.5-flash` and `3.6-flash`, which comfortably handles the workload on the free tier.

## License

MIT — see [LICENSE](LICENSE).
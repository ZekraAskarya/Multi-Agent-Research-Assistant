from graph import build_graph
import asyncio
import logging
from dotenv import load_dotenv

load_dotenv()


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")


async def run(query: str, max_research_iterations: int = 2, max_revisions: int = 1):
    app = build_graph()

    inputs = {
        "query": query,
        "sub_questions": [],
        "next_queries": [],
        "raw_findings": [],
        "analyzed": None,
        "report": "",
        "critique": None,
        "research_iteration": 0,
        "max_research_iterations": max_research_iterations,
        "revision_count": 0,
        "max_revisions": max_revisions,
    }

    print("\n" + "=" * 60)
    print(f"🚀 RESEARCH TASK: {query}")
    print("=" * 60)

    final_report = ""
    final_state = inputs

    async for chunk in app.astream(
        inputs,
        config={"recursion_limit": 50},
        stream_mode="updates",
    ):
        for node_name, update in chunk.items():
            logger.info(f"✅ node '{node_name}' finished")
            final_state = {**final_state, **(update or {})}
            if node_name == "writer" and update.get("report"):
                final_report = update["report"]

    print("\n" + "=" * 60)
    print("📄 FINAL REPORT")
    print("=" * 60 + "\n")
    print(final_report)

    # Save a copy for the repo / review — only if we actually got a report
    if final_report:
        from pathlib import Path
        Path("examples").mkdir(exist_ok=True)
        Path("examples/sample-report.md").write_text(final_report, encoding="utf-8")
        print("\n💾 Saved to examples/sample-report.md")
    else:
        print("\n⚠️  No report generated — examples/sample-report.md not overwritten.")

    a = final_state.get("analyzed") or {}
    c = final_state.get("critique") or {}
    print("\n" + "=" * 60)
    print(
        f"📊 {len(a.get('facts', []))} facts | "
        f"{final_state['research_iteration']} research rounds | "
        f"{final_state['revision_count']} revision rounds | "
        f"quality={c.get('quality_score', '?')}/10"
    )


if __name__ == "__main__":
    topic = input("Research topic: ").strip()
    asyncio.run(run(topic))

"""
Two-Agent Escalation Triage (LangGraph)
=======================================

How it works:

    You type a question, e.g. "Which technical support requests need escalation?"
          |
          v
    AGENT 1: RETRIEVER  -> picks filters and calls read_records -> Supabase
          |
          |  (no records found? -> stop and say so)
          v
    AGENT 2: TRIAGE ANALYST -> follows skills/escalation_triage.md
          |                    (no database access, only recommends)
          v
    Escalation report is printed

Settings live in .env. The triage agent's skill lives in skills/escalation_triage.md.
"""

import json
import os
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from supabase import create_client


# =====================================================
# STEP 1: Load settings from .env
# =====================================================

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_TABLE = os.getenv("SUPABASE_TABLE", "enterprise_customer_operations")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Please set SUPABASE_URL and SUPABASE_KEY in the .env file")


# =====================================================
# STEP 2: Connect to the LLM (Ollama) and the database (Supabase)
# =====================================================

# The local LLM shared by both agents. temperature=0 = predictable answers.
llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


def table():
    """Return the Supabase table configured in .env."""
    return supabase.table(SUPABASE_TABLE)


# =====================================================
# STEP 3: The database tool (used only by the Retriever agent)
# =====================================================

# Only the columns the triage step needs, to keep the prompt short.
COLUMNS = (
    "id, customer_name, company_name, service_type, request_description, "
    "priority, assigned_team, resolution_status, contract_value"
)


@tool
def read_records(
    record_id: int = 0,
    priority: str = "",
    resolution_status: str = "",
    service_type: str = "",
    assigned_team: str = "",
) -> str:
    """Read customer operations requests from the database. Read-only.

    Leave every argument empty to get all records. Filters:
    record_id (a single record), priority (low, medium, high, critical),
    resolution_status (open, in_progress, resolved),
    service_type (e.g. Technical Support, Billing, Onboarding),
    assigned_team (e.g. Technical Team, Finance Team).
    """
    try:
        query = table().select(COLUMNS)

        # Add a filter only for the values that were given.
        if record_id > 0:
            query = query.eq("id", record_id)
        if priority:
            query = query.eq("priority", priority)
        if resolution_status:
            query = query.eq("resolution_status", resolution_status)
        if service_type:
            query = query.ilike("service_type", f"%{service_type}%")  # partial match
        if assigned_team:
            query = query.ilike("assigned_team", f"%{assigned_team}%")

        result = query.order("id").limit(100).execute()
        return json.dumps(result.data, default=str)
    except Exception as exc:
        return f"Read failed: {exc}"


# =====================================================
# STEP 4: The two agents
# =====================================================

SKILL_FILE = Path(__file__).parent / "skills" / "escalation_triage.md"

# Agent 1: has the database tool. Its only job is to fetch the right records.
retriever_agent = create_agent(
    model=llm,
    tools=[read_records],
    system_prompt=(
        "You fetch customer operations requests for a triage analyst. "
        "Call read_records exactly once with filters that match the user's question. "
        "If the question is about all requests, or you are unsure, call it with no filters. "
        "Do not analyse the records; just fetch them."
    ),
)

# Agent 2: NO tools. Its instructions are the Markdown skill file.
triage_agent = create_agent(
    model=llm,
    tools=[],
    system_prompt=SKILL_FILE.read_text(encoding="utf-8"),
)


# =====================================================
# STEP 5: LangGraph orchestration
# =====================================================

class State(TypedDict):
    """Data passed from one step of the graph to the next."""
    question: str   # what the user typed
    records: list   # rows found by the Retriever
    report: str     # final answer


def retriever_step(state: State) -> dict:
    """Agent 1: let the Retriever pick filters and fetch the records."""
    response = retriever_agent.invoke({"messages": [HumanMessage(content=state["question"])]})

    # Take the data straight from the tool's output (not from the LLM's
    # summary), so Agent 2 sees exactly what the database returned.
    records = []
    for message in response["messages"]:
        if isinstance(message, ToolMessage):
            try:
                records = json.loads(message.content)
            except ValueError:
                records = []  # tool returned an error message
    return {"records": records}


def triage_step(state: State) -> dict:
    """Agent 2: apply the escalation skill to the fetched records."""
    prompt = (
        f"User question: {state['question']}\n\n"
        f"Records ({len(state['records'])}):\n"
        f"{json.dumps(state['records'], indent=2, default=str)}"
    )
    response = triage_agent.invoke({"messages": [HumanMessage(content=prompt)]})
    return {"report": str(response["messages"][-1].content)}


def nothing_found_step(state: State) -> dict:
    """Used when the Retriever found no records: no need to run Agent 2."""
    return {"report": "No matching records were found, so there is nothing to triage."}


def after_retriever(state: State) -> str:
    """Conditional edge: go to triage only if records were found."""
    return "triage" if state["records"] else "nothing_found"


# Draw the graph:  START -> retriever -> (triage | nothing_found) -> END
workflow = StateGraph(State)
workflow.add_node("retriever", retriever_step)
workflow.add_node("triage", triage_step)
workflow.add_node("nothing_found", nothing_found_step)

workflow.add_edge(START, "retriever")
workflow.add_conditional_edges(
    "retriever",
    after_retriever,
    {"triage": "triage", "nothing_found": "nothing_found"},
)
workflow.add_edge("triage", END)
workflow.add_edge("nothing_found", END)

graph = workflow.compile()


# =====================================================
# STEP 6: Chat loop in the terminal
# =====================================================

def ask(question: str) -> dict:
    """Send one question through both agents and return the final state."""
    return graph.invoke({"question": question, "records": [], "report": ""})


def main():
    """Read questions from the keyboard until the user types 'exit'."""
    print("=" * 60)
    print("TWO-AGENT ESCALATION TRIAGE")
    print(f"Table: {SUPABASE_TABLE}   |   LLM: {OLLAMA_MODEL}")
    print("Agents: Retriever -> Triage analyst (skill: escalation_triage.md)")
    print("=" * 60)
    print("\nTry:")
    print("  Which requests need escalation?")
    print("  Triage the open technical support requests")
    print("  Review the Finance Team's requests")
    print("\nType 'exit' to quit.\n")

    while True:
        question = input("You: ").strip()

        if question.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        if not question:
            continue

        try:
            result = ask(question)
            print(f"\n[Retriever found {len(result['records'])} record(s)]\n")
            print(result["report"], "\n")
        except Exception as exc:
            print(f"\nError: {exc}\n")


if __name__ == "__main__":
    main()

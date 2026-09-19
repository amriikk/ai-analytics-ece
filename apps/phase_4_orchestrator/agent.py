from typing import TypedDict, Any
from langgraph.graph import StateGraph, END
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

# Import the compiled apps from previous phases
from apps.phase_1_codegen.agent import app as codegen_app
from apps.phase_3_guardrails.agent import app as guardrails_app

# 1. Define the Orchestrator State and Schema
class OrchestratorState(TypedDict):
    question: str
    csv_path: str
    routing_decision: str
    final_answer: str

class RouteDecision(BaseModel):
    route: str = Field(description="Must be exactly 'DETERMINISTIC' or 'EXPERIMENTAL'")
    reason: str = Field(description="Brief reason for the routing decision")

# Initialize the router LLM
llm = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0)
router_llm = llm.with_structured_output(RouteDecision)

# 2. Define the Supervisor Nodes
def route_query(state: OrchestratorState):
    print("\n[Orchestrator] Analyzing query for optimal routing...")
    
    system_prompt = SystemMessage(content="""
    You are an AI supervisor for a data engineering platform.
    Analyze the user's question and choose a routing path.
    
    Routes:
    - 'DETERMINISTIC': Use for standard operations (filtering, sorting, aggregation, selecting columns). This is safe and uses pre-defined JSON operators.
    - 'EXPERIMENTAL': Use for complex tasks requiring custom math, text processing, or open-ended logic that standard operators cannot handle.
    """)
    
    user_prompt = HumanMessage(content=f"Question: {state['question']}")
    decision = router_llm.invoke([system_prompt, user_prompt])
    
    print(f"  -> Decision: {decision.route} ({decision.reason})")
    return {"routing_decision": decision.route}

def execute_deterministic(state: OrchestratorState):
    print("[Orchestrator] Handing off to Phase 3 Guardrails Agent (JSON Planner)...")
    result = guardrails_app.invoke({"question": state["question"], "csv_path": state["csv_path"]})
    return {"final_answer": result.get("final_answer", "No answer found.")}

def execute_experimental(state: OrchestratorState):
    print("[Orchestrator] Handing off to Phase 1 Code-Gen Agent (Raw Python)...")
    # Provide the initial error_count expected by Phase 1
    result = codegen_app.invoke({"question": state["question"], "csv_path": state["csv_path"], "error_count": 0})
    return {"final_answer": result.get("final_answer", "No answer found.")}

# 3. Build the Orchestrator Graph
workflow = StateGraph(OrchestratorState)

workflow.add_node("route_query", route_query)
workflow.add_node("execute_deterministic", execute_deterministic)
workflow.add_node("execute_experimental", execute_experimental)

workflow.set_entry_point("route_query")

def route_edge(state: OrchestratorState):
    if state["routing_decision"] == "DETERMINISTIC":
        return "execute_deterministic"
    return "execute_experimental"

workflow.add_conditional_edges(
    "route_query",
    route_edge,
    {
        "execute_deterministic": "execute_deterministic",
        "execute_experimental": "execute_experimental"
    }
)

workflow.add_edge("execute_deterministic", END)
workflow.add_edge("execute_experimental", END)

app = workflow.compile()

if __name__ == "__main__":
    dataset_path = "data/most-streamed-spotify-songs-2024.csv"
    
    # Test 1: Simple query (Should route to Deterministic)
    test_state_1 = {
        "question": "What are the top 5 tracks sorted by YouTube Views?",
        "csv_path": dataset_path
    }
    
    print("\n================== TEST 1 ==================")
    result_1 = app.invoke(test_state_1)
    print(f"\nFINAL ANSWER:\n{result_1.get('final_answer')}")
    
    # Test 2: Complex query (Should route to Experimental)
    test_state_2 = {
        "question": "Write a custom correlation matrix comparing Spotify Popularity to TikTok Likes and summarize the trend.",
        "csv_path": dataset_path
    }
    
    print("\n================== TEST 2 ==================")
    result_2 = app.invoke(test_state_2)
    print(f"\nFINAL ANSWER:\n{result_2.get('final_answer')}")
import pandas as pd
import traceback
from typing import TypedDict, List, Dict, Any, Literal
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage
from libs.data_hygiene.cleaner import get_cached_dataframe
from libs.shared_operators.operators import OPERATOR_DISPATCH
from libs.llm import make_llm

# 1. Define the Strict JSON Schema for the Planner
class Step(BaseModel):
    # Literal means Pydantic rejects any operator name that isn't on this list
    operator: Literal[
        "select_columns", "filter_rows", "derive_columns", "group_and_aggregate",
        "sort_rows", "limit_rows", "distinct_rows",
    ]
    arguments: Dict[str, Any] = Field(description="The exact arguments required by the operator schema")

class ExecutionPlan(BaseModel):
    steps: List[Step] = Field(description="The sequential list of operations to run against the dataset")

# 2. Define the State
class AgentState(TypedDict):
    question: str
    csv_path: str
    plan: List[Dict[str, Any]]
    execution_result: str
    final_answer: str

# Initialize the LLM
llm = make_llm()
# Bind the Pydantic schema to force JSON output
json_llm = llm.with_structured_output(ExecutionPlan)

# 3. Define the Nodes
def generate_plan(state: AgentState):
    print("\n[Node: generate_plan] Drafting deterministic execution plan...")
    df = get_cached_dataframe(state["csv_path"])
    
    system_prompt = SystemMessage(content=f"""
    You are a data architect. Map the user's question to a sequence of strict operations.
    Available columns: {list(df.columns)}
    
    Available operators and the exact arguments each one takes:
    - select_columns: {{"columns": ["Track", "Artist"]}}
    - filter_rows: {{"conditions": [{{"column": "Explicit Track", "operator": "==", "value": 1}}]}}
      operator is one of ==, !=, >, <, >=, <=, contains. All conditions must match.
    - derive_columns: {{"derive": [{{"new_column": "likes_per_view", "type": "arithmetic", "operation": "divide",
        "left": {{"type": "column", "value": "YouTube Likes"}}, "right": {{"type": "column", "value": "YouTube Views"}}}}]}}
      operation is one of add, subtract, multiply, divide. For a fixed number use {{"type": "number", "value": 1000}}.
      Or: {{"derive": [{{"new_column": "release_year", "type": "extract_date_part", "column": "Release Date", "part": "year"}}]}}
    - group_and_aggregate: {{"group_by": ["Artist"], "metrics": [{{"column": "Spotify Streams", "function": "sum", "as": "total_streams"}}]}}
      function is one of sum, mean, median, min, max, count.
    - sort_rows: {{"sort_by": [{{"column": "Spotify Streams", "direction": "desc"}}]}}
    - limit_rows: {{"k": 5}}
    - distinct_rows: {{"columns": ["Artist"]}}
    
    Only output the structured plan.
    """)
    
    user_prompt = HumanMessage(content=f"Question: {state['question']}")
    
    # This automatically returns a validated Pydantic object
    plan_obj = json_llm.invoke([system_prompt, user_prompt])
    
    # Convert back to dict for state storage
    return {"plan": [step.model_dump() for step in plan_obj.steps]}

def execute_plan(state: AgentState):
    print("[Node: execute_plan] Executing plan against shared operators...")
    df = get_cached_dataframe(state["csv_path"])
    current_df = df.copy()
    
    try:
        for i, step in enumerate(state["plan"]):
            op_name = step["operator"]
            args = step["arguments"]
            print(f"  -> Step {i+1}: Running {op_name}...")
            
            if op_name not in OPERATOR_DISPATCH:
                raise ValueError(f"Unknown operator: {op_name}")
                
            # Safely dispatch to your deterministic functions
            func = OPERATOR_DISPATCH[op_name]
            current_df = func(current_df, **args)
            
        # Convert the final resulting dataframe to a string representation
        result_str = current_df.to_string()
        return {"execution_result": result_str}
        
    except Exception as e:
        error_msg = f"Execution failed at step {i+1} ({op_name}): {str(e)}"
        print(f"  -> {error_msg}")
        return {"execution_result": error_msg}

def generate_answer(state: AgentState):
    print("[Node: generate_answer] Translating data back to natural language...")
    
    if "Execution failed" in state["execution_result"]:
        return {"final_answer": f"I couldn't run the data plan. {state['execution_result']}"}
        
    system_prompt = SystemMessage(content="""
    Generate a concise natural language answer based ONLY on the provided execution result table.
    """)
    
    user_prompt = HumanMessage(content=f"Question: {state['question']}\nResult:\n{state['execution_result']}")
    response = llm.invoke([system_prompt, user_prompt])
    
    return {"final_answer": response.text.strip()}

# 4. Build the Graph
workflow = StateGraph(AgentState)

workflow.add_node("generate_plan", generate_plan)
workflow.add_node("execute_plan", execute_plan)
workflow.add_node("generate_answer", generate_answer)

workflow.set_entry_point("generate_plan")
workflow.add_edge("generate_plan", "execute_plan")
workflow.add_edge("execute_plan", "generate_answer")
workflow.add_edge("generate_answer", END)

app = workflow.compile()

if __name__ == "__main__":
    dataset_path = "data/most-streamed-spotify-songs-2024.csv"
    
    test_state = {
        "question": "What are the top 3 tracks by Spotify Streams? Give me the track names and their streams.",
        "csv_path": dataset_path
    }
    
    print(f"\nUser Query: '{test_state['question']}'")
    result = app.invoke(test_state)
    
    print("\n--- FINAL OUTPUT ---")
    print(result.get("final_answer"))
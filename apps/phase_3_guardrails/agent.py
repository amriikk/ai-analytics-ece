import pandas as pd
import traceback
from typing import TypedDict, List, Dict, Any
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from libs.data_hygiene.cleaner import get_cached_dataframe
from libs.shared_operators.operators import OPERATOR_DISPATCH

# 1. Define the Strict JSON Schema for the Planner
class Step(BaseModel):
    operator: str = Field(description="Must be one of: select_columns, filter_rows, derive_columns, group_and_aggregate, sort_rows, limit_rows, distinct_rows")
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
llm = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0)
# Bind the Pydantic schema to force JSON output
json_llm = llm.with_structured_output(ExecutionPlan)

# 3. Define the Nodes
def generate_plan(state: AgentState):
    print("\n[Node: generate_plan] Drafting deterministic execution plan...")
    df = get_cached_dataframe(state["csv_path"])
    
    system_prompt = SystemMessage(content=f"""
    You are a data architect. Map the user's question to a sequence of strict operations.
    Available columns: {list(df.columns)}
    
    Available operators:
    - select_columns(columns: List[str])
    - filter_rows(conditions: List[Dict(column, operator, value)])
    - group_and_aggregate(group_by: List[str], metrics: List[Dict(column, function, as)])
    - sort_rows(sort_by: List[Dict(column, direction)])
    - limit_rows(k: int)
    
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
        error_msg = f"Execution failed: {str(e)}"
        print(f"  -> {error_msg}")
        return {"execution_result": error_msg}

def generate_answer(state: AgentState):
    print("[Node: generate_answer] Translating data back to natural language...")
    
    if "Execution failed" in state["execution_result"]:
        return {"final_answer": "I encountered a logical error while processing the data plan."}
        
    system_prompt = SystemMessage(content="""
    Generate a concise natural language answer based ONLY on the provided execution result table.
    """)
    
    user_prompt = HumanMessage(content=f"Question: {state['question']}\nResult:\n{state['execution_result']}")
    response = llm.invoke([system_prompt, user_prompt])
    
    return {"final_answer": response.content.strip()}

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
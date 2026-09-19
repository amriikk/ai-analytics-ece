import operator
import re
import traceback
import pandas as pd
from typing import TypedDict, Any
from langgraph.graph import StateGraph, END
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from libs.data_hygiene.cleaner import load_and_clean_spotify_data

class AgentState(TypedDict):
    question: str
    csv_path: str
    dataframe: Any  
    generated_code: str
    execution_result: str
    evaluation: str
    final_answer: str
    error_count: int

# Initialize the LLM (Ensure ANTHROPIC_API_KEY is exported in your terminal)
llm = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0)

def extract_python_code(text: str) -> str:
    """Extracts python code from markdown block."""
    match = re.search(r"```python(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()

def generate_code(state: AgentState):
    print("\n[Node: generate_code] Synthesizing pandas pipeline...")
    df = state["dataframe"]
    columns = list(df.columns)
    
    system_prompt = SystemMessage(content=f"""
    You are a highly capable data engineering AI. 
    Write Python code using pandas to answer the user's question.
    
    Rules:
    1. Assume the dataframe is already loaded in a variable named `df`.
    2. Store the final output in a variable named `result`.
    3. Output ONLY the Python code inside a ```python ``` block.
    
    Available Columns: {columns}
    """)
    
    user_prompt = HumanMessage(content=f"Question: {state['question']}")
    response = llm.invoke([system_prompt, user_prompt])
    
    code = extract_python_code(response.content)
    return {"generated_code": code, "error_count": state.get("error_count", 0)}

def execute_code(state: AgentState):
    print("[Node: execute_code] Executing generated pipeline...")
    code = state["generated_code"]
    df = state["dataframe"]
    
    # Create an isolated execution environment
    env = {"pd": pd, "df": df}
    
    try:
        exec(code, env)
        result = env.get("result", "Error: 'result' variable not found.")
        return {"execution_result": str(result)}
    except Exception as e:
        error_msg = f"Execution Error: {str(e)}\n{traceback.format_exc()}"
        print(f"  -> {error_msg.splitlines()[0]}")
        return {"execution_result": error_msg}

def evaluate_result(state: AgentState):
    print("[Node: evaluate_result] Validating output...")
    execution_result = state["execution_result"]
    
    if "Execution Error" in execution_result or "Error:" in execution_result:
        print("  -> Execution failed.")
        return {"evaluation": "FAIL", "error_count": state.get("error_count", 0) + 1}
        
    system_prompt = SystemMessage(content="""
    You are a validation agent. Evaluate if the execution result answers the question.
    Respond with EXACTLY 'PASS' if the result is valid and answers the question.
    Respond with EXACTLY 'FAIL' if the result is incorrect, empty, or invalid.
    """)
    
    user_prompt = HumanMessage(content=f"Question: {state['question']}\nResult: {execution_result}")
    response = llm.invoke([system_prompt, user_prompt])
    
    eval_status = response.content.strip().upper()
    print(f"  -> Validation status: {eval_status}")
    
    new_error_count = state.get("error_count", 0)
    if eval_status == "FAIL":
        new_error_count += 1
        
    return {"evaluation": eval_status, "error_count": new_error_count}

def generate_answer(state: AgentState):
    print("[Node: generate_answer] Translating to natural language...")
    
    if state["evaluation"] == "FAIL":
        return {"final_answer": "I was unable to compute a correct answer after multiple attempts."}
        
    system_prompt = SystemMessage(content="""
    Generate a clear, concise natural language answer based ONLY on the provided execution result.
    Do not mention the code or dataframe. Just answer the user's question directly.
    """)
    
    user_prompt = HumanMessage(content=f"Question: {state['question']}\nResult: {state['execution_result']}")
    response = llm.invoke([system_prompt, user_prompt])
    
    return {"final_answer": response.content.strip()}

def route_evaluation(state: AgentState):
    if state["evaluation"] == "FAIL" and state["error_count"] < 1:
        print("  -> Retrying code generation...")
        return "generate_code"
    return "generate_answer"

# Build the Graph
workflow = StateGraph(AgentState)

workflow.add_node("generate_code", generate_code)
workflow.add_node("execute_code", execute_code)
workflow.add_node("evaluate_result", evaluate_result)
workflow.add_node("generate_answer", generate_answer)

workflow.set_entry_point("generate_code")
workflow.add_edge("generate_code", "execute_code")
workflow.add_edge("execute_code", "evaluate_result")
workflow.add_conditional_edges(
    "evaluate_result",
    route_evaluation,
    {
        "generate_code": "generate_code",
        "generate_answer": "generate_answer"
    }
)
workflow.add_edge("generate_answer", END)

app = workflow.compile()

if __name__ == "__main__":
    dataset_path = "data/most-streamed-spotify-songs-2024.csv"
    print(f"Loading and sanitizing dataset from {dataset_path}...")
    df_clean = load_and_clean_spotify_data(dataset_path)
    
    test_state = {
        "question": "What is the average Spotify Popularity for explicit tracks vs non-explicit tracks?",
        "csv_path": dataset_path,
        "dataframe": df_clean,
        "error_count": 0
    }
    
    print(f"\nUser Query: '{test_state['question']}'")
    result = app.invoke(test_state)
    
    print("\n--- FINAL OUTPUT ---")
    print(result["final_answer"])
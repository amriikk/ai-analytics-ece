import operator
from typing import TypedDict, Annotated, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_anthropic import ChatAnthropic
from libs.data_hygiene.cleaner import load_and_clean_spotify_data

# 1. Define the State
class AgentState(TypedDict):
    question: str
    csv_path: str
    dataframe: Any  # Pandas DataFrame
    generated_code: str
    execution_result: str
    evaluation: str
    final_answer: str
    error_count: int

# 2. Initialize the LLM
# Make sure ANTHROPIC_API_KEY is in your environment variables
llm = ChatAnthropic(model="claude-3-5-sonnet-20240620", temperature=0)

# 3. Define the Nodes (Stubs for now)
def generate_code(state: AgentState):
    print("-> Generating Code")
    # TODO: Prompt LLM to write pandas code
    return {"generated_code": "result = df.head()", "error_count": state.get("error_count", 0)}

def execute_code(state: AgentState):
    print("-> Executing Code")
    # TODO: Execute code safely using exec()
    return {"execution_result": "Success"}

def evaluate_result(state: AgentState):
    print("-> Evaluating Result")
    # TODO: Check if result answers the question
    return {"evaluation": "PASS"}

def generate_answer(state: AgentState):
    print("-> Generating Final Answer")
    # TODO: Translate execution result to natural language
    return {"final_answer": "Here is the answer."}

# Routing function for the retry logic
def route_evaluation(state: AgentState):
    if state["evaluation"] == "FAIL" and state["error_count"] < 1:
        return "generate_code"
    return "generate_answer"

# 4. Build the Graph
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
    # Test the skeleton
    test_state = {
        "question": "What are the top 5 most streamed tracks?",
        "csv_path": "../../data/most-streamed-spotify-songs-2024.csv",
        "error_count": 0
    }
    
    print("Running Phase 1 Agent Skeleton...")
    result = app.invoke(test_state)
    print("Workflow complete.")
# Iterative AI Agent Architecture

An enterprise-grade monorepo demonstrating a progressive 4-phase LLM orchestration system. Built to process, analyze, and query tabular datasets using a blend of dynamic code generation and deterministic execution guardrails.

This architecture transitions a standard LangChain script into a stateful, production-ready microservice capable of safe, automated data analytics.

## 🏗 System Architecture

The project is structured into four distinct iterative phases, culminating in a Supervisor-led routing engine:

*   **Phase 1: Generative Code Agent (`apps/phase_1_codegen`)**
    A foundational ReAct loop where the LLM writes raw Pandas Python code, executes it in an isolated environment, and evaluates its own output. Includes automatic retry logic for syntax or execution failures.
*   **Phase 2: Stateful API & Memory (`apps/phase_2_stateful`)**
    Wraps the Phase 1 agent in a FastAPI backend and introduces LangGraph's `MemorySaver`. This decouples the heavy DataFrame from the conversation state (using an in-memory cache) while persisting thread history, allowing for contextual follow-up queries.
*   **Phase 3: Deterministic Guardrails (`apps/phase_3_guardrails`)**
    Replaces arbitrary Python execution with a strict "Plan-and-Execute" architecture. The LLM is forced via Pydantic to output a validated JSON schema, which is then mapped to 7 locked-down, deterministic Pandas operators (`select_columns`, `group_and_aggregate`, etc.) to guarantee bug-free, predictable execution.
*   **Phase 4: The Orchestrator (`apps/phase_4_orchestrator`)**
    A Supervisor node that acts as the routing brain. It analyzes user intent and dynamically routes standard analytical queries to the safe Phase 3 Guardrails agent, while routing complex statistical requests (e.g., correlation matrices) to the flexible Phase 1 Experimental agent.

## 🛠 Tech Stack

*   **Dependency Management:** Poetry
*   **Orchestration & State:** LangGraph, LangChain
*   **API Layer:** FastAPI, Uvicorn
*   **Data Processing:** Pandas, Pydantic (Schema Validation)
*   **LLM Provider:** Anthropic (Claude 3.5 Haiku / Sonnet)

## 🚀 Getting Started

### Prerequisites
Ensure you have Python 3.13+ and Poetry installed. 

### Installation
1. Clone the repository and navigate to the root directory.
2. Activate the Poetry virtual environment:
   ```bash
   poetry env activate
3. Install dependencies:
    ```bash
    poetry install
4. Export your API key:
    ```bash
    export ANTHROPIC_API_KEY="your-api-key-here"

### Running the Orchestrator
    ```bash
    python -m apps.phase_4_orchestrator.agent

### Running the Stateful API Server
To spin up the Phase 2 backend microservice:

    ```bash
    python -m apps.phase_2_stateful.server

Test the API endpoint via curl:

    ```bash
    curl -X POST http://localhost:8000/api/v1/analyze \
     -H "Content-Type: application/json" \
     -d '{"question": "How many total rows are in the dataset?", "session_id": "test-123"}'
     
## Author
#### Jhon Trujillo
##### jotrio.com
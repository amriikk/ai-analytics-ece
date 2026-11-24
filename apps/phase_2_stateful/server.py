import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from apps.phase_1_codegen.agent import app as analytics_agent

app = FastAPI(
    title="AI Analytics API",
    description="Stateful LLM-powered data analytics and extraction engine."
)

DATA_PATH = "data/most-streamed-spotify-songs-2024.csv"

class QueryRequest(BaseModel):
    question: str
    session_id: str = "default_session"

# A plain "def" (not "async def"): the agent makes slow, blocking API calls,
# so FastAPI runs each request in its own worker thread instead of freezing the server.
@app.post("/api/v1/analyze")
def analyze_data(request: QueryRequest):
    try:
        # State no longer contains the massive DataFrame object
        state = {
            "question": request.question,
            "csv_path": DATA_PATH,
            "error_count": 0
        }
        
        # Configure the thread ID for memory persistence
        config = {"configurable": {"thread_id": request.session_id}}
        
        result = analytics_agent.invoke(state, config=config)
        
        return {
            "status": "success",
            "session_id": request.session_id,
            "question": request.question,
            "final_answer": result.get("final_answer", "No answer generated."),
            "evaluation_status": result.get("evaluation", "N/A").split("\n")[0],
            "turns_in_session": len(result.get("history", []))
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("Starting Stateful Analytics API on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

import apps.phase_1_codegen.agent as codegen
import apps.phase_2_stateful.server as server
import apps.phase_3_guardrails.agent as guardrails
import apps.phase_4_orchestrator.agent as orchestrator
from libs.data_hygiene.cleaner import get_cached_dataframe
from tests.conftest import FakeLLM


def ask(question, csv_path, session):
    config = {"configurable": {"thread_id": session}}
    return codegen.app.invoke({"question": question, "csv_path": csv_path, "error_count": 0}, config=config)


# ---------- Phase 1: code generation ----------

def test_retry_shows_the_model_what_went_wrong(monkeypatch, csv_path):
    fake = FakeLLM([
        "```python\nresult = df['No such column'].mean()\n```",
        "```python\nresult = len(df)\n```",
        "PASS",
        "There are 4,600 tracks.",
    ])
    monkeypatch.setattr(codegen, "llm", fake)
    out = ask("How many tracks?", csv_path, "retry")
    retry_prompt = fake.prompts[1][1].content
    assert "What went wrong" in retry_prompt and "KeyError" in retry_prompt
    assert out["final_answer"] == "There are 4,600 tracks."


def test_validator_reason_is_fed_back(monkeypatch, csv_path):
    fake = FakeLLM([
        "```python\nresult = 0\n```",
        "FAIL: the result is a constant, not a count.",
        "```python\nresult = len(df)\n```",
        "PASS",
        "4,600",
    ])
    monkeypatch.setattr(codegen, "llm", fake)
    ask("How many tracks?", csv_path, "validator")
    assert "not a count" in fake.prompts[2][1].content


def test_gives_up_after_max_attempts(monkeypatch, csv_path):
    monkeypatch.setattr(codegen, "llm", FakeLLM(["```python\nx = 1\n```", "```python\nx = 2\n```"]))
    out = ask("How many tracks?", csv_path, "give-up")
    assert out["final_answer"] == f"I was unable to compute a correct answer after {codegen.MAX_ATTEMPTS} attempts."


def test_generated_code_cannot_damage_the_cached_data(monkeypatch, csv_path):
    monkeypatch.setattr(codegen, "llm", FakeLLM(["```python\ndf['Track'] = None\nresult = 1\n```", "PASS", "ok"]))
    ask("Anything", csv_path, "damage")
    assert get_cached_dataframe(csv_path)["Track"].notna().all()


def test_follow_up_questions_see_earlier_turns(monkeypatch, csv_path):
    fake = FakeLLM([
        "```python\nresult = 'Taylor Swift'\n```", "PASS", "Taylor Swift has the most streams.",
        "```python\nresult = 123\n```", "PASS", "She has 123 YouTube views.",
    ])
    monkeypatch.setattr(codegen, "llm", fake)
    ask("Which artist has the most streams?", csv_path, "memory")
    out = ask("What about her YouTube views?", csv_path, "memory")
    second_code_prompt = fake.prompts[3][1].content
    assert "Which artist has the most streams?" in second_code_prompt
    assert "Taylor Swift has the most streams." in second_code_prompt
    assert len(out["history"]) == 2


def test_sessions_do_not_share_history(monkeypatch, csv_path):
    monkeypatch.setattr(codegen, "llm", FakeLLM(["```python\nresult = 1\n```", "PASS", "one"] * 2))
    ask("First", csv_path, "session-a")
    out = ask("First", csv_path, "session-b")
    assert len(out["history"]) == 1


# ---------- Phase 2: API ----------

def test_api_answers_and_counts_turns(monkeypatch):
    monkeypatch.setattr(codegen, "llm", FakeLLM(["```python\nresult = len(df)\n```", "PASS", "4,600 rows."] * 2))
    client = TestClient(server.app)
    body = {"question": "How many rows?", "session_id": "api-test"}
    first = client.post("/api/v1/analyze", json=body)
    second = client.post("/api/v1/analyze", json=body)
    assert first.status_code == 200
    assert first.json()["final_answer"] == "4,600 rows."
    assert second.json()["turns_in_session"] == 2


# ---------- Phase 3: guardrails ----------

def test_planner_prompt_describes_all_seven_operators(monkeypatch, csv_path):
    class FakePlanner:
        def invoke(self, messages):
            self.prompt = messages[0].content
            return guardrails.ExecutionPlan(steps=[{"operator": "limit_rows", "arguments": {"k": 1}}])

    planner = FakePlanner()
    monkeypatch.setattr(guardrails, "json_llm", planner)
    monkeypatch.setattr(guardrails, "llm", FakeLLM(["one row"]))
    guardrails.app.invoke({"question": "q", "csv_path": csv_path})
    for name in guardrails.OPERATOR_DISPATCH:
        assert name in planner.prompt


def test_plan_runs_against_real_data(monkeypatch, csv_path):
    plan = guardrails.ExecutionPlan(steps=[
        {"operator": "group_and_aggregate", "arguments": {"group_by": ["Artist"], "metrics": [
            {"column": "Spotify Streams", "function": "sum", "as": "total_streams"}]}},
        {"operator": "sort_rows", "arguments": {"sort_by": [{"column": "total_streams", "direction": "desc"}]}},
        {"operator": "limit_rows", "arguments": {"k": 1}},
    ])
    monkeypatch.setattr(guardrails, "json_llm", type("P", (), {"invoke": lambda self, m: plan})())
    monkeypatch.setattr(guardrails, "llm", FakeLLM(["answer"]))
    out = guardrails.app.invoke({"question": "q", "csv_path": csv_path})
    # Same question answered directly with pandas: the plan must agree
    top_artist = get_cached_dataframe(csv_path).groupby("Artist")["Spotify Streams"].sum().idxmax()
    assert top_artist in out["execution_result"]


def test_unknown_operator_is_rejected():
    with pytest.raises(ValidationError):
        guardrails.Step(operator="drop_table", arguments={})


# ---------- Phase 4: router ----------

def test_unknown_route_is_rejected():
    with pytest.raises(ValidationError):
        orchestrator.RouteDecision(route="MAYBE", reason="unsure")
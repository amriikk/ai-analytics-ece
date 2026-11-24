import os

import pytest
from langchain_core.messages import AIMessage

# The agents create their Claude client when they're imported. A placeholder key
# lets that happen; no test ever calls the real API.
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-not-used")

CSV = "data/most-streamed-spotify-songs-2024.csv"


class FakeLLM:
    """Stands in for Claude: returns scripted replies and records every prompt."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    def invoke(self, messages):
        self.prompts.append(messages)
        # Shaped like a Haiku 5.5 reply: a thinking block before the text
        return AIMessage(content=[
            {"type": "thinking", "thinking": "", "signature": "test"},
            {"type": "text", "text": self.replies.pop(0)},
        ])


@pytest.fixture
def csv_path():
    return CSV
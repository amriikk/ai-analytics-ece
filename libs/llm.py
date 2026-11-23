import os

from langchain_anthropic import ChatAnthropic

# Claude Haiku 5.5: Anthropic's fast, low-cost model, a good fit for routing,
# planning and short pandas tasks. To try another model without editing code:
#   export ANTHROPIC_MODEL="claude-sonnet-5-5"
DEFAULT_MODEL = "claude-haiku-5-5"


def make_llm() -> ChatAnthropic:
    """One place to create the chat model that every agent uses."""
    return ChatAnthropic(
        model=os.getenv("ANTHROPIC_MODEL", DEFAULT_MODEL),
        # Haiku 5.5 thinks before it answers, and thinking counts toward this
        # limit. Set it explicitly so long answers aren't cut off.
        max_tokens=16000,
    )
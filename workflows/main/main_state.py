#worflows/main/main_state.py
from typing import TypedDict, Optional, Sequence


class MainState(TypedDict):
    """State used by the LangGraph MainWorkflow."""
    rejected: bool
    reason_rejected: Optional[str]
    chat_history: str
    language: Optional[str]
    instructions: Optional[Sequence[str]]

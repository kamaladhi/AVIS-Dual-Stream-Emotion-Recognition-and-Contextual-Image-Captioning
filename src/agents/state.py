from typing import Any, Dict, List, TypedDict


class AgentState(TypedDict):
    """
    State dictionary for the AVIS LangGraph Agent.
    This holds the visual context as it flows through the LLM reasoning nodes.
    """

    image_path: str
    base_caption: str
    faces_data: List[Dict[str, Any]]
    final_caption: str

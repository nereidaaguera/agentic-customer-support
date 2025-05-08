"""Tool implementations for agent use."""

from collections.abc import Callable

from pydantic import BaseModel


class ToolInfo(BaseModel):
    """Class representing a tool for the agent."""

    name: str
    spec: dict
    exec_fn: Callable

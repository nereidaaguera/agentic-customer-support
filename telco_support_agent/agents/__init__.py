"""Telco Agents."""

from typing import Any

from pydantic import BaseModel, Field

from telco_support_agent.tools import ToolConfig


class LLMConfig(BaseModel):
    """Model for LLM configuration from YAML."""

    endpoint: str
    params: dict[str, Any] = Field(default_factory=dict)


class AgentConfig(BaseModel):
    """Model for agent configuration from YAML."""

    name: str
    description: str
    llm: LLMConfig
    system_prompt: str
    tools: list[ToolConfig] = Field(default_factory=list)

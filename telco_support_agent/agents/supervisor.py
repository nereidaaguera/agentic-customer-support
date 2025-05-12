"""Supervisor agent to orchestrate specialized sub-agents."""

import mlflow
from mlflow.entities import SpanType

from telco_support_agent.agents.base_agent import BaseAgent
from telco_support_agent.tools import ToolInfo
from telco_support_agent.utils.config import load_agent_config


class SupervisorAgent(BaseAgent):
    """Supervisor agent to orchestrate specialized sub-agents.

    This agent analyzes customer queries and routes them to the appropriate
    specialized agent based on query content and intent.
    """

    def __init__(self, llm_endpoint: str):
        """Init supervisor agent.

        Args:
            llm_endpoint: Name of the LLM endpoint to use
        """
        # Load config
        self.config = load_agent_config("supervisor")
        system_prompt = self.config.get("system_prompt", "")

        # Create routing tool
        routing_tool = self._create_routing_tool()

        # init base agent with routing tool
        super().__init__(
            llm_endpoint=llm_endpoint, tools=[routing_tool], system_prompt=system_prompt
        )

        # placeholders for sub-agents
        # self.account_agent = None
        # self.billing_agent = None
        # self.tech_support_agent = None
        # self.product_agent = None

    def _create_routing_tool(self) -> ToolInfo:
        """Create the routing tool from configuration.

        Returns:
            Configured routing tool
        """
        # Get tool configs
        tool_configs = self.config.get("tools", [])

        # routing tool config
        for tool_config in tool_configs:
            if tool_config["name"] == "route_to_specialized_agent":
                return ToolInfo(
                    name=tool_config["name"],
                    spec={
                        "type": "function",
                        "function": {
                            "name": tool_config["name"],
                            "description": tool_config["description"],
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "agent_type": {
                                        "type": "string",
                                        "enum": tool_config["parameters"]["agent_type"][
                                            "enum"
                                        ],
                                        "description": tool_config["parameters"][
                                            "agent_type"
                                        ]["description"],
                                    },
                                    "reason": {
                                        "type": "string",
                                        "description": tool_config["parameters"][
                                            "reason"
                                        ]["description"],
                                    },
                                },
                                "required": ["agent_type", "reason"],
                            },
                        },
                    },
                    exec_fn=self.route_to_specialized_agent,
                )

    @mlflow.trace(span_type=SpanType.TOOL)
    def route_to_specialized_agent(self, agent_type: str, reason: str) -> str:
        """Route query to specialized agent.

        TODO: for now just return information about where the query would be routed
        need to call the appropriate sub-agent

        Args:
            agent_type: Sub-agent to route to
            reason: Reason for routing to sub-agent

        Returns:
            Response describing the routing decision
        """
        # TODO: testing - return a formatted response about routing
        agent_descriptions = {
            "account": "customer account information, profile details, and account management",
            "billing": "billing inquiries, payment information, and usage details",
            "tech_support": "technical issues, troubleshooting, and device setup assistance",
            "product": "product information, plan comparisons, and promotional offers",
        }

        response = f"ROUTING DECISION: Route query to {agent_type.upper()} AGENT\n\n"
        response += f"REASON: {reason}\n\n"

        if agent_type in agent_descriptions:
            response += f"NOTE: The {agent_type.upper()} AGENT specializes in {agent_descriptions[agent_type]}."

        return response

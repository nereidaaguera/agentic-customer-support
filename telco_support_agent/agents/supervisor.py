"""Supervisor agent to orchestrate specialized sub-agents."""


import mlflow
from mlflow.entities import SpanType

# from telco_support_agent.agents.account import AccountAgent
from telco_support_agent.agents.base_agent import BaseAgent

# from telco_support_agent.agents.billing import BillingAgent
# from telco_support_agent.agents.product import ProductAgent
# from telco_support_agent.agents.tech_support import TechSupportAgent
from telco_support_agent.tools import ToolInfo


class SupervisorAgent(BaseAgent):
    """Supervisor agent that orchestrates specialized sub-agents."""

    def __init__(self, llm_endpoint: str):
        """Initialize the supervisor agent with specialized sub-agents.

        Args:
            llm_endpoint: Name of the LLM endpoint to use
        """
        system_prompt = """"""  # TODO

        # init specialized agents
        # self.account_agent = AccountAgent(llm_endpoint)
        # self.billing_agent = BillingAgent(llm_endpoint)
        # self.tech_support_agent = TechSupportAgent(llm_endpoint)
        # self.product_agent = ProductAgent(llm_endpoint)

        # tool for routing to specialized agents
        routing_tool = ToolInfo(
            name="route_to_specialized_agent",
            spec={
                "type": "function",
                "function": {
                    "name": "route_to_specialized_agent",
                    "description": "Route a query to a specialized agent",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "agent_type": {
                                "type": "string",
                                "enum": [
                                    "account",
                                    "billing",
                                    "tech_support",
                                    "product",
                                ],
                                "description": "The type of specialized agent to route to",
                            },
                            "reason": {
                                "type": "string",
                                "description": "The reason for routing to this agent type",
                            },
                        },
                        "required": ["agent_type", "reason"],
                    },
                },
            },
            exec_fn=self.route_to_specialized_agent,
        )

        # init base agent with the routing tool
        super().__init__(
            llm_endpoint=llm_endpoint, tools=[routing_tool], system_prompt=system_prompt
        )

    @mlflow.trace(span_type=SpanType.TOOL)
    def route_to_specialized_agent(self, agent_type: str, reason: str) -> str:
        """Route query to specialized agent.

        Args:
            agent_type: Sub-agent to route to
            reason: Reason for routing to sub-agent

        Returns:
            Response from the sub-agent
        """
        pass  # TODO: Implement routing logic to specialized agents

"""Supervisor agent to orchestrate specialized sub-agents."""

import logging
from typing import Optional

import mlflow
from mlflow.entities import SpanType

from telco_support_agent.agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class SupervisorAgent(BaseAgent):
    """Supervisor agent to orchestrate specialized sub-agents.

    This agent analyzes customer queries and routes them to the appropriate
    sub-agent based on query content and intent.
    """

    def __init__(
        self,
        llm_endpoint: Optional[str] = None,
        config_dir: Optional[str] = None,
    ):
        """Initialize supervisor agent.

        Args:
            llm_endpoint: Optional override for LLM endpoint
            config_dir: Optional directory for config files
        """
        # init base agent
        super().__init__(
            agent_type="supervisor",
            llm_endpoint=llm_endpoint,
            config_dir=config_dir,
        )

        # init sub-agents when needed
        self._sub_agents = {}

    def get_description(self) -> str:
        """Return a description of this agent."""
        return "Supervisor agent that routes customer queries to sub-agents"

    @mlflow.trace(span_type=SpanType.TOOL)
    def tool_route_to_specialized_agent(self, agent_type: str, reason: str) -> str:
        """Route query to appropriate sub-agent.

        Args:
            agent_type: Type of sub-agent to route to
            reason: Reason for routing to this agent

        Returns:
            Response from sub-agent or routing info in test mode
        """
        agent_descriptions = {
            "account": "customer account information, profile details, and account management",
            "billing": "billing inquiries, payment information, and usage details",
            "tech_support": "technical issues, troubleshooting, and device setup assistance",
            "product": "product information, plan comparisons, and promotional offers",
        }

        logger.info(f"Routing to {agent_type} agent: {reason}")

        # TODO: In the full implementation, initialize and call sub-agent
        # for now, return a formatted response about routing
        response = f"ROUTING DECISION: Route query to {agent_type.upper()} AGENT\n\n"
        response += f"REASON: {reason}\n\n"

        if agent_type in agent_descriptions:
            response += f"NOTE: The {agent_type.upper()} AGENT specializes in {agent_descriptions[agent_type]}."

        return response

    def _get_sub_agent(self, agent_type: str) -> BaseAgent:
        """Get or initialize a sub-agent.

        Args:
            agent_type: Type of sub-agent to get

        Returns:
            Initialized sub-agent
        """
        # get cached agent if available
        if agent_type in self._sub_agents:
            return self._sub_agents[agent_type]

        # # import sub-agent classes
        # if agent_type == "account":
        #     from telco_support_agent.agents.account import AccountAgent
        #     agent_class = AccountAgent
        # elif agent_type == "billing":
        #     from telco_support_agent.agents.billing import BillingAgent
        #     agent_class = BillingAgent
        # elif agent_type == "tech_support":
        #     from telco_support_agent.agents.tech_support import TechSupportAgent
        #     agent_class = TechSupportAgent
        # elif agent_type == "product":
        #     from telco_support_agent.agents.product import ProductAgent
        #     agent_class = ProductAgent
        # else:
        #     raise ValueError(f"Unknown agent type: {agent_type}")

        # agent = # TODO

        # cache
        # self._sub_agents[agent_type] = agent

        # return agent

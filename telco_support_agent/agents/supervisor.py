"""Supervisor agent to orchestrate specialized sub-agents."""

from typing import Optional

from telco_support_agent.agents.base_agent import BaseAgent
from telco_support_agent.tools.registry import get_toolkit_for_domain
from telco_support_agent.utils.logging_utils import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)


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
        # get supervisor-specific UC function tools
        try:
            toolkit = get_toolkit_for_domain("supervisor")
            tools = toolkit.tools
            logger.info(
                f"Supervisor agent initialized with {len(tools)} UC function tools"
            )
        except Exception as e:
            logger.error(f"Error initializing UC function tools: {str(e)}")
            tools = []

        super().__init__(
            agent_type="supervisor",
            llm_endpoint=llm_endpoint,
            config_dir=config_dir,
            tools=tools,
        )

        self._sub_agents = {}

    def get_description(self) -> str:
        """Return a description of this agent."""
        return "Supervisor agent that routes customer queries to sub-agents"

    def _get_sub_agent(self, agent_type: str) -> BaseAgent:
        """Get or initialize a sub-agent.

        Args:
            agent_type: Type of sub-agent to get

        Returns:
            Initialized sub-agent
        """
        if agent_type in self._sub_agents:
            return self._sub_agents[agent_type]

        # import and initialize sub-agents
        try:
            if agent_type == "account":
                from telco_support_agent.agents.account import AccountAgent

                agent = AccountAgent(llm_endpoint=self.llm_endpoint)
            # elif agent_type == "billing":
            #     from telco_support_agent.agents.billing import BillingAgent

            #     agent = BillingAgent(llm_endpoint=self.llm_endpoint)
            # elif agent_type == "tech_support":
            #     from telco_support_agent.agents.tech_support import TechSupportAgent

            #     agent = TechSupportAgent(llm_endpoint=self.llm_endpoint)
            # elif agent_type == "product":
            #     from telco_support_agent.agents.product import ProductAgent

            #     agent = ProductAgent(llm_endpoint=self.llm_endpoint)
            else:
                raise ValueError(f"Unknown agent type: {agent_type}")

            self._sub_agents[agent_type] = agent
            return agent

        except Exception as e:
            logger.error(f"Error initializing {agent_type} agent: {str(e)}")
            raise

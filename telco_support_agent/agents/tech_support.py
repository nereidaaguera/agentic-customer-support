"""Tech support agent for handling technical support queries."""

from typing import Optional

from telco_support_agent.agents.base_agent import BaseAgent
from telco_support_agent.tools.tech_support import TechSupportRetriever
from telco_support_agent.utils.logging_utils import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)


class TechSupportAgent(BaseAgent):
    """Tech support agent to handle technical support queries.

    This agent answers technical questions by searching both:
    - Knowledge base articles (official documentation, FAQs, guides)
    - Historical support tickets (similar issues and resolutions)

    The agent combines information from both sources to provide technical support
    responses including troubleshooting steps, known issues, and proven resolution approaches.
    """

    def __init__(
        self,
        llm_endpoint: Optional[str] = None,
        config_dir: Optional[str] = None,
        environment: str = "prod",
    ) -> None:
        """Init agent

        Args:
            llm_endpoint: Optional LLM endpoint override
            config_dir: Optional directory for config files
            environment: Environment to use for retrievers (dev, prod)
        """
        self.retriever = TechSupportRetriever(environment=environment)
        retriever_tools = self.retriever.get_tools()

        logger.info(
            f"Tech support agent initialized with {len(retriever_tools)} retrieval tools"
        )

        super().__init__(
            agent_type="tech_support",
            llm_endpoint=llm_endpoint,
            config_dir=config_dir,
            tools=retriever_tools,
        )

    def get_description(self) -> str:
        """Return a description of this agent."""
        return (
            "Tech support agent that searches knowledge base articles and "
            "historical support tickets to provide technical assistance"
        )

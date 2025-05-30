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
        disable_tools: Optional[list[str]] = None,
    ) -> None:
        """Init agent.

        Args:
            llm_endpoint: Optional LLM endpoint override
            config_dir: Optional directory for config files
            environment: Environment to use for retrievers (dev, prod)
            disable_tools: Optional list of tool names to disable
        """
        self.retriever = TechSupportRetriever(environment=environment)
        retriever_tools = self.retriever.get_tools()

        # mapping of tool names to their executable objects
        vector_search_tools = {
            "knowledge_base_vector_search": self.retriever.kb_retriever.retriever,
            "support_tickets_vector_search": self.retriever.tickets_retriever.retriever,
        }

        logger.info(
            f"Tech support agent initialized with {len(retriever_tools)} retrieval tools"
        )

        super().__init__(
            agent_type="tech_support",
            llm_endpoint=llm_endpoint,
            config_dir=config_dir,
            tools=retriever_tools,  # tool specs for LLM
            vector_search_tools=vector_search_tools,  # executable objects
            disable_tools=disable_tools,
        )

    def get_description(self) -> str:
        """Return a description of this agent."""
        return (
            "Tech support agent that searches knowledge base articles and "
            "historical support tickets to provide technical assistance"
        )

"""Retrieval tools for tech support agent.

Provides vector search retrieval tools for accessing
knowledge base articles and historical support tickets to help answer
technical support queries.
"""

from typing import Optional

from databricks_openai import VectorSearchRetrieverTool

from telco_support_agent.agents.config import get_uc_config
from telco_support_agent.utils.logging_utils import get_logger

logger = get_logger(__name__)


class KnowledgeBaseRetriever:
    """Retriever for knowledge base articles, policies, FAQs, and guides stored in knowledge base index.

    This retriever searches over official documentation including:
    - Troubleshooting guides
    - Policy documents
    - FAQs
    - Setup instructions
    - Service procedures
    """

    def __init__(self, num_results: int = 5, environment: str = "dev"):
        """Init the knowledge base retriever.

        Args:
            num_results: Max number of results to return per search
            environment: Environment to use (dev, prod)
        """
        self.num_results = num_results
        self.environment = environment

        uc_config = get_uc_config(environment)
        catalog = uc_config["catalog"]

        self.index_name = f"{catalog}.agent.knowledge_base_index"

        self.columns = [
            "kb_id",
            "title",
            "category",
            "subcategory",
            "content_type",
            "content",
            "tags",
        ]

        self.retriever = VectorSearchRetrieverTool(
            index_name=self.index_name,
            tool_name="knowledge_base_vector_search",
            tool_description=(
                "Search the knowledge base articles for policies, procedures, FAQs, "
                "troubleshooting guides, and setup instructions. Use this for finding "
                "official company documentation and step-by-step guides for technical issues."
            ),
            num_results=self.num_results,
            columns=self.columns,
        )

        logger.info(f"Initialized KnowledgeBaseRetriever for index: {self.index_name}")

    def search(self, query: str, filters: Optional[dict] = None) -> dict:
        """Search knowledge base for relevant articles.

        Args:
            query: Search query text
            filters: Optional filters to apply (e.g., {"category": "Technical"})

        Returns:
            Search results from the vector index
        """
        try:
            return self.retriever.execute(query=query, filters=filters)
        except Exception as e:
            logger.error(f"Error searching knowledge base: {e}")
            raise


class SupportTicketsRetriever:
    """Retriever for support tickets.

    This retriever searches over past support interactions to find:
    - Similar issues and their solutions
    - Escalation patterns
    - Resolution approaches
    - Customer interaction history
    """

    def __init__(self, num_results: int = 3, environment: str = "dev"):
        """Init support tickets retriever.

        Args:
            num_results: Max number of results to return per search
            environment: Environment to use (dev, prod)
        """
        self.num_results = num_results
        self.environment = environment

        uc_config = get_uc_config(environment)
        catalog = uc_config["catalog"]

        self.index_name = f"{catalog}.agent.support_tickets_index"

        self.columns = [
            "ticket_id",
            "category",
            "priority",
            "status",
            "description",
            "resolution",
            "created_date",
            "resolved_date",
        ]

        self.retriever = VectorSearchRetrieverTool(
            index_name=self.index_name,
            tool_name="support_tickets_vector_search",
            tool_description=(
                "Search historical support tickets for similar issues and their resolutions. "
                "Use this to find how similar technical problems were solved in the past, "
                "including troubleshooting steps, escalation procedures, and successful resolution approaches."
            ),
            num_results=self.num_results,
            columns=self.columns,
        )

        logger.info(f"Initialized SupportTicketsRetriever for index: {self.index_name}")

    def search(self, query: str, filters: Optional[dict] = None) -> dict:
        """Search historical support tickets for similar issues.

        Args:
            query: Search query text describing the technical issue
            filters: Optional filters to apply (e.g., {"priority": "High"})

        Returns:
            Search results from the vector index
        """
        try:
            return self.retriever.execute(query=query, filters=filters)
        except Exception as e:
            logger.error(f"Error searching support tickets: {e}")
            raise


class TechSupportRetriever:
    """Composite class that provides access to both retrieval tools.

    This class combines both knowledge base and support ticket retrievers
    into single interface for tech support agent.
    """

    def __init__(self, environment: str = "dev"):
        """Init both retrievers.

        Args:
            environment: Environment to use (dev, prod)
        """
        self.kb_retriever = KnowledgeBaseRetriever(environment=environment)
        self.tickets_retriever = SupportTicketsRetriever(environment=environment)

        logger.info(
            "Initialized TechSupportRetriever with both knowledge base and tickets"
        )

    def get_tools(self) -> list:
        """Get the retriever tools for use in an agent.

        Returns:
            List of VectorSearchRetrieverTool objects ready for agent use
        """
        return [self.kb_retriever.retriever.tool, self.tickets_retriever.retriever.tool]

    def search_knowledge_base(self, query: str, filters: Optional[dict] = None) -> dict:
        """Search the knowledge base."""
        return self.kb_retriever.search(query, filters)

    def search_tickets(self, query: str, filters: Optional[dict] = None) -> dict:
        """Search historical support tickets."""
        return self.tickets_retriever.search(query, filters)

    def search(self, query: str) -> dict:
        """Search both knowledge base and support tickets.

        Args:
            query: Search query text

        Returns:
            Dictionary with results from both sources
        """
        try:
            kb_results = self.search_knowledge_base(query)
            tickets_results = self.search_tickets(query)

            return {"knowledge_base": kb_results, "support_tickets": tickets_results}
        except Exception as e:
            logger.error(f"Error in combined search: {e}")
            raise

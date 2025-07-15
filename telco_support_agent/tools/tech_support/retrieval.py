"""Retrieval tools for tech support agent.

Provides vector search retrieval tools for accessing
knowledge base articles and historical support tickets to help answer
technical support queries.
"""

import asyncio
import time
from typing import Any, Optional

import mlflow
from databricks_openai import VectorSearchRetrieverTool
from mlflow.entities import SpanType

from telco_support_agent.config import UCConfig
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

    def __init__(self, num_results: int = 5, uc_config: Optional[UCConfig] = None):
        """Init knowledge base retriever.

        Args:
            num_results: Max number of results to return per search
            uc_config: Unity Catalog configuration
        """
        self.num_results = num_results

        if not uc_config:
            # Default UC config if not provided
            uc_config = UCConfig(
                agent_catalog="telco_customer_support_prod",
                agent_schema="agent",
                data_schema="gold",
                model_name="telco_customer_support_agent",
            )

        self.index_name = uc_config.get_uc_index_name("knowledge_base_index")

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

    def search(
        self, query: str, filters: Optional[dict[str, Any]] = None
    ) -> list[dict[str, Any]]:
        """Search knowledge base for relevant articles.

        Args:
            query: Search query text
            filters: Optional filters to apply (e.g., {"category": "Technical"})

        Returns:
            List of documents
        """
        with mlflow.start_span(
            name="knowledge_base_retriever", span_type=SpanType.RETRIEVER
        ) as span:
            span.set_attributes(
                {
                    "index_name": self.index_name,
                    "num_results": self.num_results,
                    "has_filters": filters is not None,
                    "retriever_type": "knowledge_base",
                }
            )
            span.set_inputs({"query": query, "filters": filters})

            try:
                start_time = time.time()
                results = self.retriever.execute(query=query, filters=filters)
                search_time = time.time() - start_time

                results_count = len(results) if isinstance(results, list) else 0

                span.set_outputs(
                    {
                        "results": results,
                        "search_duration_ms": round(search_time * 1000, 2),
                        "results_count": results_count,
                    }
                )

                logger.info(
                    f"Knowledge base search completed in {search_time:.3f}s, found {results_count} results"
                )
                return results

            except Exception as e:
                logger.error(f"Error searching knowledge base: {e}")
                raise

    async def search_async(
        self, query: str, filters: Optional[dict[str, Any]] = None
    ) -> list[dict[str, Any]]:
        """Async search knowledge base.

        Args:
            query: Search query text
            filters: Optional filters to apply

        Returns:
            List of documents
        """
        with mlflow.start_span(
            name="knowledge_base_retriever", span_type=SpanType.RETRIEVER
        ) as span:
            span.set_attributes(
                {
                    "index_name": self.index_name,
                    "num_results": self.num_results,
                    "has_filters": filters is not None,
                    "retriever_type": "knowledge_base",
                    "execution_mode": "async",
                }
            )
            span.set_inputs({"query": query, "filters": filters})

            try:
                start_time = time.time()
                results = await asyncio.to_thread(
                    self.retriever.execute, query=query, filters=filters
                )
                search_time = time.time() - start_time

                results_count = len(results) if isinstance(results, list) else 0

                span.set_outputs(
                    {
                        "results": results,
                        "search_duration_ms": round(search_time * 1000, 2),
                        "results_count": results_count,
                    }
                )

                logger.info(
                    f"Knowledge base async search completed in {search_time:.3f}s, found {results_count} results"
                )
                return results

            except Exception as e:
                logger.error(f"Error in async knowledge base search: {e}")
                raise


class SupportTicketsRetriever:
    """Retriever for historical support tickets.

    This retriever searches over past support interactions to find:
    - Similar issues and their solutions
    - Escalation patterns
    - Resolution approaches
    - Customer interaction history
    """

    def __init__(self, num_results: int = 3, uc_config: Optional[UCConfig] = None):
        """Init support tickets retriever.

        Args:
            num_results: Max number of results to return per search
            uc_config: Unity Catalog configuration
        """
        self.num_results = num_results

        if not uc_config:
            # Default UC config if not provided
            uc_config = UCConfig(
                agent_catalog="telco_customer_support_prod",
                agent_schema="agent",
                data_schema="gold",
                model_name="telco_customer_support_agent",
            )

        self.index_name = uc_config.get_uc_index_name("support_tickets_index")

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

    def search(
        self, query: str, filters: Optional[dict[str, Any]] = None
    ) -> list[dict[str, Any]]:
        """Search historical support tickets for similar issues.

        Args:
            query: Search query text describing the technical issue
            filters: Optional filters to apply (e.g., {"priority": "High"})

        Returns:
            List of documents
        """
        with mlflow.start_span(
            name="support_tickets_retriever", span_type=SpanType.RETRIEVER
        ) as span:
            span.set_attributes(
                {
                    "index_name": self.index_name,
                    "num_results": self.num_results,
                    "has_filters": filters is not None,
                    "retriever_type": "support_tickets",
                }
            )

            span.set_inputs({"query": query, "filters": filters})

            try:
                start_time = time.time()
                results = self.retriever.execute(query=query, filters=filters)
                search_time = time.time() - start_time

                results_count = len(results) if isinstance(results, list) else 0

                span.set_outputs(
                    {
                        "results": results,
                        "search_duration_ms": round(search_time * 1000, 2),
                        "results_count": results_count,
                    }
                )

                logger.info(
                    f"Support tickets search completed in {search_time:.3f}s, found {results_count} results"
                )
                return results

            except Exception as e:
                logger.error(f"Error searching support tickets: {e}")
                raise

    async def search_async(
        self, query: str, filters: Optional[dict[str, Any]] = None
    ) -> list[dict[str, Any]]:
        """Async search historical support tickets for similar issues.

        Args:
            query: Search query text describing the technical issue
            filters: Optional filters to apply

        Returns:
            List of documents from the vector index
        """
        with mlflow.start_span(
            name="support_tickets_retriever", span_type=SpanType.RETRIEVER
        ) as span:
            span.set_attributes(
                {
                    "index_name": self.index_name,
                    "num_results": self.num_results,
                    "has_filters": filters is not None,
                    "retriever_type": "support_tickets",
                    "execution_mode": "async",
                }
            )
            span.set_inputs({"query": query, "filters": filters})

            try:
                start_time = time.time()
                results = await asyncio.to_thread(
                    self.retriever.execute, query=query, filters=filters
                )
                search_time = time.time() - start_time

                results_count = len(results) if isinstance(results, list) else 0

                span.set_outputs(
                    {
                        "results": results,
                        "search_duration_ms": round(search_time * 1000, 2),
                        "results_count": results_count,
                    }
                )

                logger.info(
                    f"Support tickets async search completed in {search_time:.3f}s, found {results_count} results"
                )
                return results

            except Exception as e:
                logger.error(f"Error in async support tickets search: {e}")
                raise


class TechSupportRetriever:
    """Composite retriever that provides access to both knowledge base and support tickets.

    This class combines both knowledge base and support ticket retrievers
    into a single interface for tech support agent with async parallel search capabilities.
    """

    def __init__(self, uc_config: Optional[UCConfig] = None):
        """Init both retrievers.

        Args:
            uc_config: Unity Catalog configuration
        """
        self.kb_retriever = KnowledgeBaseRetriever(uc_config=uc_config)
        self.tickets_retriever = SupportTicketsRetriever(uc_config=uc_config)

        logger.info(
            "Initialized TechSupportRetriever with both knowledge base and support tickets"
        )

    def get_tools(self) -> list[dict[str, Any]]:
        """Get retriever tools.

        Returns:
            List of VectorSearchRetrieverTool objects ready for agent use
        """
        return [self.kb_retriever.retriever.tool, self.tickets_retriever.retriever.tool]

    def search_knowledge_base(
        self, query: str, filters: Optional[dict[str, Any]] = None
    ) -> list[dict[str, Any]]:
        """Search the knowledge base only.

        Args:
            query: Search query text
            filters: Optional filters to apply

        Returns:
            Knowledge base search results as list of documents
        """
        return self.kb_retriever.search(query, filters)

    def search_tickets(
        self, query: str, filters: Optional[dict[str, Any]] = None
    ) -> list[dict[str, Any]]:
        """Search historical support tickets only.

        Args:
            query: Search query text
            filters: Optional filters to apply

        Returns:
            Support tickets search results as list of documents
        """
        return self.tickets_retriever.search(query, filters)

    async def async_search(
        self,
        query: str,
        kb_filters: Optional[dict[str, Any]] = None,
        tickets_filters: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Search both knowledge base and support tickets in parallel using asyncio.

        This method executes searches against both indexes concurrently using asyncio
        for optimal performance and simpler context management.

        Args:
            query: Search query text
            kb_filters: Optional filters for knowledge base search
            tickets_filters: Optional filters for support tickets search

        Returns:
            Dictionary with results from both sources plus metadata
        """
        with mlflow.start_span(
            name="tech_support_retriever", span_type=SpanType.RETRIEVER
        ) as span:
            span.set_inputs(
                {
                    "query": query,
                    "kb_filters": kb_filters,
                    "tickets_filters": tickets_filters,
                }
            )

            try:
                start_time = time.time()

                # execute searches in parallel
                kb_results, tickets_results = await asyncio.gather(
                    self.kb_retriever.search_async(query, kb_filters),
                    self.tickets_retriever.search_async(query, tickets_filters),
                    return_exceptions=True,  # don't fail if one search fails
                )

                total_duration = time.time() - start_time

                results = {}
                if isinstance(kb_results, Exception):
                    logger.error(f"Knowledge base search failed: {kb_results}")
                    results["knowledge_base"] = {"error": str(kb_results)}
                else:
                    results["knowledge_base"] = kb_results

                if isinstance(tickets_results, Exception):
                    logger.error(f"Support tickets search failed: {tickets_results}")
                    results["support_tickets"] = {"error": str(tickets_results)}
                else:
                    results["support_tickets"] = tickets_results

                results["metadata"] = {
                    "query": query,
                    "total_duration_ms": round(total_duration * 1000, 2),
                    "timestamp": time.time(),
                }

                kb_success = "error" not in results.get("knowledge_base", {})
                tickets_success = "error" not in results.get("support_tickets", {})

                total_results = 0
                if kb_success and isinstance(results.get("knowledge_base"), list):
                    total_results += len(results["knowledge_base"])
                if tickets_success and isinstance(results.get("support_tickets"), list):
                    total_results += len(results["support_tickets"])

                span.set_outputs(
                    {
                        "results": results,
                        "total_duration_ms": round(total_duration * 1000, 2),
                        "total_results": total_results,
                    }
                )

                logger.info(
                    f"Async parallel search completed in {total_duration:.3f}s, "
                    f"found {total_results} total results"
                )

                return results

            except Exception as e:
                logger.error(f"Error in async parallel search: {e}")
                raise

"""Supervisor agent to orchestrate specialized sub-agents."""

from collections.abc import Generator
from typing import Optional
from uuid import uuid4

import mlflow
from mlflow.entities import SpanType
from mlflow.types.responses import (
    ResponsesRequest,
    ResponsesResponse,
    ResponsesStreamEvent,
)

from telco_support_agent.agents.base_agent import BaseAgent
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
        # NOTE: don't need UC function tools for supervisor
        # the routing logic will be implemented directly in this class
        super().__init__(
            agent_type="supervisor",
            llm_endpoint=llm_endpoint,
            config_dir=config_dir,
            tools=[],  # no tools needed for routing
        )

        self._sub_agents = {}

        logger.info("Supervisor agent initialized")

    def get_description(self) -> str:
        """Return a description of this agent."""
        return "Supervisor agent that routes customer queries to specialized sub-agents"

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
            logger.info(f"Initialized {agent_type} agent")
            return agent

        except Exception as e:
            logger.error(f"Error initializing {agent_type} agent: {str(e)}")
            raise

    @mlflow.trace(span_type=SpanType.AGENT)
    def route_query(self, query: str) -> str:
        """Determine which sub-agent should handle the query.

        Args:
            query: User query to classify

        Returns:
            The agent type that should handle this query
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": query},
        ]

        try:
            response = self.call_llm(messages)
            agent_type = response.get("content", "").strip().lower()

            valid_agents = ["account", "billing", "tech_support", "product"]
            if agent_type not in valid_agents:
                logger.warning(
                    f"LLM returned invalid agent type: {agent_type}. Falling back to account agent."
                )
                agent_type = "account"

            logger.info(f"Routing query to {agent_type} agent")
            return agent_type
        except Exception as e:
            logger.error(
                f"Error in routing query: {str(e)}. Falling back to account agent."
            )
            return "account"

    @mlflow.trace(span_type=SpanType.AGENT)
    def predict(self, model_input: ResponsesRequest) -> ResponsesResponse:
        """Process the user query and route to appropriate sub-agent.

        Args:
            model_input: The request containing user messages

        Returns:
            The response from the appropriate sub-agent
        """
        # extract the user query from the input
        user_messages = [msg for msg in model_input.input if msg.role == "user"]
        if not user_messages:
            # no user messages found, return an error response
            return ResponsesResponse(
                output=[
                    {
                        "role": "assistant",
                        "type": "message",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "No user query found in the request.",
                            }
                        ],
                    }
                ]
            )

        # use last user message as the query
        query = user_messages[-1].content

        # determine which agent should handle query
        agent_type = self.route_query(query)

        # add routing decision to custom outputs
        custom_outputs = (
            model_input.custom_inputs.copy() if model_input.custom_inputs else {}
        )
        custom_outputs["routing"] = {
            "agent_type": agent_type,
            "decision_time": mlflow.get_run(
                mlflow.active_run().info.run_id
            ).info.start_time
            if mlflow.active_run()
            else None,
        }

        # get sub-agent
        sub_agent = self._get_sub_agent(agent_type)

        # let sub-agent handle query
        sub_response = sub_agent.predict(model_input)

        # combine custom outputs
        if sub_response.custom_outputs:
            custom_outputs.update(sub_response.custom_outputs)

        # return sub-agent's response with custom outputs
        return ResponsesResponse(
            output=sub_response.output, custom_outputs=custom_outputs
        )

    @mlflow.trace(span_type=SpanType.AGENT)
    def predict_stream(
        self, model_input: ResponsesRequest
    ) -> Generator[ResponsesStreamEvent, None, None]:
        """Stream the response from the appropriate sub-agent.

        Args:
            model_input: The request containing user messages

        Yields:
            ResponsesStreamEvent objects from the sub-agent
        """
        # extract the user query from the input
        user_messages = [msg for msg in model_input.input if msg.role == "user"]
        if not user_messages:
            # no user messages found, return an error response
            yield ResponsesStreamEvent(
                type="response.output_item.done",
                item={
                    "id": str(uuid4()),
                    "content": [
                        {
                            "type": "output_text",
                            "text": "No user query found in the request.",
                        }
                    ],
                    "role": "assistant",
                    "type": "message",
                },
            )
            return

        # use the last user message as query
        query = user_messages[-1].content

        # determine which agent should handle this query
        agent_type = self.route_query(query)

        # emit debug event (visible in traces but not UI)
        yield ResponsesStreamEvent(
            type="response.debug",
            item={
                "id": str(uuid4()),
                "routing_decision": f"Query routed to {agent_type} agent",
            },
        )

        # get sub-agent
        try:
            sub_agent = self._get_sub_agent(agent_type)
            yield from sub_agent.predict_stream(model_input)

        except Exception as e:
            logger.error(f"Error processing with {agent_type} agent: {str(e)}")
            yield ResponsesStreamEvent(
                type="response.output_item.done",
                item={
                    "id": str(uuid4()),
                    "content": [
                        {
                            "type": "output_text",
                            "text": f"Error processing your request: {str(e)}",
                        }
                    ],
                    "role": "assistant",
                    "type": "message",
                },
            )

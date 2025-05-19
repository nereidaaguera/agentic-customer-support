"""Supervisor agent to orchestrate specialized sub-agents."""

from collections.abc import Generator
from typing import Optional, Union
from uuid import uuid4

import mlflow
from mlflow.entities import SpanType
from mlflow.models import set_model
from mlflow.types.responses import (
    ResponsesRequest,
    ResponsesResponse,
    ResponsesStreamEvent,
)

from telco_support_agent.agents.account import AccountAgent
from telco_support_agent.agents.base_agent import BaseAgent
from telco_support_agent.agents.types import AgentType
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

    def _get_sub_agent(self, agent_type: Union[AgentType, str]) -> Optional[BaseAgent]:
        """Get or initialize a sub-agent if implemented.

        Args:
            agent_type: Type of sub-agent to get

        Returns:
            Initialized sub-agent or None if not implemented
        """
        agent_type_str = (
            agent_type.value if isinstance(agent_type, AgentType) else agent_type
        )

        if agent_type_str in self._sub_agents:
            return self._sub_agents[agent_type_str]

        agent_type_enum = (
            agent_type
            if isinstance(agent_type, AgentType)
            else AgentType.from_string(agent_type)
        )

        if agent_type_enum == AgentType.ACCOUNT:
            try:
                agent = AccountAgent(llm_endpoint=self.llm_endpoint)
                self._sub_agents[agent_type_str] = agent
                logger.info(f"Initialized {agent_type_str} agent")
                return agent
            except Exception as e:
                logger.error(f"Error initializing {agent_type_str} agent: {str(e)}")
                raise

        # TODO: other agent types' not implemented
        logger.warning(f"{agent_type_str.capitalize()} agent not implemented yet.")
        return None

    @mlflow.trace(span_type=SpanType.AGENT)
    def route_query(self, query: str) -> AgentType:
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
            agent_type_str = response.get("content", "").strip().lower()

            try:
                agent_type = AgentType.from_string(agent_type_str)
                logger.info(f"Routing query to {agent_type.value} agent")
                return agent_type
            except ValueError:
                logger.warning(
                    f"LLM returned invalid agent type: {agent_type_str}. Falling back to account agent."
                )
                return AgentType.ACCOUNT

        except Exception as e:
            logger.error(
                f"Error in routing query: {str(e)}. Falling back to account agent."
            )
            return AgentType.ACCOUNT

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
            "agent_type": agent_type.value,
            "decision_time": mlflow.get_run(
                mlflow.active_run().info.run_id
            ).info.start_time
            if mlflow.active_run()
            else None,
        }

        # get sub-agent
        sub_agent = self._get_sub_agent(agent_type)

        # if sub-agent not implemented generate non-response
        if sub_agent is None:
            non_response = self.generate_non_response(agent_type, query)
            return ResponsesResponse(
                output=[non_response], custom_outputs=custom_outputs
            )

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
                "routing_decision": f"Query routed to {agent_type.value} agent",
            },
        )

        # get sub-agent if implemented
        sub_agent = self._get_sub_agent(agent_type)

        # if sub-agent not implemented generate non-response
        if sub_agent is None:
            non_response = self.generate_non_response(agent_type, query)
            yield ResponsesStreamEvent(
                type="response.output_item.done", item=non_response
            )
            return

        try:
            # stream response from sub-agent
            yield from sub_agent.predict_stream(model_input)

        except Exception as e:
            logger.error(f"Error processing with {agent_type.value} agent: {str(e)}")
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

    def generate_non_response(
        self, agent_type: Union[AgentType, str], query: str
    ) -> dict:
        """Generate a graceful response when an agent type is not implemented.

        Args:
            agent_type: The agent type that isn't implemented
            query: The original user query

        Returns:
            A response item dictionary
        """
        agent_type_str = (
            agent_type.value if isinstance(agent_type, AgentType) else agent_type
        )

        return {
            "role": "assistant",
            "type": "message",
            "content": [
                {
                    "type": "output_text",
                    "text": f"I apologize, but our {agent_type_str} support system is currently being upgraded and isn't available yet. "
                    f"We expect this feature to be available in the next few weeks. "
                    f"In the meantime, I can help with account information, profiles, and subscription details. "
                    f"Would you like me to help you with any account-related questions?",
                }
            ],
            "id": str(uuid4()),
        }


# Create an instance and set the model
# This will be executed when the module is imported by MLflow
agent = SupervisorAgent()
set_model(agent)

"""Supervisor agent to orchestrate specialized sub-agents."""

from collections.abc import Generator
from typing import NamedTuple, Optional, Union
from uuid import uuid4

import mlflow
from mlflow.entities import SpanType
from mlflow.models import set_model
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
)

from telco_support_agent.agents.account import AccountAgent
from telco_support_agent.agents.base_agent import BaseAgent
from telco_support_agent.agents.billing import BillingAgent
from telco_support_agent.agents.product import ProductAgent
from telco_support_agent.agents.tech_support import TechSupportAgent
from telco_support_agent.agents.types import AgentType
from telco_support_agent.utils.logging_utils import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)


class AgentExecutionResult(NamedTuple):
    """Result of agent execution preparation."""

    sub_agent: Optional[BaseAgent]
    agent_type: AgentType
    query: str
    custom_outputs: dict
    error_response: Optional[dict] = None


class SupervisorAgent(BaseAgent):
    """Supervisor agent to orchestrate specialized sub-agents.

    This agent analyzes customer queries and routes them to the appropriate
    sub-agent based on query content and intent.
    """

    def __init__(
        self,
        llm_endpoint: Optional[str] = None,
        config_dir: Optional[str] = None,
        disable_tools: Optional[list[str]] = None,
    ):
        """Initialize supervisor agent.

        Args:
            llm_endpoint: Optional override for LLM endpoint
            config_dir: Optional directory for config files
            disable_tools: Optional list of tool names to disable.
                Can be either simple names (e.g., 'get_usage_info') or full UC function
                names (e.g., 'telco_customer_support_dev.agent.get_usage_info').
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
        self.disable_tools = disable_tools or []

        if self.disable_tools:
            logger.info(
                f"Supervisor configured with disabled tools: {self.disable_tools}"
            )

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

        agents_classes = {
            AgentType.ACCOUNT: AccountAgent,
            AgentType.BILLING: BillingAgent,
            AgentType.TECH_SUPPORT: TechSupportAgent,
            AgentType.PRODUCT: ProductAgent,
        }

        if agent_type_enum in agents_classes:
            try:
                agent = agents_classes[agent_type_enum](
                    llm_endpoint=self.llm_endpoint, disable_tools=self.disable_tools
                )
                self._sub_agents[agent_type_str] = agent
                logger.info(f"Initialized {agent_type_str} agent")
                return agent
            except Exception as e:
                logger.error(f"Error initializing {agent_type_str} agent: {str(e)}")
                raise
        else:
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

    def _prepare_agent_execution(
        self, request: ResponsesAgentRequest
    ) -> AgentExecutionResult:
        """Prepare for agent execution by handling routing / validation.

        Consolidates common logic between predict and predict_stream.

        Args:
            request: The request containing user messages

        Returns:
            AgentExecutionResult containing all necessary information for execution
        """
        # extract the user query from the input
        user_messages = [msg for msg in request.input if msg.role == "user"]
        if not user_messages:
            # no user messages found, return error response
            error_response = {
                "role": "assistant",
                "type": "message",
                "content": [
                    {
                        "type": "output_text",
                        "text": "No user query found in the request.",
                    }
                ],
                "id": str(uuid4()),
            }
            return AgentExecutionResult(
                sub_agent=None,
                agent_type=AgentType.ACCOUNT,  # placeholder
                query="",
                custom_outputs={},
                error_response=error_response,
            )

        # use last user message as the query
        query = user_messages[-1].content

        # determine which agent should handle query
        agent_type = self.route_query(query)

        # prepare custom outputs with routing decision
        custom_outputs = request.custom_inputs.copy() if request.custom_inputs else {}
        custom_outputs["routing"] = {
            "agent_type": agent_type.value,
        }

        # add disabled tools info to custom outputs
        if self.disable_tools:
            custom_outputs["routing"]["disable_tools"] = self.disable_tools

        # get sub-agent
        sub_agent = self._get_sub_agent(agent_type)

        # if sub-agent not implemented, prepare non-response
        if sub_agent is None:
            error_response = self.generate_non_response(agent_type, query)
            return AgentExecutionResult(
                sub_agent=None,
                agent_type=agent_type,
                query=query,
                custom_outputs=custom_outputs,
                error_response=error_response,
            )

        return AgentExecutionResult(
            sub_agent=sub_agent,
            agent_type=agent_type,
            query=query,
            custom_outputs=custom_outputs,
            error_response=None,
        )

    @mlflow.trace(span_type=SpanType.AGENT, name="supervisor")
    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        """Process user query, route to, and yield response from sub-agent.

        Args:
            request: The request containing user messages

        Returns:
            The response from the sub-agent
        """
        execution_result = self._prepare_agent_execution(request)

        if execution_result.error_response:
            return ResponsesAgentResponse(
                output=[execution_result.error_response],
                custom_outputs=execution_result.custom_outputs,
            )

        with mlflow.start_span(
            span_type=SpanType.AGENT, name=f"{execution_result.agent_type.value}_agent"
        ) as span:
            span.set_attributes(
                {
                    "agent_type": execution_result.agent_type.value,
                    "query": execution_result.query,
                    "customer_id": request.custom_inputs.get("customer")
                    if request.custom_inputs
                    else None,
                    "disable_tools": self.disable_tools,
                }
            )
            span.set_inputs(
                {
                    "request": request.model_dump(),
                    "custom_inputs": request.custom_inputs,
                }
            )

            sub_response = execution_result.sub_agent.predict(request)

            # combine custom outputs
            final_custom_outputs = execution_result.custom_outputs.copy()
            if sub_response.custom_outputs:
                final_custom_outputs.update(sub_response.custom_outputs)

            span.set_outputs(
                {
                    "response": sub_response.output,
                    "custom_outputs": final_custom_outputs,
                }
            )

        return ResponsesAgentResponse(
            output=sub_response.output, custom_outputs=final_custom_outputs
        )

    @mlflow.trace(span_type=SpanType.AGENT, name="supervisor")
    def predict_stream(
        self, request: ResponsesAgentRequest
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        """Stream response from the selected sub-agent.

        Args:
            request: request containing user messages

        Yields:
            ResponsesAgentStreamEvent objects from the sub-agent
        """
        execution_result = self._prepare_agent_execution(request)

        if execution_result.error_response:
            yield ResponsesAgentStreamEvent(
                type="response.output_item.done", item=execution_result.error_response
            )
            return

        yield ResponsesAgentStreamEvent(
            type="response.debug",
            item={
                "id": str(uuid4()),
                "routing_decision": f"Query routed to {execution_result.agent_type.value} agent",
            },
        )

        try:
            with mlflow.start_span(
                name=f"{execution_result.agent_type.value}_agent"
            ) as span:
                span.set_attributes(
                    {
                        "agent_type": execution_result.agent_type.value,
                        "query": execution_result.query,
                        "streaming": True,
                        "disable_tools": self.disable_tools,
                    }
                )
                span.set_inputs(
                    {
                        "request": request.model_dump(),
                        "customer_id": request.custom_inputs.get("customer")
                        if request.custom_inputs
                        else None,
                    }
                )

                response_count = 0
                for event in execution_result.sub_agent.predict_stream(request):
                    response_count += 1
                    yield event

                span.set_outputs({"events_streamed": response_count})

        except Exception as e:
            logger.error(
                f"Error processing with {execution_result.agent_type.value} agent: {str(e)}"
            )
            yield ResponsesAgentStreamEvent(
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
                    f"In the meantime, I can help with account information, profiles, and technical support. "
                    f"Would you like me to help you with any other questions?",
                }
            ],
            "id": str(uuid4()),
        }


# Create an instance and set the model
# This will be executed when the module is imported by MLflow
agent = SupervisorAgent()
set_model(agent)

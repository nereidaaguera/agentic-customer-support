"""Supervisor agent to orchestrate specialized sub-agents."""

import mlflow
from mlflow.entities import SpanType

from telco_support_agent.agents.base_agent import BaseAgent
from telco_support_agent.tools import ToolInfo


class SupervisorAgent(BaseAgent):
    """Supervisor agent to orchestrate specialized sub-agents.

    This agent analyzes customer queries and routes them to the appropriate
    specialized agent based on query content and intent.
    """

    def __init__(self, llm_endpoint: str):
        """Init supervisor agent.

        Args:
            llm_endpoint: Name of the LLM endpoint to use
        """
        system_prompt = """You are an intelligent supervisor for a telecom customer support system. Your job is to analyze customer queries and route them to the appropriate specialized agent.

When a customer submits a query, you must:
1. Carefully analyze the query to understand its intent, topic, and required expertise
2. Determine which specialized agent would be best suited to handle this query
3. Use the route_to_specialized_agent tool to route the query to that agent
4. Provide a clear reason for your routing decision

You have the following specialized agents available:

1. ACCOUNT AGENT: Handles queries related to customer profiles, account status, subscription details, and account management.
   Examples: "What plan am I on?", "When did I create my account?", "Is my autopay enabled?", "How many lines do I have?"

2. BILLING AGENT: Handles queries related to bills, payments, charges, billing cycles, and usage.
   Examples: "Why is my bill higher this month?", "When is my payment due?", "I see a charge I don't recognize", "How much data did I use?"

3. TECH_SUPPORT AGENT: Handles queries related to troubleshooting, connectivity issues, device setup, and technical problems.
   Examples: "My phone won't connect", "I can't make calls", "How do I reset my voicemail?", "Why is my internet slow?"

4. PRODUCT AGENT: Handles queries related to service plans, devices, promotions, and plan comparisons.
   Examples: "What's the difference between plans?", "Do you have promotions?", "Is my phone 5G compatible?", "Which plan has the most data?"

Route each query to the most appropriate agent based on its primary topic. If a query spans multiple domains, route it to the agent that would handle the most central aspect of the query.

Always use the route_to_specialized_agent tool to route queries, providing a clear and specific reason for your choice.
"""

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

        # self.account_agent = None  # Placeholder for account agent
        # self.billing_agent = None  # Placeholder for billing agent
        # self.tech_support_agent = None  # Placeholder for tech support agent
        # self.product_agent = None  # Placeholder for product agent

    @mlflow.trace(span_type=SpanType.TOOL)
    def route_to_specialized_agent(self, agent_type: str, reason: str) -> str:
        """Route query to specialized agent.

        TODO: for now just return information about where the query would be routed
        need to call the appropriate sub-agent

        Args:
            agent_type: Sub-agent to route to
            reason: Reason for routing to sub-agent

        Returns:
            Response describing the routing decision
        """
        # TODO: testing: just return routing info
        response = f"ROUTING DECISION: Route query to {agent_type.upper()} AGENT\n\n"
        response += f"REASON: {reason}\n\n"

        return response

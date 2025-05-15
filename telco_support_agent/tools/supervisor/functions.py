"""UC functions for supervisor agent."""

import json

from telco_support_agent.tools.registry import uc_function
from telco_support_agent.utils.logging_utils import get_logger

logger = get_logger(__name__)


@uc_function(domain="supervisor")
def route_to_specialized_agent(agent_type: str, reason: str) -> str:
    """Routes a query to the appropriate specialized agent based on query content and intent.

    Args:
        agent_type (str): Type of sub-agent to route to (account, billing, tech_support, product)
        reason (str): Reason for routing to this agent

    Returns:
        str: Routing decision and information about the selected agent
    """
    agent_descriptions = {
        "account": "customer account information, profile details, and account management",
        "billing": "billing inquiries, payment information, and usage details",
        "tech_support": "technical issues, troubleshooting, and device setup assistance",
        "product": "product information, plan comparisons, and promotional offers",
    }

    logger.info(f"Routing to {agent_type} agent: {reason}")

    response = {
        "decision": f"Route query to {agent_type.upper()} AGENT",
        "reason": reason,
    }

    if agent_type in agent_descriptions:
        response["agent_description"] = (
            f"The {agent_type.upper()} AGENT specializes in {agent_descriptions[agent_type]}."
        )

    return json.dumps(response, indent=2)

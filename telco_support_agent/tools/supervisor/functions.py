"""UC functions for supervisor agent."""

import json

from unitycatalog.ai.core.databricks import DatabricksFunctionClient

client = DatabricksFunctionClient()


def register_routing_function():
    """Register the routing function with Unity Catalog."""
    try:

        def route_to_specialized_agent(agent_type: str, reason: str) -> str:
            """Routes a query to the appropriate specialized agent.

            Args:
                agent_type: Type of sub-agent to route to (account, billing, tech_support, product)
                reason: Reason for routing to this agent

            Returns:
                JSON string with routing decision information
            """
            agent_descriptions = {
                "account": "customer account information, profile details, and account management",
                "billing": "billing inquiries, payment information, and usage details",
                "tech_support": "technical issues, troubleshooting, and device setup assistance",
                "product": "product information, plan comparisons, and promotional offers",
            }

            response = {
                "decision": f"Route query to {agent_type.upper()} AGENT",
                "reason": reason,
            }

            if agent_type in agent_descriptions:
                response["agent_description"] = (
                    f"The {agent_type.upper()} AGENT specializes in {agent_descriptions[agent_type]}."
                )

            return json.dumps(response, indent=2)

        # register function
        client.create_python_function(
            func=route_to_specialized_agent,
            catalog="telco_customer_support_dev",
            schema="agent",
            replace=True,
        )
        print("Registered route_to_specialized_agent UC function")

    except Exception as e:
        print(f"Error registering route_to_specialized_agent: {str(e)}")


# call registration function
register_routing_function()

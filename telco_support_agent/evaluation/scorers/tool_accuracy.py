"""Tool accuracy evaluation scorers for telco support agent."""

from typing import Any, Optional

from databricks.agents.evals import metric
from mlflow.entities import Assessment, Feedback
from mlflow.genai.scorers import scorer
from mlflow.genai.judges import meets_guidelines

from ..utils import (
    extract_request_text,
    extract_response_text,
    extract_trace_routing_info,
)


def _get_tool_accuracy_guidelines() -> list[str]:
    """Guidelines for evaluating tool usage accuracy in telco support.

    Returns:
        List of guidelines for tool usage evaluation
    """
    return [
        "If the query requires customer-specific information (account details, profile, status), the 'get_customer_info' tool should be called with the correct customer ID",
        "If the query is about billing, payments, or charges, the 'get_billing_info' tool should be called with appropriate date ranges and filters",
        "If the query asks about available plans, upgrades, or service options, the 'get_plans_info' tool should be called",
        "If the query is about technical issues or troubleshooting, technical support retrieval tools should be used to find relevant documentation",
        "Tools should be called with correct and complete parameters - missing required parameters or incorrect formats indicate poor tool usage",
        "Tools should not be called unnecessarily when the query can be answered without customer-specific data",
        "If customer-specific information is needed but no tools were called, this indicates missing tool usage",
    ]


@metric
def tool_accuracy_metric(
    *, request: str, response: str, tools_used: Optional[list] = None, **kwargs
) -> Assessment:
    """Evaluate tool usage accuracy.

    Args:
        request: The customer's original query
        response: The agent's response
        tools_used: List of tools that were called
        **kwargs: Additional parameters (ignored)

    Returns:
        Assessment object with binary score and rationale
    """
    try:
        request_text = extract_request_text(request)
        response_text = extract_response_text(response)

        # format tools used for evaluation
        tools_list = tools_used or []
        tools_str = ", ".join(tools_list) if tools_list else "none"

        # guidelines judge to evaluate tool usage
        feedback = meets_guidelines(
            name="tool_accuracy",
            guidelines=_get_tool_accuracy_guidelines(),
            context={
                "request": request_text,
                "response": response_text,
                "tools_used": tools_str,
            },
        )

        # convert "yes"/"no" to numeric score for metric compatibility
        score = 1.0 if feedback.value == "yes" else 0.0

        return Assessment(value=score, rationale=feedback.rationale)

    except Exception as e:
        return Assessment(
            value=0.0, rationale=f"Error evaluating tool accuracy: {str(e)}"
        )


@scorer
def tool_accuracy_scorer(
    *, request: str, response: str, trace: Optional[dict[str, Any]] = None, **kwargs
) -> Feedback:
    """Evaluate tool usage accuracy.

    Args:
        request: The customer's original query
        response: The agent's response
        trace: Trace information containing tool usage details
        **kwargs: Additional parameters (ignored)

    Returns:
        Feedback object with binary score and rationale
    """
    try:
        request_text = extract_request_text(request)
        response_text = extract_response_text(response)

        # get tool information from trace
        routing_info = extract_trace_routing_info(trace)
        tools_used = routing_info.get("tools_used", [])
        tools_str = ", ".join(tools_used) if tools_used else "none"

        # context with trace info
        context = {
            "request": request_text,
            "response": response_text,
            "tools_used": tools_str,
        }

        # guidelines judge to evaluate tool usage
        feedback = meets_guidelines(
            name="tool_accuracy",
            guidelines=_get_tool_accuracy_guidelines(),
            context=context,
        )

        # enhance rationale with tool information
        enhanced_rationale = feedback.rationale
        if tools_used:
            enhanced_rationale += f" (Tools used: {tools_str})"
        else:
            enhanced_rationale += " (No tools used)"

        return Feedback(
            value=feedback.value,
            rationale=enhanced_rationale,
        )

    except Exception as e:
        return Feedback(
            value="no", rationale=f"Error evaluating tool accuracy: {str(e)}"
        )

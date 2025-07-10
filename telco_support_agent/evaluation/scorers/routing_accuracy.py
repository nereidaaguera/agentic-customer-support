"""Routing accuracy evaluation scorers for telco support agent."""

from typing import Any, Optional

from databricks.agents.evals import metric
from mlflow.entities import Assessment, Feedback
from mlflow.genai.scorers import scorer
from mlflow.genai.judges import meets_guidelines

from ..utils import extract_request_text, extract_trace_routing_info


def _get_routing_accuracy_guidelines() -> list[str]:
    """Get guidelines for evaluating routing accuracy in telco support.

    Returns:
        List of guidelines for routing accuracy evaluation
    """
    return [
        "If the query is about account management, profile updates, personal information, or login issues, it should be routed to the 'account' agent",
        "If the query is about bills, payments, charges, refunds, payment methods, or billing disputes, it should be routed to the 'billing' agent",
        "If the query is about technical issues, internet problems, equipment troubleshooting, service outages, or connectivity problems, it should be routed to the 'tech_support' agent",
        "If the query is about plan information, upgrades, downgrades, new services, or feature questions, it should be routed to the 'product' agent",
        "The routed_agent must be the most appropriate specialist for the type of request in the query",
    ]


@metric
def routing_accuracy_metric(
    *,
    inputs: str,
    outputs: str,
    routed_agent: Optional[str] = None,
    topic: Optional[str] = None,
    **kwargs,
) -> Assessment:
    """Evaluate routing accuracy.

    Args:
        inputs: The customer's original query
        outputs: The agent's response (not directly used but required for compatibility)
        routed_agent: The agent type the query was routed to
        topic: The detected topic/category
        **kwargs: Additional parameters (ignored)

    Returns:
        Assessment object with binary score and rationale
    """
    try:
        query_text = extract_request_text(inputs)

        if not routed_agent:
            return Assessment(
                value=0.0, rationale="No routing information available for evaluation"
            )

        # guidelines judge to evaluate routing accuracy
        feedback = meets_guidelines(
            name="routing_accuracy",
            guidelines=_get_routing_accuracy_guidelines(),
            context={
                "query": query_text,
                "routed_agent": routed_agent,
                "detected_topic": topic or "not detected",
            },
        )

        # convert "yes"/"no" to numeric score for metric compatibility
        score = 1.0 if feedback.value == "yes" else 0.0

        return Assessment(value=score, rationale=feedback.rationale)

    except Exception as e:
        return Assessment(
            value=0.0, rationale=f"Error evaluating routing accuracy: {str(e)}"
        )


@scorer
def routing_accuracy_scorer(
    *, inputs: str, outputs: str, traces: Optional[dict[str, Any]] = None, **kwargs
) -> Feedback:
    """Evaluate routing accuracy.

    Args:
        inputs: The customer's original query
        outputs: The agent's response (not directly used but required for compatibility)
        traces: Trace containing routing details
        **kwargs: Additional parameters (ignored)

    Returns:
        Feedback object with binary score and rationale
    """
    try:
        query_text = extract_request_text(inputs)

        # get routing information from trace
        routing_info = extract_trace_routing_info(traces)
        routed_agent = routing_info.get("agent_type", "unknown")
        topic = routing_info.get("topic", "not detected")

        if routed_agent == "unknown":
            return Feedback(
                value="no", rationale="No routing information found in trace data"
            )

        # guidelines judge to evaluate routing accuracy
        feedback = meets_guidelines(
            name="routing_accuracy",
            guidelines=_get_routing_accuracy_guidelines(),
            context={
                "query": query_text,
                "routed_agent": routed_agent,
                "detected_topic": topic,
            },
        )

        enhanced_rationale = feedback.rationale
        enhanced_rationale += f" (Routed to: {routed_agent})"
        if routing_info.get("tools_used"):
            tools_str = ", ".join(routing_info["tools_used"])
            enhanced_rationale += f" (Tools used: {tools_str})"

        return Feedback(
            value=feedback.value,  # "yes" or "no"
            rationale=enhanced_rationale,
        )

    except Exception as e:
        return Feedback(
            value="no", rationale=f"Error evaluating routing accuracy: {str(e)}"
        )

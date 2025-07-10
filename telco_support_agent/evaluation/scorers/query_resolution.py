"""Query resolution evaluation scorers for telco support agent."""

from typing import Any, Optional

from databricks.agents.evals import metric
from mlflow.entities import Assessment, Feedback
from mlflow.genai.scorers import scorer
from mlflow.genai.judges import meets_guidelines

from ..utils import (
    extract_request_text,
    extract_response_text,
)


def _get_query_resolution_guidelines() -> list[str]:
    """Guidelines for evaluating query resolution in telco support.

    Returns:
        List of guidelines for query resolution evaluation
    """
    return [
        "The response must directly address the specific question or concern raised in the request",
        "If the customer has a problem, the response must provide a clear solution, workaround, or next steps to resolve it",
        "The response must not leave important questions unanswered or ignore key parts of the customer's request",
        "If complete resolution is not possible immediately, the response must explain why and provide a clear path forward",
        "For telco-specific queries (billing, service issues, account changes), the response must provide actionable information relevant to telecommunications services",
    ]


@metric
def query_resolution_metric(*, request: str, response: str, **kwargs) -> Assessment:
    """Evaluate if the response resolves the customer's query.

    Args:
        request: The customer's original query
        response: The agent's response
        **kwargs: Additional parameters (ignored)

    Returns:
        Assessment object with binary score and rationale
    """
    try:
        request_text = extract_request_text(request)
        response_text = extract_response_text(response)

        # guidelines judge to evaluate resolution
        feedback = meets_guidelines(
            name="query_resolution",
            guidelines=_get_query_resolution_guidelines(),
            context={"request": request_text, "response": response_text},
        )

        # convert "yes"/"no" to numeric score for metric compatibility
        score = 1.0 if feedback.value == "yes" else 0.0

        return Assessment(value=score, rationale=feedback.rationale)

    except Exception as e:
        return Assessment(
            value=0.0, rationale=f"Error evaluating query resolution: {str(e)}"
        )


@scorer
def query_resolution_scorer(
    *, request: str, response: str, trace: Optional[dict[str, Any]] = None, **kwargs
) -> Feedback:
    """Evaluate whether the response resolves the customer's query.

    Args:
        request: The customer's original query
        response: The agent's response
        trace: Optional trace information for additional context
        **kwargs: Additional parameters (ignored)

    Returns:
        Feedback object with binary score and rationale
    """
    try:
        request_text = extract_request_text(request)
        response_text = extract_response_text(response)

        # context with trace info
        context = {"request": request_text, "response": response_text}

        # guidelines judge to evaluate resolution
        feedback = meets_guidelines(
            name="query_resolution",
            guidelines=_get_query_resolution_guidelines(),
            context=context,
        )

        return Feedback(value=feedback.value, rationale=feedback.rationale)

    except Exception as e:
        return Feedback(
            value="no", rationale=f"Error evaluating query resolution: {str(e)}"
        )

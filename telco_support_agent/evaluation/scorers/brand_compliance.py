"""Brand compliance evaluation scorers for telco support agent."""

from typing import Any, Optional

from databricks.agents.evaluation import Assessment, Feedback, metric, scorer
from mlflow.genai.judges import meets_guidelines

from ..utils import (
    extract_request_text,
    extract_response_text,
)


def _get_telco_brand_guidelines() -> list[str]:
    """Brand compliance guidelines for telco customer service.

    Returns:
        List of guidelines for brand compliance evaluation
    """
    return [
        "The response must maintain a professional yet friendly tone appropriate for telecommunications customer service",
        "The response must not make promises about specific delivery dates, service restoration times, or technical capabilities without verification",
        "The response must use clear, non-technical language that customers can easily understand, explaining technical terms when necessary",
        "The response must show empathy and understanding for customer concerns, especially when addressing service issues or billing problems",
        "The response must provide actionable next steps or clear escalation paths when unable to fully resolve an issue",
        "The response must not include any discriminatory language or make assumptions about customer demographics or technical knowledge",
    ]


@metric
def brand_compliance_metric(*, request: str, response: str, **kwargs) -> Assessment:
    """Evaluate brand compliance for telco customer service.

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

        brand_guidelines = _get_telco_brand_guidelines()

        #  guidelines judge to evaluate brand compliance
        feedback = meets_guidelines(
            name="brand_compliance",
            guidelines=brand_guidelines,
            context={"request": request_text, "response": response_text},
        )

        # convert "yes"/"no" to numeric score for metric compatibility
        score = 1.0 if feedback.value == "yes" else 0.0

        return Assessment(value=score, rationale=feedback.rationale)

    except Exception as e:
        return Assessment(
            value=0.0, rationale=f"Error evaluating brand compliance: {str(e)}"
        )


@scorer
def brand_compliance_scorer(
    *, request: str, response: str, trace: Optional[dict[str, Any]] = None, **kwargs
) -> Feedback:
    """Evaluate brand compliance for telco customer service.

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

        brand_guidelines = _get_telco_brand_guidelines()

        # context with trace info
        context = {"request": request_text, "response": response_text}

        # guidelines judge to evaluate brand compliance
        feedback = meets_guidelines(
            name="brand_compliance", guidelines=brand_guidelines, context=context
        )

        return Feedback(
            value=feedback.value,  # "yes" or "no"
            rationale=feedback.rationale,
        )

    except Exception as e:
        return Feedback(
            value="no", rationale=f"Error evaluating brand compliance: {str(e)}"
        )

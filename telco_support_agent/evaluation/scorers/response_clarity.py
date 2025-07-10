"""Response clarity evaluation scorers for telco support agent."""

from typing import Any, Optional

from databricks.agents.evals import metric
from mlflow.entities import Assessment, Feedback
from mlflow.genai.scorers import scorer
from mlflow.genai.judges import custom_prompt_judge

from ..utils import (
    extract_request_text,
    extract_response_text,
)


def _get_clarity_judge():
    """Custom prompt judge for evaluating response clarity.

    Returns:
        Custom prompt judge for clarity evaluation
    """
    return custom_prompt_judge(
        name="response_clarity",
        prompt_template="""Evaluate the clarity and understandability of this telco customer service response:

Customer Request: {{request}}
Agent Response: {{response}}

Rate the clarity of the response based on these criteria:
- Language is clear and easy to understand
- Technical terms are explained when used
- Information is well-organized and logical
- Instructions or next steps are specific and actionable
- Response length is appropriate (not too verbose or too brief)

Choose the most appropriate clarity rating:

[[excellent]]: Crystal clear, perfectly organized, easy to follow for any customer
[[good]]: Clear and understandable with minor room for improvement
[[adequate]]: Generally clear but some parts could be clearer or better organized
[[poor]]: Confusing, unclear language, or difficult to understand""",
        numeric_values={"excellent": 1.0, "good": 0.75, "adequate": 0.5, "poor": 0.0},
    )


# Create the judge instance to reuse
_clarity_judge = _get_clarity_judge()


@metric
def response_clarity_metric(*, inputs: str, outputs: str, **kwargs) -> Assessment:
    """Evaluate the clarity of the response.

    Args:
        inputs: The customer's original query
        outputs: The agent's response
        **kwargs: Additional parameters (ignored)

    Returns:
        Assessment object with numeric score and rationale
    """
    try:
        request_text = extract_request_text(inputs)
        response_text = extract_response_text(outputs)

        # prompt-based judge to evaluate clarity
        feedback = _clarity_judge(request=request_text, response=response_text)

        return Assessment(
            value=feedback.value,
            rationale=feedback.rationale,
        )

    except Exception as e:
        return Assessment(
            value=0.0, rationale=f"Error evaluating response clarity: {str(e)}"
        )


@scorer
def response_clarity_scorer(
    *, inputs: str, outputs: str, traces: Optional[dict[str, Any]] = None, **kwargs
) -> Feedback:
    """Evaluate the clarity of the response.

    Args:
        inputs: The customer's original query
        outputs: The agent's response
        traces: Optional trace information for additional context
        **kwargs: Additional parameters (ignored)

    Returns:
        Feedback object with numeric score and rationale
    """
    try:
        request_text = extract_request_text(inputs)
        response_text = extract_response_text(outputs)

        # prompt-based judge to evaluate clarity
        feedback = _clarity_judge(request=request_text, response=response_text)

        return Feedback(
            value=feedback.value,
            rationale=feedback.rationale,
        )

    except Exception as e:
        return Feedback(
            value=0.0, rationale=f"Error evaluating response clarity: {str(e)}"
        )

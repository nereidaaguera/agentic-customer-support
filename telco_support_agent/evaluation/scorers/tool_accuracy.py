"""Tool accuracy evaluation scorers for telco support agent."""

from typing import Any, Optional

from databricks.agents.evals import metric
from mlflow.entities import Trace

from telco_support_agent.evaluation.scorers.base_scorer import GuidelinesScorer


class ToolAccuracyScorer(GuidelinesScorer):
    guidelines = [
        "If the query requires customer-specific information (account details, profile, status), the 'get_customer_info' tool should be called with the correct customer ID",
        "If the query is about billing, payments, or charges, the 'get_billing_info' tool should be called with appropriate date ranges and filters",
        "If the query asks about available plans, upgrades, or service options, the 'get_plans_info' tool should be called",
        "If the query is about technical issues or troubleshooting, technical support retrieval tools should be used to find relevant documentation",
        "Tools should be called with correct and complete parameters - missing required parameters or incorrect formats indicate poor tool usage",
        "Tools should not be called unnecessarily when the query can be answered without customer-specific data",
        "If customer-specific information is needed but no tools were called, this indicates missing tool usage",
    ]

    def __init__(self):
        super().__init__("tool_accuracy", guidelines=self.guidelines)

    def get_context(
        self,
        inputs: dict[str, Any],
        outputs: Optional[Any],
        trace: Optional[Trace] = None,
    ) -> dict[str, Any]:
        """Function to create the context used by the guidelines judge.

        Args:
            inputs: Model input represented as dict.
            outputs: Model output represented as dict.
            trace: Model execution trace as MLflow entity.
        """
        spans = trace.data.spans
        tool_calls = []
        for span in spans:
            if span.span_type == "TOOL":
                tool_name = span.name.replace("tool_", "")
                tool_calls.append(tool_name)
        tools_used = ",".join(tool_calls) if tool_calls else ""
        request = str(inputs["input"])
        response = str(outputs["output"][-1])
        return {"tools_used": tools_used, "request": request, "response": response}

    def get_custom_metric(self):
        """Implementation of custom metric for offline evaluation."""

        @metric
        def tool_accuracy(
            request: dict[str, Any],
            response: Optional[dict[str, Any]],
            trace: Optional[Trace] = None,
        ):
            from mlflow.genai.judges import meets_guidelines

            spans = trace.data.spans
            tool_calls = []
            for span in spans:
                if span.span_type == "TOOL":
                    tool_name = span.name.replace("tool_", "")
                    tool_calls.append(tool_name)
            tools_used = ",".join(tool_calls) if tool_calls else ""

            request = str(request["request"]["input"])
            response = str(response["output"][-1])

            context = {
                "tools_used": tools_used,
                "request": request,
                "response": response,
            }

            guidelines = [
                "If the query requires customer-specific information (account details, profile, status), the 'get_customer_info' tool should be called with the correct customer ID",
                "If the query is about billing, payments, or charges, the 'get_billing_info' tool should be called with appropriate date ranges and filters",
                "If the query asks about available plans, upgrades, or service options, the 'get_plans_info' tool should be called",
                "If the query is about technical issues or troubleshooting, technical support retrieval tools should be used to find relevant documentation",
                "Tools should be called with correct and complete parameters - missing required parameters or incorrect formats indicate poor tool usage",
                "Tools should not be called unnecessarily when the query can be answered without customer-specific data",
                "If customer-specific information is needed but no tools were called, this indicates missing tool usage",
            ]
            feedback = meets_guidelines(
                guidelines=guidelines, context=context, name="tool_accuracy"
            )
            return feedback

        return tool_accuracy

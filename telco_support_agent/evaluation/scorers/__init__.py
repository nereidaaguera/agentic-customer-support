"""Telco support agent evaluation scorers."""

# TODO: at the moment, the custom scorers cannnot be used in monitoring
# as such the current workaround is to use @metrics to define judges for monitoring
# future API release will allow for custom scorers to be used in monitoring
# when this is the case, we can remove the @metrics and use the custom scorers directly

from .brand_compliance import brand_compliance_metric, brand_compliance_scorer
from .data_privacy import data_privacy_metric, data_privacy_scorer
from .query_resolution import query_resolution_metric, query_resolution_scorer
from .response_clarity import response_clarity_metric, response_clarity_scorer
from .routing_accuracy import routing_accuracy_metric, routing_accuracy_scorer
from .tool_accuracy import tool_accuracy_metric, tool_accuracy_scorer

# metrics for production monitoring
ONLINE_METRICS = [
    brand_compliance_metric,
    data_privacy_metric,
    query_resolution_metric,
    response_clarity_metric,
    routing_accuracy_metric,
    tool_accuracy_metric,
]

# scorers for offline evaluation
OFFLINE_SCORERS = [
    brand_compliance_scorer,
    data_privacy_scorer,
    query_resolution_scorer,
    response_clarity_scorer,
    routing_accuracy_scorer,
    tool_accuracy_scorer,
]

__all__ = [
    "brand_compliance_metric",
    "brand_compliance_scorer",
    "data_privacy_metric",
    "data_privacy_scorer",
    "query_resolution_metric",
    "query_resolution_scorer",
    "response_clarity_metric",
    "response_clarity_scorer",
    "routing_accuracy_metric",
    "routing_accuracy_scorer",
    "tool_accuracy_metric",
    "tool_accuracy_scorer",
    "ONLINE_METRICS",
    "OFFLINE_SCORERS",
]

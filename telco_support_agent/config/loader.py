"""Config loader for dbutils widgets."""

import json
import logging
from typing import Any, Optional, TypeVar, get_args, get_origin

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

logger = logging.getLogger(__name__)


class WidgetConfigLoader:
    """Load config from widgets into Pydantic models."""

    def __init__(self, dbutils=None):
        self.dbutils = dbutils

    def load(self, config_class: type[T]) -> T:
        """Load config from widgets into the specified config class."""
        if not self.dbutils:
            return self._create_minimal_config(config_class)

        widget_values = {}

        for field_name, field_info in config_class.model_fields.items():
            try:
                value = self.dbutils.widgets.get(field_name)

                converted_value = self._convert_widget_value(value, field_info)
                if converted_value is not None:
                    widget_values[field_name] = converted_value

            except Exception as e:
                logger.debug(f"Failed to get widget '{field_name}': {e}")
                pass

        return config_class(**widget_values)

    def _convert_widget_value(self, value: str, field_info) -> Any:
        """Convert widget string value to appropriate type."""
        if not value:
            return None

        annotation = field_info.annotation

        if get_origin(annotation) is Optional:
            annotation = get_args(annotation)[0]

        if annotation is bool:
            return value.lower() in ("true", "1", "yes", "on")
        elif annotation is int:
            return int(value)
        elif annotation is float:
            return float(value)
        elif annotation is str:
            return value
        elif get_origin(annotation) is list:
            list_type = get_args(annotation)[0] if get_args(annotation) else str

            if list_type is str:
                # comma-separated string lists
                return [v.strip() for v in value.split(",") if v.strip()]
            elif list_type is dict:
                # JSON list of dicts
                return json.loads(value) if value else []
            else:
                # list handling
                return json.loads(value) if value else []
        elif get_origin(annotation) is dict:
            # JSON dictionaries
            return json.loads(value) if value else {}
        else:
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return value

    def _create_minimal_config(self, config_class: type[T]) -> T:
        """Create minimal config for local testing."""
        # for local testing get basic required fields
        minimal_data = {
            "env": "dev",
            "uc_catalog": "telco_customer_support_dev",
            "agent_schema": "agent",
            "data_schema": "gold",
            "model_name": "telco_customer_support_agent",
            "experiment_name": "/Shared/telco_support_agent/dev/dev_telco_support_agent",
            "endpoint_name": "telco-customer-support-agent-dev",
        }

        filtered_data = {}
        for field_name in config_class.model_fields.keys():
            if field_name in minimal_data:
                filtered_data[field_name] = minimal_data[field_name]

        return config_class(**filtered_data)

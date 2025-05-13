from abc import ABC, abstractmethod
from collections.abc import Callable
from enum import Enum
from io import StringIO
from typing import Any, Union

import pandas as pd
from unitycatalog.ai.core.databricks import DatabricksFunctionClient
from unitycatalog.ai.openai.toolkit import ChatCompletionToolParam, UCFunctionToolkit

from telco_support_agent.tools import ToolInfo


class FunctionType(Enum):
    SQL = 1
    PYTHON = 2


class UCTool(ABC):
    """Base UC Tool class all tools will inherit from."""

    def __init__(
        self,
        catalog: str,
        schema: str,
        function_name: str,
        function_type: FunctionType
    ) -> None:
        self.client = DatabricksFunctionClient()
        self.catalog = catalog
        self.schema = schema
        self.function_name = function_name
        self.uc_name = f"{self.catalog}.{self.schema}.{self.function_name}"
        self.function_type = function_type
        self.spec = self.create_function(self.create_function_value())
        # Remove strict parameter from function. it does not work with claude.
        self.spec["function"].pop("strict")

    def create_function(
        self, function_value: Union[str | Callable]
    ) -> ChatCompletionToolParam:
        """Create tool function in unity catalog."""
        if self.function_type is FunctionType.SQL:
            self.client.create_function(sql_function_body=function_value)
        else:
            self.client.create_python_function(
                func=function_value, catalog=self.catalog, schema=self.schema
            )
        toolkit = UCFunctionToolkit(function_names=[self.uc_name])
        return toolkit.tools[0]
    
    @abstractmethod
    def create_function_value(self) -> Union[str | Callable]:
        """Create function value for the tool."""
        pass

    @abstractmethod
    def exec_fn(self, **kwargs: dict[str, Any]) -> Any:
        """Executing of the function in unity catalog."""
        output = self.client.execute_function(self.uc_name, parameters=kwargs)
        return output.value

    def get_tool_info(self) -> ToolInfo:
        """Return tool info to an agent inherent from BaseAgent."""
        return ToolInfo(
            name=self.uc_name.replace(".", "__"),
            spec=self.spec,
            exec_fn=self.exec_fn,
        )


class AccountInfoTool(UCTool):
    SQL_BODY = """
                CREATE OR REPLACE FUNCTION {function_name}(customer STRING COMMENT 'ID of the customer whose info to look up.')
                RETURNS TABLE
                COMMENT 'Provides detailed information about a specific customer, including their registration date and address.'
                RETURN SELECT * from telco_customer_support_dev.bronze.customers where customer_id = customer LIMIT 1
            """

    def __init__(self) -> None:
        super().__init__(
            "telco_customer_support_dev",
            "bronze",
            "account_info_tool",
            FunctionType.SQL
        )

    def create_function_value(self) -> Union[str | Callable]:
        return self.SQL_BODY.format(
            function_name=f"{self.uc_name}")

    def exec_fn(self, **kwargs: dict[str, Any]) -> Any:
        """Executing of the function in unity catalog and formatting output to markdown."""
        result = super().exec_fn(**kwargs)
        df = pd.read_csv(StringIO(result))
        return df.to_markdown(index=False)


class PlansInfoTool(UCTool):

    SQL_BODY = """
                CREATE OR REPLACE FUNCTION {function_name}(customer STRING COMMENT 'ID of the customer whose info to look up.')
                RETURNS TABLE
                COMMENT 'Retrieves information regarding the particular plans purchased by the customer.'
                RETURN SELECT * EXCEPT(plans.plan_id) FROM telco_customer_support_dev.bronze.subscriptions,
                telco_customer_support_dev.bronze.plans where subscriptions.plan_id = plans.plan_id and subscriptions.customer_id = customer
            """

    def __init__(self) -> None:
        super().__init__(
            "telco_customer_support_dev",
            "bronze",
            "plans_info_tool",
            FunctionType.SQL
        )
    
    def create_function_value(self) -> Union[str | Callable]:
        return self.SQL_BODY.format(
            function_name=f"{self.uc_name}")

    def exec_fn(self, **kwargs: dict[str, Any]) -> Any:
        """Executing of the function in unity catalog and formatting output to markdown."""
        result = super().exec_fn(**kwargs)
        df = pd.read_csv(StringIO(result))
        return df.to_markdown(index=False)

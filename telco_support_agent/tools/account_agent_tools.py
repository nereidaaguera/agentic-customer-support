from abc import ABC, abstractmethod
from typing import Any

from unitycatalog.ai.core.databricks import DatabricksFunctionClient
from unitycatalog.ai.openai.toolkit import ChatCompletionToolParam, UCFunctionToolkit

from telco_support_agent.tools import ToolInfo


class UCTool(ABC):
    """Base UC Tool class all tools will inherit from."""

    def __init__(self, function_name: str):
        self.client = DatabricksFunctionClient()
        self.function_name = function_name
        self.spec = self.create_function()

    @abstractmethod
    def create_function(self) -> ChatCompletionToolParam:
        """Create tool function in unity catalog."""
        pass

    def get_tool_info(self) -> ToolInfo:
        """Return tool info to an agent inherent from BaseAgent."""

        def exec_fn(**kwargs: dict[str, Any]) -> Any:
            output = self.client.execute_function(self.function_name, parameters=kwargs)
            return output.value

        return ToolInfo(name=self.function_name, spec=self.spec, exec_fn=exec_fn)


class AccountInfoTool(UCTool):
    SQL_BODY = """
                CREATE OR REPLACE FUNCTION {function_name}(customer_id STRING COMMENT 'ID of the customer whose info to look up.')
                RETURNS TABLE
                COMMENT 'Returns data about a specific customer including their registration date and address.'
                RETURN SELECT * from telco_customer_support_dev.bronze.customers where customer_id = customer_id LIMIT 1
            """

    def __init__(self) -> None:
        super().__init__("telco_customer_support_dev.bronze.account_info_tool")

    def create_function(self) -> ChatCompletionToolParam:
        """Create tool function in unity catalog."""
        self.client.create_function(
            sql_function_body=self.SQL_BODY.format(function_name=self.function_name)
        )
        toolkit = UCFunctionToolkit(function_names=[self.function_name])
        return toolkit.tools[0]

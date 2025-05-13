from typing import Optional

from telco_support_agent.agents.base_agent import BaseAgent
from telco_support_agent.tools.account_agent_tools import (
    AccountInfoTool,
    SubscriptionsInfoTool,
)


class AccountAgent(BaseAgent):
    """Account agent to manage user's account queries.

    This agent answers questions related with user's account information.
    """

    uc_tools = [AccountInfoTool(), SubscriptionsInfoTool()]

    def __init__(
        self, llm_endpoint: Optional[str] = None, config_dir: Optional[str] = None
    ) -> None:
        tools = [uc_tool.get_tool_info() for uc_tool in self.uc_tools]
        super().__init__(
            agent_type="account",
            llm_endpoint=llm_endpoint,
            config_dir=config_dir,
            tools=tools,
        )

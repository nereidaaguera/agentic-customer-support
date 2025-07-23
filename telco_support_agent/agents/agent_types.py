"""Type definitions for telco support agents."""

from enum import Enum


class AgentType(str, Enum):
    """Enumeration of supported agent types."""

    ACCOUNT = "account"
    BILLING = "billing"
    TECH_SUPPORT = "tech_support"
    PRODUCT = "product"

    @classmethod
    def values(cls) -> list[str]:
        """Return a list of all valid agent type values."""
        return [member.value for member in cls]

    @classmethod
    def from_string(cls, value: str) -> "AgentType":
        """Convert a string to an AgentType."""
        value = value.lower()

        mapping = {member.value: member for member in cls}

        if value in mapping:
            return mapping[value]

        valid_values = ", ".join(mapping.keys())
        raise ValueError(
            f"Invalid agent type: {value}. Valid types are: {valid_values}"
        )

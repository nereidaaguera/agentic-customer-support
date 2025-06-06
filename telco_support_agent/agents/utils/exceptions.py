"""Custom exceptions for agent operations."""


class MissingCustomInputError(ValueError):
    """Raised when custom inputs are missing from the request."""

    def __init__(self, missing_inputs: list[str], required_inputs: list[str]):
        """Initialize the error with specific input details.

        Args:
            missing_inputs: List of missing input parameter names
            required_inputs: List of all required input parameter names
        """
        self.missing_inputs = missing_inputs
        self.required_inputs = required_inputs

        message = (
            f"Missing required custom inputs: {missing_inputs}. "
            f"This agent requires: {required_inputs}"
        )
        super().__init__(message)


class AgentConfigurationError(ValueError):
    """Raised when agent configuration is invalid."""

    def __init__(self, agent_type: str, config_issue: str):
        """Initialize the error with agent and config details.

        Args:
            agent_type: Type of agent that has configuration issues
            config_issue: Description of the configuration problem
        """
        self.agent_type = agent_type
        self.config_issue = config_issue

        message = f"Invalid configuration for {agent_type} agent: {config_issue}"
        super().__init__(message)


class ToolExecutionError(Exception):
    """Raised when tool execution fails."""

    def __init__(
        self, tool_name: str, error_details: str, original_error: Exception = None
    ):
        """Initialize the error with tool execution details.

        Args:
            tool_name: Name of the tool that failed
            error_details: Description of what went wrong
            original_error: Original exception that caused the failure
        """
        self.tool_name = tool_name
        self.error_details = error_details
        self.original_error = original_error

        message = f"Error executing tool {tool_name}: {error_details}"
        super().__init__(message)


class AgentRoutingError(Exception):
    """Raised when supervisor agent cannot route a query properly."""

    def __init__(self, query: str, routing_details: str):
        """Initialize the error with routing details.

        Args:
            query: The user query that couldn't be routed
            routing_details: Description of the routing problem
        """
        self.query = query
        self.routing_details = routing_details

        message = f"Failed to route query '{query[:100]}...': {routing_details}"
        super().__init__(message)


class VectorSearchError(Exception):
    """Raised when vector search operations fail."""

    def __init__(self, index_name: str, query: str, error_details: str):
        """Initialize the error with vector search details.

        Args:
            index_name: Name of the vector search index
            query: The search query that failed
            error_details: Description of what went wrong
        """
        self.index_name = index_name
        self.query = query
        self.error_details = error_details

        message = f"Vector search failed on index {index_name}: {error_details}"
        super().__init__(message)

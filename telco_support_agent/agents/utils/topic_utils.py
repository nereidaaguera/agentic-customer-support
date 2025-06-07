"""Topic classification utilities for the telco support agent."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import yaml
from mlflow.deployments import get_deploy_client

from telco_support_agent import PROJECT_ROOT

logging.basicConfig(
    level=logging.ERROR, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
_logger = logging.getLogger(__name__)


def run_llm(
    message: str,
    system_prompt: Optional[str] = None,
    model: str = "gpt-4o",
) -> str:
    """Run LLM query using Databricks deployment client.

    Args:
        message: The user message to send to the LLM
        system_prompt: Optional system prompt to include
        model: The model endpoint to use (default: gpt-4o)

    Returns:
        The LLM response content
    """
    messages = [{"role": "user", "content": message}]
    if system_prompt is not None:
        messages = [{"role": "system", "content": system_prompt}] + messages

    return get_deploy_client("databricks").predict(
        endpoint=model,
        inputs={"messages": messages},
    )["choices"][0]["message"]["content"]


@dataclass(frozen=True, eq=True)
class TopicCategory:
    """Represents a topic category for classification.

    Args:
        name: The name of the topic category
        description: Optional description of the topic category
    """

    name: str
    description: Optional[str] = None

    def __hash__(self) -> int:
        """Generate hash for the topic category."""
        return hash((self.name, self.description or ""))

    @classmethod
    def from_dict(cls, topic_dict: dict[str, Any]) -> "TopicCategory":
        """Create a TopicCategory from a dictionary.

        Args:
            topic_dict: Dictionary containing topic information

        Returns:
            TopicCategory instance
        """
        return cls(
            name=topic_dict.get("name") or topic_dict.get("topic"),
            description=topic_dict.get("description"),
        )


def _create_topic_classification_prompt(
    message: str, topic_categories: list[TopicCategory]
) -> str:
    """Create a prompt for topic classification."""
    formatted_topic_categories = "\n".join(
        [
            f"""    <topic>
            <name>{topic_category.name}</name>
            <description>{topic_category.description or "No description"}</description>
            </topic>"""
            for topic_category in topic_categories
        ]
    )[4:]

    return f"""You are classifying customer support queries for a mobile/wireless telecommunications company.

Consider the following customer message and topic categories. Categorize the message into at most one of the existing topic categories based on the primary intent of the customer's inquiry.

    <message>{message}</message>
    <topic_categories>
        {formatted_topic_categories}
    </topic_categories>

    Please return the result using the following JSON format. Do not use any newlines or additional spaces:
    {{
        "rationale": "Brief explanation of why this topic was chosen based on the customer's primary intent.",
        "topic": "The chosen topic name. If none of the topics are applicable, return 'other'"
    }}
    """


def topic_classification(
    content: str, topic_categories: list[TopicCategory], model: str = "gpt-4o"
) -> dict[str, str]:
    """Classify content into topics using an LLM.

    Args:
        content: The text content to classify
        topic_categories: List of topic categories
        model: The LLM model to use for classification

    Returns:
        Dictionary with 'topic' and 'rationale' keys
    """
    if not content.strip() or not topic_categories:
        return {"topic": "other", "rationale": "Empty content or no categories"}

    try:
        prompt = _create_topic_classification_prompt(content, topic_categories)
        result = run_llm(prompt, model=model)

        deserialized_result = json.loads(result)
        topic = deserialized_result.get("topic", "other")
        rationale = deserialized_result.get("rationale", "No rationale provided")

        # Validate topic is one of the available categories or 'other'
        valid_topics = {cat.name for cat in topic_categories} | {"other"}
        if topic not in valid_topics:
            topic = "other"
            rationale = "Invalid topic returned, defaulting to other"

        return {"topic": topic, "rationale": rationale}

    except Exception as e:
        _logger.error(f"Failed to classify content: {str(e)}")
        return {"topic": "other", "rationale": f"Classification error: {str(e)}"}


def load_topics_from_yaml(
    yaml_path: Optional[str | Path] = None,
) -> list[TopicCategory]:
    """Load topic categories from a YAML file.

    Args:
        yaml_path: Optional path to the YAML file. If not provided, will search for topics.yaml
                  in the project's configs directory.

    Returns:
        List of TopicCategory objects loaded from the YAML file.

    Raises:
        FileNotFoundError: If the YAML file cannot be found
        yaml.YAMLError: If the YAML file is malformed
        ValueError: If the YAML structure is invalid
    """
    if yaml_path is None:
        search_paths = [
            PROJECT_ROOT / "configs" / "topics.yaml",
            Path("/Workspace/Repos/") / "*/telco-support-agent/configs/topics.yaml",
            Path.cwd() / "configs" / "topics.yaml",
            Path("/model/artifacts/configs") / "topics.yaml",
            Path("/model/artifacts") / "topics.yaml",
        ]

        yaml_path = None
        for path in search_paths:
            if path.exists():
                yaml_path = path
                break

        if yaml_path is None:
            raise FileNotFoundError(
                "Could not find topics.yaml in any of the expected locations"
            )

    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict) or "topics" not in data:
        raise ValueError(
            "YAML file must contain a 'topics' key with a list of topic definitions"
        )

    if not isinstance(data["topics"], list):
        raise ValueError("'topics' must be a list of topic definitions")

    topics = []
    for topic_data in data["topics"]:
        try:
            topic = TopicCategory.from_dict(topic_data)
            if topic.name:
                topics.append(topic)
        except Exception as e:
            _logger.warning(f"Skipping invalid topic: {e}")
            continue

    return topics

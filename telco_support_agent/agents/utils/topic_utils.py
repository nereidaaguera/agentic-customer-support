"""Topic classification utilities for the telco support agent."""

import json
import logging
from pathlib import Path
from typing import Optional

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
    model: str = "dbdemos-openai-gpt4",
) -> str:
    """Run LLM query using Databricks deployment client.

    Args:
        message: The user message to send to the LLM
        system_prompt: Optional system prompt to include
        model: The model endpoint to use (default: dbdemos-openai-gpt4)

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


# Removed TopicCategory class - using simple dictionaries instead


def _create_topic_classification_prompt(
    message: str, topic_categories: list[dict]
) -> str:
    """Create a prompt for topic classification."""
    formatted_topic_categories = "\n".join(
        [
            f"""    <topic>
            <name>{topic.get('name', '')}</name>
            <description>{topic.get('description', 'No description')}</description>
            </topic>"""
            for topic in topic_categories
        ]
    )[4:]

    return f"""You are classifying customer support queries for a mobile/wireless telecommunications company.

Consider the following customer message and topic categories. Categorize the message into at most one of the existing topic categories based on the primary intent of the customer's inquiry.

If the query spans multiple domains or involves complex interconnected issues across different service areas, use "multi_domain".

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
    content: str,
    topic_categories: list[dict],
    model: str = "dbdemos-openai-gpt4",
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
        valid_topics = {cat.get('name', '') for cat in topic_categories} | {"other"}
        if topic not in valid_topics:
            topic = "other"
            rationale = "Invalid topic returned, defaulting to other"

        return {"topic": topic, "rationale": rationale}

    except Exception as e:
        _logger.error(f"Failed to classify content: {str(e)}")
        return {"topic": "other", "rationale": f"Classification error: {str(e)}"}


def load_topics_from_yaml(yaml_path: Optional[str | Path] = None) -> list[dict]:
    """Load topic categories from a YAML file.

    Args:
        yaml_path: Optional path to the YAML file. If not provided, will search for topics.yaml

    Returns:
        List of topic dictionaries loaded from the YAML file.
    """
    if yaml_path is None:
        # Simple path checking - check the main scenarios we know work
        search_paths = [
            # Model serving: MLflow flattens artifacts to /model/artifacts/
            Path("/model/artifacts") / "topics.yaml",
            # Development/notebook: configs in project structure  
            PROJECT_ROOT / "configs" / "agents" / "topics.yaml",
            # Current working directory (for local development)
            Path.cwd() / "configs" / "agents" / "topics.yaml",
        ]

        yaml_path = None
        for path in search_paths:
            if path.exists():
                yaml_path = path
                break

        if yaml_path is None:
            raise FileNotFoundError("Could not find topics.yaml")

    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    # Simple validation and return the topics list
    topics = data.get("topics", [])
    return [topic for topic in topics if topic.get("name")]

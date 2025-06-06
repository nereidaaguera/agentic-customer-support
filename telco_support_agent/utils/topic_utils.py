import json
import logging
import yaml
from pathlib import Path

from dataclasses import dataclass
from typing import List, Dict, Optional, Any

from mlflow.deployments import get_deploy_client

from telco_support_agent.agents.config import config_manager

logging.basicConfig(
    level=logging.ERROR, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
_logger = logging.getLogger(__name__)


def run_llm(
    message: str,
    system_prompt: Optional[str] = None,
    model: str = "gpt-4o",
):
    messages = [{"role": "user", "content": message}]

    if system_prompt is not None:
        messages = [{"role": "system", "content": system_prompt}] + messages

    return get_deploy_client("databricks").predict(
        endpoint=model,
        inputs={"messages": messages},
    )["choices"][0]["message"]["content"]


@dataclass(frozen=True, eq=True)
class TopicCategory:
    name: str
    description: Optional[str] = None

    def __hash__(self):
        return hash((self.name, self.description or ""))

    @classmethod
    def from_dict(cls, topic_dict: Dict[str, Any]) -> "TopicCategory":
        return cls(
            name=topic_dict.get("name") or topic_dict.get("topic"),
            description=topic_dict.get("description"),
        )


def _create_topic_classification_prompt(
    message: str, topic_categories: List[TopicCategory]
):
    formatted_topic_categories = "\n".join(
        [
            f"""    <topic>
            <name>{topic_category.name}</name>
            <description>{topic_category.description}</description>
            </topic>"""
            for topic_category in topic_categories
        ]
    )[4:]

    return f"""Consider the following message and topic categories, each containing a name and description. You must categorize the message into at most one of the existing topic categories using the topic name and description to inform the categorization. Do not return any markdown.

    <message>{message}</message>
    <topic_categories>
        {formatted_topic_categories}
    </topic_categories>

    Please return the result using the following JSON format. Do not use any newlines or additional spaces:
    {{
        "rationale": Reason for the categorization.,
        "topic": The chosen topic. If none of the topics are applicable, return 'other',
    }}
    """


def topic_classification(content: str, topic_categories: List[TopicCategory]):
    """
    Classify content into topics using an LLM

    Args:
        content: The text content to classify
        topic_categories: List of topic categories

    Returns:
        Classification result from the model
    """
    try:
        result = run_llm(
            _create_topic_classification_prompt(content, topic_categories)
        )
        deserialized_result = json.loads(result)
        topic = deserialized_result["topic"]
        rationale = deserialized_result["rationale"]
        return {"topic": topic, "rationale": rationale}
    except Exception as e:
        _logger.error(f"Failed to classify {content}: {str(e)}")
        _logger.error(f"Classification response: {result}")
        return {}


def load_topics_from_yaml(yaml_path: Optional[str | Path] = None) -> List[TopicCategory]:
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
        # Use config manager to find project root and look for topics.yaml
        project_root = config_manager._project_root
        search_paths = [
            project_root / "configs" / "topics.yaml",
            Path("/Workspace/Repos/") / "*/telco-support-agent/configs/topics.yaml",
            Path.cwd() / "configs" / "topics.yaml",
            Path("/model/artifacts/configs") / "topics.yaml",
            Path("/model/artifacts") / "topics.yaml",
        ]

        for path in search_paths:
            if isinstance(path, Path) and path.exists():
                yaml_path = path
                _logger.info(f"Found topics file at {path}")
                break
            elif isinstance(path, str) and Path(path).exists():
                yaml_path = Path(path)
                _logger.info(f"Found topics file at {path}")
                break

        if yaml_path is None:
            raise FileNotFoundError("Could not find topics.yaml in any of the expected locations")

    try:
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict) or 'topics' not in data:
            raise ValueError("YAML file must contain a 'topics' key with a list of topic definitions")

        if not isinstance(data['topics'], list):
            raise ValueError("'topics' must be a list of topic definitions")

        return [TopicCategory.from_dict(topic) for topic in data['topics']]
    except FileNotFoundError:
        _logger.error(f"Topics YAML file not found at {yaml_path}")
        raise
    except yaml.YAMLError as e:
        _logger.error(f"Error parsing topics YAML file: {e}")
        raise
    except Exception as e:
        _logger.error(f"Error loading topics: {e}")
        raise

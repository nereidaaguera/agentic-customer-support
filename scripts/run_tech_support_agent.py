import logging
import os
import sys

# Add the project root to Python path
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from telco_support_agent.agents.tech_support import TechSupportAgent

logger = logging.getLogger("mlflow")
logger.setLevel(logging.WARNING)
AGENT = TechSupportAgent(
    config_dir=os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "configs"
    )
)


def _query_agent(query):
    for event in AGENT.predict_stream({"input": [{"role": "user", "content": query}]}):
        if item := getattr(event, "item", None):
            item_type = item.get("type")

            # When the model requests a function/tool call:
            if item_type == "function_call":
                func_name = item.get("name")
                args_json = item.get("arguments", "{}")
                # Pretty-print the tool name and its arguments
                print(f"\n→ Calling tool: {func_name}({args_json})")

            # When the tool returns its output:
            elif item_type == "function_call_output":
                raw_output = item.get("output", "")
                print(f"Tool call result: {raw_output}")

            # 2) Otherwise, if this event has assistant content (text chunks), print them:
            elif item_type == "message":
                for chunk in item.get("content", []):
                    if chunk.get("type") == "output_text":
                        # 'text' is a string containing whatever the assistant is saying
                        print(chunk["text"], end="")
                print("")
            else:
                print(f"Unexpected agent output item: {item}")


query = "Is there an outage in Moscone center?"
_query_agent(query=query)

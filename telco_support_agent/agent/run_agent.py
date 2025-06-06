import logging
from agent import AGENT

def query_agent(query):
    assistant_response = {
        "type": "message",
        "id": "0bd86922-aa94-4857-817f-4b1bcf653643",
        "content": [
            {
                "type": "output_text",
                "text": "Hello! I'm here to help you with any questions you might have about telecommunications services, billing information, subscription plans, devices, or technical support. How can I assist you today?"
            }
        ],
        "role": "assistant"
    }
    for event in AGENT.predict_stream({"input": [{"role": "user", "content": "hi"}, assistant_response, {"role": "user", "content": query}]}):
        if item:= getattr(event, "item", None):
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
                print(f"Unexpected agent output item, displaying it anyways: {item}")


if __name__ == "__main__":
    logger = logging.getLogger("mlflow")
    logger.setLevel(logging.WARNING)

    query = "What was customer CUS-10001's average total bill per month over the last year?"
    # query = "What can I do if I have added charges on my bill?"
    query_agent(query=query)


from agent import AGENT

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.WARNING)
    # user_query = "What was my bill for the last few months (my customer ID is CUS-10001)?"
    user_query = "Is there an outage in Moscone center right now?"
    for event in AGENT.predict_stream({"messages": [{"role": "user", "content": user_query}]}):
        print(event)

    # for chunk in AGENT.predict_stream({"messages": [{"role": "user", "content": user_query}]}):
    #     if chunk.delta.tool_calls:
    #         print(chunk.delta.tool_calls)
    #     print(chunk.delta.content)

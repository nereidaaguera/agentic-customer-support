from agent import AGENT

if __name__ == "__main__":
    # user_query = "What was my bill for the last few months (my customer ID is CUS-10001)?"
    # user_query = "Is there an outage in Moscone center right now?"
    user_query = "What is the 200th fibonacci #?"

    for chunk in AGENT.predict_stream({"messages": [{"role": "user", "content": user_query}]}):
        print(chunk.delta.content)
        if chunk.delta.tool_calls:
            print("Tool calls: ")
            print(chunk.delta.tool_calls)

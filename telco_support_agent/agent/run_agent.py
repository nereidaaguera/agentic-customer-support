from agent import AGENT

if __name__ == "__main__":
    user_query = "What was my bill for the last few months (my customer ID is CUS-10001)?"

    for chunk in AGENT.predict_stream({"messages": [{"role": "user", "content": user_query}]}):
        print(chunk.delta.content, "-----------\n")

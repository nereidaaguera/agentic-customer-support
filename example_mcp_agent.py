#!/usr/bin/env python3
"""Example usage of the tech support agent with MCP integration."""

import os
import sys

# Add project root to path
root_path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, root_path)

from mlflow.types.responses import ResponsesAgentRequest
from telco_support_agent.agents.tech_support import TechSupportAgent
from telco_support_agent.utils.logging_utils import setup_logging

def query_agent(agent, query):
    """Query the agent and print the response."""
    print(f"\n{'='*80}")
    print(f"QUERY: {query}")
    print('='*80)
    
    request = ResponsesAgentRequest(
        input=[{"role": "user", "content": query}]
    )
    
    try:
        # Get streaming response for better debugging
        for event in agent.predict_stream(request):
            if hasattr(event, "item") and event.item:
                item = event.item
                item_type = item.get("type")
                
                # Function calls show which tools are being used
                if item_type == "function_call":
                    func_name = item.get("name", "unknown")
                    args = item.get("arguments", "{}")
                    print(f"\n→ Calling tool: {func_name}({args})")
                
                # Tool outputs show what the tools returned
                elif item_type == "function_call_output":
                    output = item.get("output", "")[:200]
                    print(f"   Tool result: {output}...")
                
                # Final assistant message
                elif item_type == "message":
                    for chunk in item.get("content", []):
                        if chunk.get("type") == "output_text":
                            print(f"\nRESPONSE:")
                            print(chunk["text"])
                            break
    except Exception as e:
        print(f"Error querying agent: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n{'='*80}\n")

def main():
    """Demo the MCP-enabled tech support agent."""
    setup_logging()
    
    print("MCP-Enabled Tech Support Agent Demo")
    print("=" * 50)
    
    # Initialize agent with MCP servers
    print("Initializing agent...")
    agent = TechSupportAgent(
        environment="prod",
        mcp_server_urls=[
            "https://db-ml-models-dev-us-west.cloud.databricks.com/api/2.0/mcp/vector-search/telco_customer_support_dev/mcp_agent",
            "https://db-ml-models-dev-us-west.cloud.databricks.com/api/2.0/mcp/functions/telco_customer_support_dev/mcp_agent",
            "https://mcp-telco-outage-server-3217006663075879.aws.databricksapps.com/mcp/"
        ]
    )
    
    print(f"✓ Agent initialized with {len(agent.tools)} total tools")
    print(f"  - {len(agent.mcp_tool_infos)} MCP tools")
    print(f"  - {len(agent.tools) - len(agent.mcp_tool_infos)} traditional tools")
    
    # Demo queries that should trigger different types of tools
    queries = [
        # Should trigger outage checking MCP tool
        "I'd like to report an outage at Moscone center - I can't send texts, and it seems like a total outage",
        
        # Should trigger network metrics MCP tool  
        "What are the current network performance metrics in the San Francisco area?",
        
        # Should trigger traditional knowledge base search
        "How do I set up international roaming on my iPhone?",
        
        # Should trigger support tickets search for troubleshooting
        "My phone keeps dropping calls and I can't connect to data networks"
    ]
    
    for query in queries:
        query_agent(agent, query)

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""Simple test script to verify MCP integration in tech support agent."""

import os
import sys

# Add project root to path
root_path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, root_path)

from mlflow.types.responses import ResponsesAgentRequest
from telco_support_agent.agents.tech_support import TechSupportAgent
from telco_support_agent.utils.logging_utils import setup_logging

def test_mcp_integration():
    """Test that tech support agent can initialize with MCP tools."""
    print("Testing MCP integration...")
    
    # Initialize with limited MCP servers for testing
    try:
        agent = TechSupportAgent(
            environment="prod",
            mcp_server_urls=[
                "https://mcp-telco-outage-server-3217006663075879.aws.databricksapps.com/mcp/"
            ]
        )
        print(f"✓ Agent initialized successfully")
        print(f"  Agent type: {agent.agent_type}")
        print(f"  Total tools: {len(agent.tools)}")
        print(f"  MCP tools discovered: {len(agent.mcp_tool_infos)}")
        
        # List available tools
        print("\nAvailable tools:")
        for i, tool in enumerate(agent.tools, 1):
            if "function" in tool:
                tool_name = tool["function"]["name"]
                tool_desc = tool["function"].get("description", "No description")[:80] + "..."
                print(f"  {i}. {tool_name}: {tool_desc}")
        
        # Test a simple query that might use MCP tools
        print("\nTesting outage query...")
        request = ResponsesAgentRequest(
            input=[{"role": "user", "content": "Are there any outages in the Moscone Center area?"}]
        )
        
        response = agent.predict(request)
        
        if response and hasattr(response, 'output') and response.output:
            print("✓ Query executed successfully")
            # Extract the final message response
            for output_item in reversed(response.output):
                if (hasattr(output_item, 'type') and output_item.type == "message" and 
                    hasattr(output_item, 'content')):
                    if isinstance(output_item.content, list):
                        for content_item in output_item.content:
                            if (hasattr(content_item, 'type') and 
                                content_item.type == "output_text"):
                                print(f"Response: {content_item.text[:200]}...")
                                break
                    break
        else:
            print("⚠ No response received")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    setup_logging()
    
    print("MCP Integration Test")
    print("=" * 40)
    
    success = test_mcp_integration()
    
    if success:
        print("\n✓ MCP integration test completed successfully!")
    else:
        print("\n✗ MCP integration test failed!")
        sys.exit(1)
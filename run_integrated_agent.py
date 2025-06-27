#!/usr/bin/env python3
"""
Run the integrated tech support agent with MCP capabilities.
Similar to telco_support_agent/agent/run_agent.py but using the merged functionality.
"""

import logging
import os
import sys

# Add project root to path
root_path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, root_path)

from telco_support_agent.utils.logging_utils import setup_logging

# Import the enhanced TechSupportAgent
try:
    from telco_support_agent.agents.tech_support import TechSupportAgent
    from mlflow.types.responses import ResponsesAgentRequest
    INTEGRATION_AVAILABLE = True
except ImportError as e:
    print(f"Import error: {e}")
    INTEGRATION_AVAILABLE = False


def query_agent(agent, query):
    """Query the agent and display the response in a user-friendly format."""
    print(f"\n{'='*80}")
    print(f"QUERY: {query}")
    print('='*80)
    
    try:
        request = ResponsesAgentRequest(
            input=[{"role": "user", "content": query}]
        )
        
        # Use streaming to show tool calls and outputs
        for event in agent.predict_stream(request):
            if item := getattr(event, "item", None):
                item_type = item.get("type")

                # When the model requests a function/tool call:
                if item_type == "function_call":
                    func_name = item.get("name")
                    args_json = item.get("arguments", "{}")
                    print(f"\n→ Calling tool: {func_name}({args_json})")

                # When the tool returns its output:
                elif item_type == "function_call_output":
                    raw_output = item.get("output", "")
                    # Truncate long outputs for readability
                    display_output = raw_output if len(raw_output) <= 200 else raw_output[:200] + "..."
                    print(f"   Tool result: {display_output}")

                # Final assistant message
                elif item_type == "message":
                    for chunk in item.get("content", []):
                        if chunk.get("type") == "output_text":
                            print(f"\nRESPONSE:")
                            print(chunk["text"])
                            break
                else:
                    print(f"Unexpected agent output item: {item}")
                    
    except Exception as e:
        print(f"Error querying agent: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n{'='*80}\n")


def demo_agent_functionality():
    """Demonstrate the enhanced agent with both traditional and MCP tools."""
    print("Enhanced Tech Support Agent with MCP Integration")
    print("=" * 60)
    
    if not INTEGRATION_AVAILABLE:
        print("❌ Integration not available - missing dependencies or authentication")
        return
    
    print("\n🔧 Initializing agent...")
    try:
        # Initialize agent with both traditional and MCP capabilities
        # In a real environment with proper auth, this would work
        agent = TechSupportAgent(
            environment="prod",  # Would normally use vector search
            mcp_server_urls=[
                # Your custom outage server
                "https://db-ml-models-prod-us-west.cloud.databricks.com/api/2.0/mcp/functions/system/ai",
            ]
        )
        
        print(f"✅ Agent initialized successfully!")
        print(f"   Agent type: {agent.agent_type}")
        print(f"   Total tools: {len(agent.tools)}")
        print(f"   MCP tools: {len(agent.mcp_tool_infos)}")
        print(f"   Traditional tools: {len(agent.tools) - len(agent.mcp_tool_infos)}")
        
        print(f"\n📋 Available tools:")
        for i, tool in enumerate(agent.tools, 1):
            if "function" in tool:
                tool_name = tool["function"]["name"]
                tool_desc = tool["function"].get("description", "No description")
                # Mark MCP tools
                is_mcp = any(t.name == tool_name for t in agent.mcp_tool_infos)
                marker = "🔌 MCP" if is_mcp else "🔍 Traditional"
                print(f"   {i}. {marker} {tool_name}")
                print(f"      {tool_desc[:80]}...")
        
        # Demo queries that would trigger different tool types
        demo_queries = [
            # Should trigger MCP outage tool
            "I'd like to report an outage at Moscone center - I can't send texts, and it seems like a total outage",
            
            # Should trigger MCP network metrics tool
            "What are the current network performance metrics in the San Francisco area?",
            
            # Should trigger traditional knowledge base search
            "How do I set up international roaming on my iPhone?",
            
            # Should trigger traditional support tickets search
            "My phone keeps dropping calls and I can't connect to data networks"
        ]
        
        print(f"\n🧪 Demo queries (would execute with proper auth):")
        for i, query in enumerate(demo_queries, 1):
            print(f"{i}. {query}")
            # In real environment: query_agent(agent, query)
        
        return True
        
    except Exception as e:
        raise
        print(f"❌ Error initializing agent: {e}")
        # This is expected without proper Databricks authentication
        print("   (This is expected without Databricks authentication)")
        return False


def simulate_mcp_workflow():
    """Simulate how the MCP workflow would execute."""
    print("\n🔄 Simulated MCP Workflow:")
    print("-" * 40)
    
    print("1. Agent initialization:")
    print("   ✓ Load traditional tools (knowledge base, support tickets)")
    print("   ✓ Discover MCP tools from configured servers")
    print("   ✓ Combine tool specs for LLM")
    
    print("\n2. Query processing:")
    print("   📥 User: 'Are there any outages in Moscone Center?'")
    print("   🤖 LLM decides to call: check_outage_status_tool(region='Moscone Center')")
    print("   🔌 Execute MCP tool via async session")
    print("   📤 Return outage information to user")
    
    print("\n3. Tool execution flow:")
    print("   ✓ Check if tool is MCP tool (yes)")
    print("   ✓ Use MCP exec_fn with proper authentication")
    print("   ✓ Return results to LLM for final response")
    
    print("\n4. Benefits:")
    print("   ✓ Dynamic tool discovery from MCP servers")
    print("   ✓ Seamless integration with existing agent framework")
    print("   ✓ Preserves all existing functionality")
    print("   ✓ Proper error handling and logging")


def main():
    """Main function to demonstrate the integration."""
    # Set up logging
    setup_logging()
    logger = logging.getLogger("mlflow")
    logger.setLevel(logging.WARNING)
    
    print("🚀 Tech Support Agent with MCP Integration Demo")
    print("=" * 60)
    
    # Try to demo with real initialization (will likely fail due to auth)
    agent_initialized = demo_agent_functionality()
    
    if not agent_initialized:
        print("\n⚠️  Real demo failed due to authentication/environment issues")
        # print("    Showing simulated workflow instead...")
        # simulate_mcp_workflow()
    
    # print("\n✅ Integration Summary:")
    # print("   🔗 MCP functionality successfully merged into TechSupportAgent")
    # print("   🛠️  Agent supports both traditional and MCP tools")
    # print("   📡 Ready for deployment with proper Databricks authentication")
    # print("   🔄 Maintains backwards compatibility with existing multi-agent system")
    #
    # print(f"\n📚 Next steps:")
    # print("   1. Set up Databricks authentication: databricks auth login")
    # print("   2. Deploy MCP server using: telco_support_agent/mcp_servers/outage_info_server/README.md")
    # print("   3. Test with: python example_mcp_agent.py")
    # print("   4. Run notebook tests: notebooks/02_run_agent/standalone_sub_agent/test_tech_support_agent.py")


if __name__ == "__main__":
    main()
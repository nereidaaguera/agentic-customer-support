from starlette.responses import HTMLResponse, JSONResponse

# --- Demo web interface routes ---
async def demo_homepage(request):
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Telco Operations MCP Server Updated</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
            .container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            .header { text-align: center; margin-bottom: 40px; }
            .tool-section { margin: 30px 0; padding: 20px; border: 1px solid #ddd; border-radius: 8px; }
            .button { background: #007cba; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; margin: 5px; }
            .button:hover { background: #005a8a; }
            .response-box { background: #f8f9fa; padding: 15px; border-radius: 4px; margin-top: 10px; border-left: 4px solid #007cba; }
            .loading { color: #666; font-style: italic; }
            pre { white-space: pre-wrap; }
            .status { padding: 8px 12px; border-radius: 4px; margin: 10px 0; }
            .status.running { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🏢 Telco Operations MCP Server</h1>
                <p>Network Operations Team's Established Tools</p>
                <div class="status running">✅ Server Status: RUNNING</div>
            </div>
            
            <div class="tool-section">
                <h2>📡 Check Outage Status</h2>
                <p>Monitor current network outages across regions</p>
                <button id="btn-check-outage-all" class="button">Check All Regions</button>
                <button id="btn-check-outage-bay" class="button">Bay Area Only</button>
                <div id="outage-response" class="response-box" style="display:none;"></div>
            </div>
            
            <div class="tool-section">
                <h2>📊 Network Metrics</h2>
                <p>Get real-time network performance metrics</p>
                <button id="btn-metrics-all" class="button">All Regions</button>
                <button id="btn-metrics-dtla" class="button">Downtown LA</button>
                <div id="metrics-response" class="response-box" style="display:none;"></div>
            </div>
            
            <div class="tool-section">
                <h2>⚠️ Report Network Issue</h2>
                <p>Report incidents to the operations team</p>
                <button id="btn-report-degraded" class="button">Report Degraded Performance</button>
                <div id="report-response" class="response-box" style="display:none;"></div>
            </div>
            
            <div class="tool-section">
                <h2>🔧 Available MCP Tools</h2>
                <button id="btn-list-tools" class="button">List All Tools</button>
                <div id="tools-response" class="response-box" style="display:none;"></div>
            </div>
        </div>
        
        <script>
            async function callTool(toolName, args = {}) {
                let responseDivId;
                switch(toolName) {
                    case 'check-outage-status':
                        responseDivId = 'outage-response';
                        break;
                    case 'get-network-metrics':
                        responseDivId = 'metrics-response';
                        break;
                    case 'report-network-issue':
                        responseDivId = 'report-response';
                        break;
                    default:
                        console.error("Unknown tool name:", toolName);
                        return;
                }
    
                const div = document.getElementById(responseDivId);
                div.style.display = 'block';
                div.innerHTML = '<div class="loading">🔄 Calling telco operations API...</div>';
    
                try {
                    const response = await fetch(`/mcp/tools/${toolName}`, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(args)
                    });
                    const result = await response.json();
                    div.innerHTML = '<pre>' + JSON.stringify(result, null, 2) + '</pre>';
                } catch (error) {
                    div.innerHTML = '<div style="color: red;">Error: ' + error.message + '</div>';
                }
            }
    
            async function listTools() {
              const div = document.getElementById('tools-response');
              div.style.display = 'block';
              div.innerHTML = '<div class="loading">🔄 Fetching available tools...</div>';
            
              try {
                const response = await fetch(`/mcp/`, {
                  method: 'POST',
                  headers: { 
                    'Content-Type': 'application/json',
                    'Accept': 'application/json, text/event-stream',
                    'MCP-Session-ID': '12345'
                  },
                  body: JSON.stringify({method: "tools/list", jsonrpc: "2.0", id: "1"})
                });
            
                if (!response.body) throw new Error("ReadableStream not supported");
            
                const reader = response.body.getReader();
                const decoder = new TextDecoder("utf-8");
                let buffer = "";
                const toolsList = [];
            
                while (true) {
                  const { done, value } = await reader.read();
                  if (done) break;
            
                  buffer += decoder.decode(value, { stream: true });
            
                  // Split events on actual newlines
                  let events = buffer.split("\\r\\n");
                  // Process all complete events except last partial one
                  for (let i = 0; i < events.length - 1; i++) {
                    const event = events[i];
                    for (const line of event.split("\\n")) {
                      if (line.startsWith("data: ")) {
                        const jsonStr = line.slice(6);
                        try {
                          const parsed = JSON.parse(jsonStr);
                          debugger;
                          if (parsed.result && parsed.result.tools) {
                            // Accumulate tools from each event
                            toolsList.push(...parsed.result.tools);
                          }
                        } catch (e) {
                          console.error("Failed to parse SSE JSON data", e);
                        }
                      }
                    }
                  }
            
                  // Keep the last partial event in the buffer for next chunk
                  buffer = events[events.length - 1];
                }
            
                if (toolsList.length > 0) {
                  // Render the combined tools list nicely
                  div.innerHTML = '<h3>Available Tools:</h3><ul>' +
                    toolsList.map(tool => `<li><strong>${tool.name}</strong>: ${tool.description}</li>`).join('') +
                    '</ul>';
                } else {
                  div.innerHTML = '<div>No tools found in response.</div>';
                }
            
              } catch (error) {
                div.innerHTML = '<div style="color: red;">Error: ' + error.message + '</div>';
              }
            }


    
            document.addEventListener("DOMContentLoaded", () => {
                // Wire up buttons with click handlers
                document.getElementById("btn-check-outage-all").addEventListener("click", () => callTool('check-outage-status'));
                document.getElementById("btn-check-outage-bay").addEventListener("click", () => callTool('check-outage-status', {region: 'Bay Area'}));
                document.getElementById("btn-metrics-all").addEventListener("click", () => callTool('get-network-metrics'));
                document.getElementById("btn-metrics-dtla").addEventListener("click", () => callTool('get-network-metrics', {region: 'Downtown LA'}));
                document.getElementById("btn-report-degraded").addEventListener("click", () => callTool('report-network-issue', { issue_type: 'degraded_performance', region: 'Bay Area', description: 'Multiple customer reports of slow data speeds during peak hours' }));
                document.getElementById("btn-list-tools").addEventListener("click", listTools);
            });
        </script>
    </body>
    </html>

    """
    return HTMLResponse(html)

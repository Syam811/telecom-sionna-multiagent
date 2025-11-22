import requests
from core.schemas import ToolResult

class MCPClient:
    """
    Thin HTTP client to your MCP server.
    Assumes you already expose endpoints like:
      /simulate_constellation
      /simulate_ber
      /simulate_ber_mimo
      /simulate_radio_map
      /simulate_multi_radio_map
    """
    def __init__(self, base_url="http://localhost:8080"):
        self.base_url = base_url.rstrip("/")

    def call_tool(self, tool_name: str, params: dict) -> ToolResult:
        url = f"{self.base_url}/{tool_name}"
        try:
            r = requests.post(url, json=params, timeout=120)
            r.raise_for_status()
            return ToolResult(ok=True, payload=r.json())
        except Exception as e:
            return ToolResult(ok=False, payload={}, error=str(e))

from core.logger import setup_logger

TASK_TO_TOOL = {
    "constellation": "simulate_constellation",
    "ber": "simulate_ber",
    "mimo_comparison": "simulate_ber_mimo",
    "radiomap": "simulate_radio_map",
    "multi_radio_map": "simulate_multi_radio_map",
}

class SimulationAgent:
    def __init__(self, mcp_client):
        self.mcp = mcp_client
        self.logger = setup_logger("SimulationAgent")

    def run(self, task_spec):
        tool_name = TASK_TO_TOOL.get(task_spec.task_type)
        task_spec.tool_name = tool_name

        self.logger.info(f"Calling tool: {tool_name} with params: {task_spec.parameters}")
        result = self.mcp.call_tool(tool_name, task_spec.parameters)

        if not result.ok:
            self.logger.error(f"Tool call failed: {result.error}")
        else:
            self.logger.info("Tool call success.")
        return task_spec, result

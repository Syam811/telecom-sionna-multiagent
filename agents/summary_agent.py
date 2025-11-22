from core.logger import setup_logger

class SummaryAgent:
    def __init__(self):
        self.logger = setup_logger("SummaryAgent")

    def run(self, task_spec, tool_result):
        if not tool_result.ok:
            return f"Simulation failed: {tool_result.error}"

        payload = tool_result.payload
        # You can enrich this based on what your tools return.
        summary = [
            f"Task type: {task_spec.task_type}",
            f"Tool used: {task_spec.tool_name}",
            "Result generated successfully.",
        ]

        if "plots" in payload:
            summary.append(f"Plots: {payload['plots']}")
        if "kpis" in payload:
            summary.append(f"KPIs: {payload['kpis']}")

        final = "\n".join(summary)
        self.logger.info("Summary ready.")
        return final

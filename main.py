from core.task_decomposer import TaskDecomposer
from core.mcp_client import MCPClient
from core.session_store import SessionStore

from agents.interpreter_agent import InterpreterAgent
from agents.parameter_extractor_agent import ParameterExtractorAgent
from agents.simulation_agent import SimulationAgent
from agents.summary_agent import SummaryAgent


class TelecomMultiAgentAssistant:
    def __init__(self, mcp_url="http://localhost:8080"):
        self.decomposer = TaskDecomposer()
        self.mcp = MCPClient(mcp_url)
        self.memory = SessionStore(maxlen=5)

        self.interpreter = InterpreterAgent(self.decomposer)
        self.extractor = ParameterExtractorAgent(self.decomposer)
        self.simulator = SimulationAgent(self.mcp)
        self.summarizer = SummaryAgent()

    def chat(self, prompt: str):
        task = self.interpreter.run(prompt)
        task = self.extractor.run(task)
        task, result = self.simulator.run(task)
        summary = self.summarizer.run(task, result)

        self.memory.add({
            "prompt": prompt,
            "task_type": task.task_type,
            "params": task.parameters,
            "tool": task.tool_name,
            "result_ok": result.ok
        })

        return summary, result.payload if result.ok else {}


if __name__ == "__main__":
    assistant = TelecomMultiAgentAssistant()
    while True:
        prompt = input("\nYou: ")
        if prompt.strip().lower() in {"exit", "quit"}:
            break
        summary, payload = assistant.chat(prompt)
        print("\nAssistant:\n", summary)

from core.schemas import TaskSpec
from core.logger import setup_logger

class InterpreterAgent:
    def __init__(self, decomposer):
        self.decomposer = decomposer
        self.logger = setup_logger("InterpreterAgent")

    def run(self, prompt: str) -> TaskSpec:
        self.logger.info(f"Input prompt: {prompt}")
        task_type = self.decomposer.classify(prompt)
        self.logger.info(f"Classified task_type: {task_type}")
        return TaskSpec(task_type=task_type, raw_prompt=prompt)

import logging

from core.schemas import TaskSpec

class InterpreterAgent:
    """
    Classifies prompt into one of:
      constellation, ber, mimo_comparison, radiomap, multi_radio_map
    Must accept decomposer because main.py passes it.
    """
    def __init__(self, decomposer):
        self.decomposer = decomposer
        self.logger = logging.getLogger("InterpreterAgent")

    def run(self, prompt: str) -> TaskSpec:
        prompt_l = prompt.lower()
        self.logger.info(f"Input prompt: {prompt}")

        # ---- MIMO DETECTION ----
        mimo_keywords = ["mimo", "1x1", "2x2", "4x4", "8x8", "antenna", "antennas"]
        if any(k in prompt_l for k in mimo_keywords):
            task_type = "mimo_comparison"
            self.logger.info(f"Classified task_type: {task_type}")
            return TaskSpec(task_type=task_type, raw_prompt=prompt)

        # ---- MULTI RADIO MAP ----
        if ("multi" in prompt_l and "radio map" in prompt_l) \
           or ("multi-transmitter" in prompt_l) \
           or ("multiple tx" in prompt_l) \
           or ("many transmitters" in prompt_l):
            task_type = "multi_radio_map"
            self.logger.info(f"Classified task_type: {task_type}")
            return TaskSpec(task_type=task_type, raw_prompt=prompt)

        # ---- SINGLE RADIO MAP ----
        if "radio map" in prompt_l or "heatmap" in prompt_l or "coverage" in prompt_l:
            task_type = "radiomap"
            self.logger.info(f"Classified task_type: {task_type}")
            return TaskSpec(task_type=task_type, raw_prompt=prompt)

        # ---- BER ----
        ber_keywords = ["ber", "bit error", "error rate"]
        if any(k in prompt_l for k in ber_keywords):
            task_type = "ber"
            self.logger.info(f"Classified task_type: {task_type}")
            return TaskSpec(task_type=task_type, raw_prompt=prompt)

        # ---- CONSTELLATION ----
        const_keywords = ["constellation", "scatter", "symbol plot", "iq plot"]
        if any(k in prompt_l for k in const_keywords):
            task_type = "constellation"
            self.logger.info(f"Classified task_type: {task_type}")
            return TaskSpec(task_type=task_type, raw_prompt=prompt)

        # ---- FALLBACK ----
        task_type = "constellation"
        self.logger.info(f"Could not classify → defaulting to {task_type}")
        return TaskSpec(task_type=task_type, raw_prompt=prompt)

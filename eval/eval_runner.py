import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import json
import logging
from main import TelecomMultiAgentAssistant


def run_eval(path="eval/sample_tasks.json"):
    # ---- Silence INFO logs during evaluation ----
    logging.getLogger().setLevel(logging.WARNING)
    logging.getLogger("InterpreterAgent").setLevel(logging.WARNING)
    logging.getLogger("ParameterExtractorAgent").setLevel(logging.WARNING)

    assistant = TelecomMultiAgentAssistant()
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    correct_task = 0
    correct_tool = 0

    tool_map = {
        "constellation": "simulate_constellation",
        "ber": "simulate_ber",
        "mimo_comparison": "simulate_ber_mimo",
        "radiomap": "simulate_radio_map",
        "multi_radio_map": "simulate_multi_radio_map"
    }

    print("\n--- Running Evaluation (clean summary) ---\n")

    for t in data:
        prompt = t["prompt"]

        expected_task = t["expected_task_type"]
        expected_tool = t["expected_tool"]

        # 1. Interpret task type
        task = assistant.interpreter.run(prompt)

        # 2. Extract parameters
        task = assistant.extractor.run(task)

        predicted_task = task.task_type
        predicted_tool = tool_map.get(predicted_task, "UNKNOWN")

        ok_task = predicted_task == expected_task
        ok_tool = predicted_tool == expected_tool

        if ok_task:
            correct_task += 1
        if ok_tool:
            correct_tool += 1

        # ---- Print per-task summary ----
        print(f"[{t['id']}] {prompt}")
        print(f"   Expected task : {expected_task}")
        print(f"   Predicted task: {predicted_task}   -> {'OK' if ok_task else 'FAIL'}")
        print(f"   Expected tool : {expected_tool}")
        print(f"   Predicted tool: {predicted_tool}   -> {'OK' if ok_tool else 'FAIL'}")
        print(f"   Extracted params: {task.parameters}\n")

    print("=== Final Scores ===")
    print(f"Task-type accuracy : {correct_task}/{len(data)}")
    print(f"Tool-choice accuracy: {correct_tool}/{len(data)}\n")


if __name__ == "__main__":
    run_eval()

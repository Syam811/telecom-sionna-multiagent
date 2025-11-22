import json
from telecom_sionna_multiagent.main import TelecomMultiAgentAssistant

def run_eval(path="eval/sample_tasks.json"):
    assistant = TelecomMultiAgentAssistant()
    data = json.load(open(path))

    correct_task = 0
    correct_tool = 0

    tool_map = {
        "constellation": "simulate_constellation",
        "ber": "simulate_ber",
        "mimo_comparison": "simulate_ber_mimo",
        "radiomap": "simulate_radio_map",
        "multi_radio_map": "simulate_multi_radio_map"
    }

    for t in data:
        prompt = t["prompt"]

        # 1. Interpret task type
        task = assistant.interpreter.run(prompt)

        # 2. Extract parameters
        task = assistant.extractor.run(task)

        expected_task = t["expected_task_type"]
        expected_tool = t["expected_tool"]

        # Accuracy counts
        if task.task_type == expected_task:
            correct_task += 1

        actual_tool = tool_map.get(task.task_type)
        if actual_tool == expected_tool:
            correct_tool += 1

    print(f"Task-type accuracy: {correct_task}/{len(data)}")
    print(f"Tool-choice accuracy: {correct_tool}/{len(data)}")

if __name__ == "__main__":
    run_eval()

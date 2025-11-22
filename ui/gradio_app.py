import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import gradio as gr
from main import TelecomMultiAgentAssistant

assistant = TelecomMultiAgentAssistant()

def run_agent(prompt):
    summary, payload = assistant.chat(prompt)
    plots = payload.get("plots", [])
    return summary, plots

with gr.Blocks() as demo:
    gr.Markdown("# Multi-Agent Telecom Simulation Assistant (Sionna + MCP)")
    inp = gr.Textbox(label="Enter telecom simulation request")
    out_summary = gr.Textbox(label="Agent Summary")
    out_gallery = gr.Gallery(label="Plots", columns=2)
    btn = gr.Button("Run")

    btn.click(run_agent, inp, [out_summary, out_gallery])

demo.launch()

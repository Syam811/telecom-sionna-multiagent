# telecom-sionna-multiagent

# Multi-Agent Telecom Simulation Assistant  
### Natural-language → Constellation, BER, MIMO, and Radio-Map Simulations via MCP Tools

This project was developed for the **Google AI Agents Intensive – Capstone Project (2025)** under the **Enterprise Agents Track**.  
It demonstrates how a **multi-agent architecture** can translate **natural-language telecom simulation requests** into **structured tasks**, run **Sionna-based simulations** via **MCP tools**, and generate **plots + summaries** automatically.

---

# Project Goal

Telecom simulations (constellation diagrams, BER curves, MIMO performance, and radio maps) normally require manual configuration, code editing, and Sionna API knowledge.  

This assistant automates all of that.

A user simply provides a prompt like:

> “Compare BER for QPSK in AWGN and Rayleigh from −5 to 15 dB”

Or:

> “Generate a multi-transmitter 3.5 GHz radio map for TX at (0,0,10), (60,0,10), and (–60,0,10).”

The system:
1. Interprets the task  
2. Extracts needed parameters  
3. Selects the correct MCP tool  
4. Runs the simulation  
5. Returns plots + KPIs in a summary  

No manual coding. No Sionna knowledge required.

---

# Key Features (meets Capstone requirements)

### Multi-Agent System
- InterpreterAgent — understands the user's natural language  
- ParameterExtractorAgent — converts NL → structured JSON task  
- SimulationAgent — calls the correct MCP tool  
- SummaryAgent — produces human-readable results  

### MCP Tool Integration
Custom tools built under `tools/`:
- `simulate_constellation.py`  
- `simulate_ber.py`  
- `simulate_ber_mimo.py`  
- `simulate_radio_map.py`  
- `simulate_multi_radio_map.py`

### Sessions & State
- InMemorySessionService  
- Context-preserving multi-turn conversations  

### Observability
- Logging for each agent step  
- Print-level tracing for debugging and evaluation  

### Agent Evaluation
- Synthetic dataset (16 tasks)
- Automated evaluator in `eval/eval_runner.py`

### Accuracy

(.venv) PS C:\Users\evemsya\telecom_sionna_multiagent> py eval/eval_runner.py
 
--- Running Evaluation (clean summary) ---
 
[T1] Show constellation for 16-QAM at SNR 15 dB
   Expected task : constellation
   Predicted task: constellation   -> OK
   Expected tool : simulate_constellation
   Predicted tool: simulate_constellation   -> OK
   Extracted params: {'modulation': 'qpsk', 'snr_db': 15.0}
 
[T2] Plot QPSK constellation at SNR 5 dB
   Expected task : constellation
   Predicted task: constellation   -> OK
   Expected tool : simulate_constellation
   Predicted tool: simulate_constellation   -> OK
   Extracted params: {'modulation': 'qpsk', 'snr_db': 5.0}
 
[T3] Generate constellation diagram for 64-QAM at SNR -5 dB
   Expected task : constellation
   Predicted task: constellation   -> OK
   Expected tool : simulate_constellation
   Predicted tool: simulate_constellation   -> OK
   Extracted params: {'modulation': 'qpsk', 'snr_db': -5.0}
 
[T4] Show ideal 256-QAM constellation with SNR 20 dB
   Expected task : constellation
   Predicted task: constellation   -> OK
   Expected tool : simulate_constellation
   Predicted tool: simulate_constellation   -> OK
   Extracted params: {'modulation': 'qpsk', 'snr_db': 20.0}
 
[S1] Compute BER curve for QPSK in AWGN from -5 to 15 dB
   Expected task : ber
   Predicted task: ber   -> OK
   Expected tool : simulate_ber
   Predicted tool: simulate_ber   -> OK
   Extracted params: {'modulation': 'qpsk', 'channel': 'awgn', 'snr_db_list': [-5, 0, 5, 10, 15]}
 
[S2] BER vs SNR for 64-QAM under Rayleigh fading with SNR values -5, 0, 5, 10, 15
   Expected task : ber
   Predicted task: ber   -> OK
   Expected tool : simulate_ber
   Predicted tool: simulate_ber   -> OK
   Extracted params: {'modulation': 'qpsk', 'channel': 'rayleigh', 'snr_db_list': [64, -5, 0, 5, 10, 15]}
 
[S3] Simulate BER for 16-QAM in AWGN at SNR 0 to 20 dB
   Expected task : ber
   Predicted task: ber   -> OK
   Expected tool : simulate_ber
   Predicted tool: simulate_ber   -> OK
   Extracted params: {'modulation': 'qpsk', 'channel': 'awgn', 'snr_db_list': [16, 0, 20]}
 
[S4] Compare BER performance of QPSK in Rayleigh fading channel from -5 to 15 dB
   Expected task : ber
   Predicted task: ber   -> OK
   Expected tool : simulate_ber
   Predicted tool: simulate_ber   -> OK
   Extracted params: {'modulation': 'qpsk', 'channel': 'rayleigh', 'snr_db_list': [-5, 0, 5, 10, 15]}
 
[S5] Generate BER curve for 256-QAM in AWGN for SNR -5 to 25 dB
   Expected task : ber
   Predicted task: ber   -> OK
   Expected tool : simulate_ber
   Predicted tool: simulate_ber   -> OK
   Extracted params: {'modulation': 'qpsk', 'channel': 'awgn', 'snr_db_list': [256, -5, 25]}
 
[M1] Compare BER of 64-QAM for 1x1 vs 4x4 MIMO over Rayleigh fading from -5 to 15 dB
   Expected task : mimo_comparison
   Predicted task: mimo_comparison   -> OK
   Expected tool : simulate_ber_mimo
   Predicted tool: simulate_ber_mimo   -> OK
   Extracted params: {'modulation': 'qpsk', 'snr_db_list': [-5, 0, 5, 10, 15], 'configs': [{'nt': 1, 'nr': 1}, {'nt': 4, 'nr': 4}]}
 
[M2] Run MIMO BER comparison for 16-QAM with configs 1x1, 2x2, and 4x4
   Expected task : mimo_comparison
   Predicted task: mimo_comparison   -> OK
   Expected tool : simulate_ber_mimo
   Predicted tool: simulate_ber_mimo   -> OK
   Extracted params: {'modulation': 'qpsk', 'snr_db_list': [16, 1, 1, 2, 2, 4, 4], 'configs': [{'nt': 1, 'nr': 1}, {'nt': 2, 'nr': 2}, {'nt': 4, 'nr': 4}]}
 
[M3] How does BER change when increasing antennas from 1x1 to 8x8 using 64-QAM?
   Expected task : mimo_comparison
   Predicted task: mimo_comparison   -> OK
   Expected tool : simulate_ber_mimo
   Predicted tool: simulate_ber_mimo   -> OK
   Extracted params: {'modulation': 'qpsk', 'snr_db_list': [1, 1, 8, 8, 64], 'configs': [{'nt': 1, 'nr': 1}, {'nt': 8, 'nr': 8}]}
 
[R1] Generate a radio map for a transmitter at (0,0,10) in a 200m x 200m area
   Expected task : radiomap
   Predicted task: radiomap   -> OK
   Expected tool : simulate_radio_map
   Predicted tool: simulate_radio_map   -> OK
   Extracted params: {'tx_pos': [0.0, 0.0, 10.0]}
 
[R2] Coverage heatmap with transmitter at x=50, y=0, z=10 at 3.5 GHz
   Expected task : radiomap
   Predicted task: radiomap   -> OK
   Expected tool : simulate_radio_map
   Predicted tool: simulate_radio_map   -> OK
   Extracted params: {'tx_pos': [0, 0, 10]}
 
[MR1] Generate multi-transmitter radio map with TX at (0,0,10), (60,0,10), and (-60,0,10), using strongest-signal combination
   Expected task : multi_radio_map
   Predicted task: multi_radio_map   -> OK
   Expected tool : simulate_multi_radio_map
   Predicted tool: simulate_multi_radio_map   -> OK
   Extracted params: {'tx_positions': [[0.0, 0.0, 10.0], [60.0, 0.0, 10.0], [-60.0, 0.0, 10.0]], 'combine_mode': 'max'}
 
[MR2] Create a multi TX radio map for transmitters at (0,0,10) and (80,0,10) and combine powers by summing
   Expected task : multi_radio_map
   Predicted task: multi_radio_map   -> OK
   Expected tool : simulate_multi_radio_map
   Predicted tool: simulate_multi_radio_map   -> OK
   Extracted params: {'tx_positions': [[0.0, 0.0, 10.0], [80.0, 0.0, 10.0]], 'combine_mode': 'sum'}
 
=== Final Scores ===
Task-type accuracy : 16/16
Tool-choice accuracy: 16/16
 
(.venv) PS C:\Users\evemsya\telecom_sionna_multiagent>

---

# Repository Structure

telecom-sionna-multiagent/
│

├── agents/

│   ├── interpreter_agent.py

│   ├── parameter_extractor_agent.py

│   ├── simulation_agent.py

│   └── summary_agent.py

│

├── core/

│   ├── local_tools.py               # optional older tools

│   ├── local_tools_unused.py        # unused (safe to delete)

│   ├── logger.py                    # logging helper

│   ├── mcp_client.py                # MCP server client wrapper

│   ├── schemas.py                   # TaskSpec + schema definitions

│   ├── session_store.py             # InMemorySessionService

│   ├── sionna_compat.py             # optional compatibility utilities

│   └── task_decomposer.py           # natural-language → task type logic

│

├── eval/

│   ├── eval_runner.py               # automated evaluation script

│   └── sample_tasks.json            # 16 synthetic tasks (trivial/simple/medium)

│

├── tools/                           # MCP simulation tools

│   ├── simulate_ber.py

│   ├── simulate_ber_mimo.py

│   ├── simulate_constellation.py

│   ├── simulate_multi_radio_map.py

│   └── simulate_radio_map.py

│

├── ui/

│   ├── gradio_app.py                # web UI (optional)

│   ├── README.md                    # UI instructions only

│   ├── __init__.py

│   └── gradio_app_results.pdf               # UI screenshot document

|   └── outputs.pdf               # screenshot document

│

├── main.py                          # Entry point for core agent pipeline

├── requirements.txt

└── README.md                        # Main project description, architecture, evaluation



---

# 🧠 Architecture Diagram

User Prompt

│

▼

[Interpreter Agent]

→ identifies task type (constellation, BER, MIMO, radio map)

│

▼

[Parameter Extractor Agent]

→ extracts modulation, SNR list, antennas, tx positions, etc.

│

▼

[Simulation Agent]

→ maps task_type → MCP tool

→ executes Sionna simulation

→ returns plots + KPIs

│

▼

[Summary Agent]

→ natural language explanation of results

yaml

Copy code

---

# How to Run Locally

### 1. Clone the repo

git clone https://github.com/<YOUR_USERNAME>/telecom-sionna-multiagent.git
cd telecom-sionna-multiagent

### 2. Create virtual environment
bash
Copy code
py -m venv .venv
.\.venv\Scripts\activate

### 3. Install dependencies
bash
Copy code
py -m pip install --upgrade pip
py -m pip install -r requirements.txt
(the sionna may give trouble around pillow try to uninstall and install)
py eval/eval_runner.py
pip install "pillow<12.0"

### 4. Run the multi-agent system
bash
Copy code
py main.py
Example prompts:

Show constellation for 16-QAM at 15 dB

Compute BER for QPSK in AWGN from -5 to 15 dB

Compare 1x1 vs 4x4 MIMO with 64-QAM

Multi-TX radio map at (0,0,10),(60,0,10),(−60,0,10)

### 5. Run Gradio UI

Copy code
py ui/gradio_app.py


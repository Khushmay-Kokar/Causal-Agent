# Causal-Agent
both synchronus and non synchronus version of the causal agent
- **the main.py and async.py are the same but one main.py does not has asynchronous in it while async is asynchtonus so use main.py for the normal one **
This project is an **async-first causal analysis engine** that:
- Accepts structured data and a natural language question.
- Performs CATE/ATE estimation, root cause analysis, and counterfactuals.
- Generates visual trees and a business-friendly PDF report.

the langraph causal model notebook contains the test notebook for the causal model with langraph flow as shown
<img width="498" height="1277" alt="image" src="https://github.com/user-attachments/assets/1ef5cbb9-697f-4ae6-9c3a-7dee4c2dd8ba" />

---
# Instructions
## 🧰 Project Structure


```
.
├── async.py                # code with asynchronous aspects
├── sample.csv              # csv to use for analysis(later we will get it via the chatbot)
├── .env                    # API keys, environment vars
├── pyproject.toml          # Project dependencies
├── main.py          	      # main code with no asynchronous aspects
├── reports/                # Auto-generated PDFs and JSON
└── README.md               # You are here
```

---

## 🚀 How to Set Up

You can use either `uv` (fast dependency manager) or `pip`.

---

## ✅ Setup using `uv` (recommended)

### 1. Install `uv`

```bash
curl -Ls https://astral.sh/uv/install.sh | sh
```
[Official docs → https://docs.astral.sh/uv/]

---

### 2. Create virtual environment

```bash
uv venv
```

Activate it:
- **Linux/macOS:** `source .venv/bin/activate`
- **Windows:** `.\.venv\Scripts ctivate`

---

### 3. Install dependencies

```bash
uv pip install -r pyproject.toml
and ensure pythin version is 3.10 or less
or uv add .
```

---

## Setup using `pip` (standard way)

### 1. Create virtual environment

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

---

### 2. Install dependencies

```bash
pip install -r requirements.txt
and ensure pythin version is 3.10 or less
```

---

## ⚙️ Required Environment Variables

Create a `.env` file in root directory:

```bash
OPENAI_API_KEY=your-openai-key
```

---

## Run the main.py  for causal Engine
- uv run main.py




---

## 📤 Output

After a run, you will get:

- A full JSON result (parsed from LangGraph final state)
- A Markdown + PDF report in `/reports`
- Visual trees (CATE, Policy)

---

## 🔄 Agent Integration (MCP)

You can now integrate this with your **MCP server**. The causal model can:

- Accept JSON input (`{ df, question }`)
- Return async output (`{ result, explanation, report, visuals }`)
- Plug into your existing chatbot agent

---



# Causal-Agent
both synchronus and non synchronus version of the causal agent

This project is an **async-first causal analysis engine** that:
- Accepts structured data and a natural language question.
- Performs CATE/ATE estimation, root cause analysis, and counterfactuals.
- Generates visual trees and a business-friendly PDF report.
---
# Instructions
## 🧰 Project Structure

```
.
├── async.py                # code with asynchronous aspects
├── .env                    # API keys, environment vars
├── pyproject.toml          # Project dependencies
├── main.py          	    # main code with no asynchronous aspects
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
or uv add pyproject.toml
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
pip install -r pyproject.toml
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



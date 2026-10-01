# Browser Agent Evals

A Python 3.11 evaluation suite and framework for browser-based AI agents.

## Project Structure

```text
browser-agent-evals/
  agent/
    __init__.py
    browser.py
    llm.py
    guardrails.py
    agent.py
  evals/
    tests.json
    checks.py
    run_evals.py
  traces/            (gitignored)
  .env.example
  .gitignore
  requirements.txt
  README.md
```

## Setup

Follow the exact commands below to set up your environment:

### 1. Create and Activate Virtual Environment

```bash
python3.11 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
playwright install chromium
```

### 3. Configure Environment Variables

Copy the example environment file and configure your API key and model:

```bash
cp .env.example .env
```

Edit `.env` with your credentials and configuration:

```env
ANTHROPIC_API_KEY=your_anthropic_api_key_here
MODEL_NAME=claude-3-5-sonnet-20241022
```

> **Note**: `MODEL_NAME` is fully configurable and should never be hardcoded in application logic.

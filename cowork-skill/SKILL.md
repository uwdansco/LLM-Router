# LLM Router — Cowork Skill

## What This Skill Does
When the user asks a question or gives a task and wants it routed to the best AI model, use this skill to run the LLM router script. It automatically selects between Claude, GPT-4o, Gemini, and Perplexity.

## Trigger Conditions
Use this skill when the user says things like:
- "Route this to the best AI…"
- "Use the LLM router for…"
- "Which AI should answer: …"
- "Ask the best model: …"
- Or when the user just asks a question and wants optimal model selection

## Setup Requirements
Before using, ensure:
1. The `llm-router` project is installed (requirements.txt installed)
2. A `.env` file exists at the project root with all 4 API keys
3. Python is available in the environment

## How to Use This Skill

### Step 1 — Identify the user's question
Extract the exact question or task from the user's message.

### Step 2 — Run the router script
Execute via Bash:
```bash
cd /path/to/llm-router && python cowork-skill/router_cli.py "USER_QUESTION_HERE"
```

To force a specific LLM (if user requests):
```bash
python cowork-skill/router_cli.py "USER_QUESTION_HERE" --llm claude
python cowork-skill/router_cli.py "USER_QUESTION_HERE" --llm gpt4
python cowork-skill/router_cli.py "USER_QUESTION_HERE" --llm gemini
python cowork-skill/router_cli.py "USER_QUESTION_HERE" --llm perplexity
```

### Step 3 — Parse stderr for routing info
The script prints routing info to stderr (model chosen + reason) and the answer to stdout. Display both to the user.

### Step 4 — Present the answer
Format your response as:
```
🤖 Answered by: [Model Name]
📌 Why: [routing reason]

[answer]
```

## Routing Logic Reference
- **Perplexity** → current events, news, real-time data, live prices, "recently"
- **Gemini** → images, video, very long docs, Google Workspace
- **GPT-4o** → creative writing, fiction, storytelling, poetry, casual chat
- **Claude** → code, reasoning, analysis, research, technical work (default)

## Error Handling
If the script fails, fall back to answering the question directly as Claude and note that the router was unavailable.

## Web App Alternative
If the user wants the chat interface instead of CLI, remind them to start the server:
```bash
cd /path/to/llm-router && uvicorn app:app --reload
```
Then open http://localhost:8000 in their browser.

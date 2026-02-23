# 🔀 LLM Router

Automatically routes your questions and tasks to the best AI model — Claude, GPT-4o, Gemini, or Perplexity — based on what you're asking.

---

## How It Works

1. You send a question
2. **Claude Haiku** (fast + cheap) reads the question and decides which LLM to use
3. The chosen LLM answers your question
4. You see the answer along with which model was used and why

**Routing logic:**
| Model | Best For |
|-------|----------|
| Perplexity | Current events, live data, news, real-time prices, citations |
| Gemini | Images/video, very long documents, Google Workspace |
| GPT-4o | Creative writing, fiction, storytelling, casual conversation |
| Claude | Code, reasoning, analysis, research, technical tasks |

---

## Setup (One Time)

### Prerequisites
- Python 3.10 or higher
- API keys for all 4 services (links below)

### Step 1 — Download / move the project
Place the `llm-router` folder wherever you want to keep it (e.g., `~/projects/llm-router`).

### Step 2 — Create a virtual environment
```bash
cd llm-router
python3 -m venv venv
source venv/bin/activate        # Mac/Linux
# OR
venv\Scripts\activate           # Windows
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Add your API keys
```bash
cp .env.example .env
```

Open `.env` in any text editor and fill in your keys:

| Key | Where to get it |
|-----|----------------|
| `ANTHROPIC_API_KEY` | https://console.anthropic.com/settings/keys |
| `OPENAI_API_KEY` | https://platform.openai.com/api-keys |
| `GOOGLE_API_KEY` | https://aistudio.google.com/app/apikey |
| `PERPLEXITY_API_KEY` | https://www.perplexity.ai/settings/api |

---

## Running the Web App

```bash
# Make sure you're in the llm-router directory with venv activated
python app.py
```

Then open your browser to: **http://localhost:8000**

You'll see a chat interface where you can:
- Type any question and let the router pick the best model
- Or manually override which model to use via the buttons at the top

To stop the server: press `Ctrl+C` in the terminal.

---

## Using from the Command Line

```bash
# Auto-route
python cowork-skill/router_cli.py "What is the current price of Bitcoin?"

# Force a specific model
python cowork-skill/router_cli.py "Write a haiku about coffee" --llm gpt4
python cowork-skill/router_cli.py "Explain async/await in Python" --llm claude
```

---

## Using from Claude Cowork

1. Copy the `cowork-skill/SKILL.md` file into your Cowork skills directory:
   - Usually at: `~/.skills/skills/llm-router/SKILL.md`
   - Or wherever your Cowork plugin folder is
2. Copy `cowork-skill/router_cli.py` to the same folder
3. In Cowork, just ask Claude to route your question and it will invoke the skill

---

## Optional: Run on Startup (Mac)

To have the server start automatically when you log in, create a Launch Agent:

```bash
# Create the plist file
cat > ~/Library/LaunchAgents/com.llmrouter.plist << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>com.llmrouter</string>
  <key>ProgramArguments</key>
  <array>
    <string>/path/to/llm-router/venv/bin/python</string>
    <string>/path/to/llm-router/app.py</string>
  </array>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <true/>
  <key>WorkingDirectory</key>
  <string>/path/to/llm-router</string>
</dict>
</plist>
EOF

# Replace /path/to/llm-router with your actual path, then load it:
launchctl load ~/Library/LaunchAgents/com.llmrouter.plist
```

---

## Deployment (Optional)

To make it accessible from anywhere (not just localhost), you can deploy to:

**Railway (free tier available):**
1. Push this folder to a GitHub repo
2. Go to https://railway.app → New Project → Deploy from GitHub
3. Add your env vars in Railway's Variables tab
4. Your app will get a public URL

**Render (free tier available):**
1. Same as above — connect your GitHub repo
2. Set Start Command to: `uvicorn app:app --host 0.0.0.0 --port $PORT`

---

## Project Structure

```
llm-router/
├── app.py              ← FastAPI backend (routing + LLM calls)
├── index.html          ← Chat UI served by the app
├── requirements.txt    ← Python dependencies
├── .env.example        ← Template for your API keys
├── .env                ← Your actual API keys (never commit this!)
├── README.md           ← This file
└── cowork-skill/
    ├── SKILL.md        ← Cowork skill instructions
    └── router_cli.py   ← CLI version of the router
```

---

## Costs

This router uses Claude Haiku for every routing decision (about 100 tokens = ~$0.00025 per question), then calls the chosen LLM. All four APIs have pay-as-you-go pricing — typical usage is pennies per day.

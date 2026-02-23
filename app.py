"""
Jarvis — AI voice assistant that routes questions to the best model.
Supports: Claude (Anthropic), GPT-4o (OpenAI), Gemini (Google), Perplexity
Voice output via ElevenLabs TTS.
"""

import os
import json
import requests
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import anthropic
import openai
import google.generativeai as genai

# Load from project .env first, then fall back to ~/.llm-router.env
load_dotenv()
load_dotenv(Path.home() / ".llm-router.env")

app = FastAPI(title="Jarvis", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# API KEYS (loaded from .env)
# ─────────────────────────────────────────────
ANTHROPIC_API_KEY   = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY      = os.getenv("OPENAI_API_KEY", "")
GOOGLE_API_KEY      = os.getenv("GOOGLE_API_KEY", "")
PERPLEXITY_API_KEY  = os.getenv("PERPLEXITY_API_KEY", "")
ELEVENLABS_API_KEY  = os.getenv("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "")

# ─────────────────────────────────────────────
# ROUTING LOGIC — Claude Haiku as the router
# ─────────────────────────────────────────────
ROUTING_PROMPT = """You are an expert AI model router. Analyze the question or task below and decide which AI model is best suited to handle it.

Model strengths:
- **perplexity**: Current events, breaking news, real-time web info, recent statistics, live prices, sports scores, weather, anything that happened recently or requires up-to-date sources with citations
- **gemini**: Tasks involving images or video analysis, very long documents (100k+ tokens), Google Workspace tasks, multimodal reasoning
- **gpt4**: Creative writing, fiction, storytelling, poetry, humor, casual conversation, broad pop culture knowledge, general lifestyle questions
- **claude**: Code analysis, debugging, complex multi-step reasoning, document summarization, research synthesis, nuanced writing, technical analysis, math, logic puzzles, ethical questions, tasks requiring careful thought

Rules:
1. If the question mentions "latest", "current", "today", "recently", "news", "price", "score" → lean toward perplexity
2. If the question involves a visual file or Google services → lean toward gemini
3. If the question is creative or conversational → lean toward gpt4
4. Default to claude for complex analytical tasks

Respond ONLY with valid JSON, no markdown, no explanation outside the JSON:
{{"llm": "perplexity|gemini|gpt4|claude", "reason": "one sentence explanation"}}

Question/Task: {question}"""


def route_question(question: str) -> tuple[str, str]:
    if not ANTHROPIC_API_KEY:
        return "claude", "No routing key configured — defaulting to Claude"
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    try:
        message = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=150,
            messages=[{"role": "user", "content": ROUTING_PROMPT.format(question=question)}],
        )
        raw = message.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        result = json.loads(raw)
        llm = result.get("llm", "claude")
        reason = result.get("reason", "Best overall model for this task")
        if llm not in ["perplexity", "gemini", "gpt4", "claude"]:
            llm = "claude"
        return llm, reason
    except Exception as e:
        return "claude", f"Routing failed ({e}), defaulting to Claude"


# ─────────────────────────────────────────────
# LLM CALLERS
# ─────────────────────────────────────────────

def call_claude(question: str) -> str:
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    message = client.messages.create(
        model="claude-opus-4-5-20251101",
        max_tokens=4096,
        messages=[{"role": "user", "content": question}],
    )
    return message.content[0].text


def call_gpt4(question: str) -> str:
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": question}],
        max_tokens=4096,
    )
    return response.choices[0].message.content


def call_gemini(question: str) -> str:
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel("gemini-1.5-pro")
    response = model.generate_content(question)
    return response.text


def call_perplexity(question: str) -> str:
    payload = {
        "model": "llama-3.1-sonar-large-128k-online",
        "messages": [{"role": "user", "content": question}],
        "max_tokens": 4096,
    }
    response = requests.post(
        "https://api.perplexity.ai/chat/completions",
        json=payload,
        headers={"Authorization": f"Bearer {PERPLEXITY_API_KEY}", "Content-Type": "application/json"},
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()
    answer = data["choices"][0]["message"]["content"]
    citations = data.get("citations", [])
    if citations:
        answer += "\n\nSources:\n" + "\n".join(f"[{i+1}] {url}" for i, url in enumerate(citations))
    return answer


LLM_CALLERS = {
    "claude": call_claude,
    "gpt4": call_gpt4,
    "gemini": call_gemini,
    "perplexity": call_perplexity,
}

LLM_DISPLAY_NAMES = {
    "claude": "Claude",
    "gpt4": "GPT-4o",
    "gemini": "Gemini",
    "perplexity": "Perplexity",
}

LLM_COLORS = {
    "claude": "#D97757",
    "gpt4": "#10a37f",
    "gemini": "#4285F4",
    "perplexity": "#7c3aed",
}


# ─────────────────────────────────────────────
# API ENDPOINTS
# ─────────────────────────────────────────────

@app.post("/api/ask")
async def ask(request: Request):
    body = await request.json()
    question = body.get("question", "").strip()
    override_llm = body.get("override_llm", None)

    if not question:
        return JSONResponse({"error": "No question provided"}, status_code=400)

    if override_llm and override_llm in LLM_CALLERS:
        chosen_llm = override_llm
        routing_reason = "Manually selected"
    else:
        chosen_llm, routing_reason = route_question(question)

    caller = LLM_CALLERS.get(chosen_llm, call_claude)
    try:
        answer = caller(question)
    except Exception as e:
        error_msg = str(e)
        try:
            answer = call_claude(question)
            chosen_llm = "claude"
            routing_reason = f"Fell back to Claude after error: {error_msg}"
        except Exception as fallback_error:
            return JSONResponse({"error": f"All LLMs failed: {fallback_error}"}, status_code=500)

    return JSONResponse({
        "answer": answer,
        "llm": chosen_llm,
        "llm_display": LLM_DISPLAY_NAMES[chosen_llm],
        "llm_color": LLM_COLORS[chosen_llm],
        "routing_reason": routing_reason,
    })


@app.post("/api/speak")
async def speak(request: Request):
    """Convert text to speech using ElevenLabs and stream back audio."""
    if not ELEVENLABS_API_KEY or not ELEVENLABS_VOICE_ID:
        return JSONResponse({"error": "ElevenLabs not configured"}, status_code=503)

    body = await request.json()
    text = body.get("text", "").strip()
    if not text:
        return JSONResponse({"error": "No text provided"}, status_code=400)

    # Trim very long responses for TTS — first ~800 chars is natural for speech
    if len(text) > 800:
        # Find a sentence boundary near 800 chars
        cutoff = text[:800].rfind(". ")
        text = text[:cutoff + 1] if cutoff > 400 else text[:800]

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}"
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json",
        "Accept": "audio/mpeg",
    }
    payload = {
        "text": text,
        "model_id": "eleven_turbo_v2",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.75,
            "style": 0.0,
            "use_speaker_boost": True,
        },
    }

    response = requests.post(url, json=payload, headers=headers, timeout=30)
    response.raise_for_status()

    return StreamingResponse(
        iter([response.content]),
        media_type="audio/mpeg",
        headers={"Content-Disposition": "inline; filename=jarvis.mp3"},
    )


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "api_keys_configured": {
            "anthropic":   bool(ANTHROPIC_API_KEY),
            "openai":      bool(OPENAI_API_KEY),
            "google":      bool(GOOGLE_API_KEY),
            "perplexity":  bool(PERPLEXITY_API_KEY),
            "elevenlabs":  bool(ELEVENLABS_API_KEY),
        },
    }


# ─────────────────────────────────────────────
# SERVE THE FRONTEND
# ─────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    html_path = Path(__file__).parent / "index.html"
    return HTMLResponse(content=html_path.read_text(), status_code=200)


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    print(f"\n🤖 Jarvis running at http://localhost:{port}\n")
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)

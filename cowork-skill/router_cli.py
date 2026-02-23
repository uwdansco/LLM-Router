#!/usr/bin/env python3
"""
LLM Router CLI — for use as a Cowork skill.
Usage: python router_cli.py "your question here"
       python router_cli.py "your question" --llm claude   (force a specific LLM)
"""

import sys
import os
import json
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Load .env from the parent llm-router directory
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY", "")

ROUTING_PROMPT = """You are an expert AI model router. Analyze the question or task and choose the best model.

Model strengths:
- perplexity: Current events, breaking news, real-time web info, recent prices, live data
- gemini: Image/video analysis, very long documents, Google Workspace tasks
- gpt4: Creative writing, fiction, storytelling, poetry, casual conversation
- claude: Code, complex reasoning, document analysis, research synthesis, technical work

Respond ONLY with valid JSON: {{"llm": "perplexity|gemini|gpt4|claude", "reason": "one sentence"}}

Question: {question}"""


def route(question: str) -> tuple[str, str]:
    import anthropic
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=100,
        messages=[{"role": "user", "content": ROUTING_PROMPT.format(question=question)}],
    )
    raw = msg.content[0].text.strip()
    if raw.startswith("```"):
        raw = "\n".join(raw.split("\n")[1:-1])
    result = json.loads(raw)
    return result.get("llm", "claude"), result.get("reason", "")


def call_claude(q):
    import anthropic
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    r = client.messages.create(
        model="claude-opus-4-5-20251101", max_tokens=4096,
        messages=[{"role": "user", "content": q}]
    )
    return r.content[0].text


def call_gpt4(q):
    import openai
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    r = client.chat.completions.create(
        model="gpt-4o", max_tokens=4096,
        messages=[{"role": "user", "content": q}]
    )
    return r.choices[0].message.content


def call_gemini(q):
    import google.generativeai as genai
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel("gemini-1.5-pro")
    return model.generate_content(q).text


def call_perplexity(q):
    import requests
    r = requests.post(
        "https://api.perplexity.ai/chat/completions",
        json={"model": "llama-3.1-sonar-large-128k-online", "messages": [{"role": "user", "content": q}]},
        headers={"Authorization": f"Bearer {PERPLEXITY_API_KEY}", "Content-Type": "application/json"},
        timeout=60,
    )
    r.raise_for_status()
    data = r.json()
    answer = data["choices"][0]["message"]["content"]
    citations = data.get("citations", [])
    if citations:
        answer += "\n\nSources:\n" + "\n".join(f"[{i+1}] {u}" for i, u in enumerate(citations))
    return answer


LLM_NAMES = {
    "claude": "Claude (Anthropic)",
    "gpt4": "GPT-4o (OpenAI)",
    "gemini": "Gemini 1.5 Pro (Google)",
    "perplexity": "Perplexity (Online)",
}

CALLERS = {
    "claude": call_claude,
    "gpt4": call_gpt4,
    "gemini": call_gemini,
    "perplexity": call_perplexity,
}


def main():
    parser = argparse.ArgumentParser(description="LLM Router CLI")
    parser.add_argument("question", help="Your question or task")
    parser.add_argument("--llm", choices=["claude", "gpt4", "gemini", "perplexity"],
                        help="Force a specific LLM (skips routing)")
    args = parser.parse_args()

    question = args.question

    if args.llm:
        chosen, reason = args.llm, "Manually specified"
    else:
        print("🔀 Routing...", file=sys.stderr)
        chosen, reason = route(question)

    print(f"\n{'─'*60}", file=sys.stderr)
    print(f"🤖 Routed to: {LLM_NAMES[chosen]}", file=sys.stderr)
    print(f"   Reason: {reason}", file=sys.stderr)
    print(f"{'─'*60}\n", file=sys.stderr)

    answer = CALLERS[chosen](question)
    print(answer)


if __name__ == "__main__":
    main()

"""
Jarvis — AI voice assistant that routes questions to the best model.
Supports: Claude (Anthropic), GPT-4o (OpenAI), Gemini (Google), Perplexity
Voice output via ElevenLabs TTS.
Google Calendar + Gmail integration.
"""

import os
import json
import base64
import requests
from pathlib import Path
from datetime import datetime, timedelta, timezone
from email.mime.text import MIMEText
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

app = FastAPI(title="Jarvis", version="3.0.0")

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

# Google OAuth credentials (for Calendar + Gmail)
GOOGLE_CLIENT_ID     = os.getenv("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "")
GOOGLE_REFRESH_TOKEN = os.getenv("GOOGLE_REFRESH_TOKEN", "")

# TenantStack Blog API
TENANTSTACK_BLOG_API_KEY = os.getenv("TENANTSTACK_BLOG_API_KEY", "")
TENANTSTACK_BLOG_URL     = "https://tsikzygmwawvxheisdhc.supabase.co/functions/v1/blog-api"

# PhysicianPad Blog API
PHYSICIANPAD_BLOG_API_KEY = os.getenv("PHYSICIANPAD_BLOG_API_KEY", "")
PHYSICIANPAD_BLOG_URL     = "https://blog.physicianpad.com/api/admin/posts"


# ─────────────────────────────────────────────
# ROUTING LOGIC — Claude Haiku as the router
# ─────────────────────────────────────────────
ROUTING_PROMPT = """You are an expert AI model router. Analyze the question or task below and decide which AI model or action is best suited to handle it.

Options:
- **perplexity**: Current events, breaking news, real-time web info, recent statistics, live prices, sports scores, weather, anything that happened recently or requires up-to-date sources with citations
- **gemini**: Tasks involving images or video analysis, very long documents (100k+ tokens), Google Workspace tasks, multimodal reasoning
- **gpt4**: Creative writing, fiction, storytelling, poetry, humor, casual conversation, broad pop culture knowledge, general lifestyle questions
- **claude**: Code analysis, debugging, complex multi-step reasoning, document summarization, research synthesis, nuanced writing, technical analysis, math, logic puzzles, ethical questions, tasks requiring careful thought
- **calendar_read**: User wants to check their schedule, view upcoming events, or know what's on their calendar (e.g. "what's on my calendar", "what do I have today/this week")
- **calendar_create**: User wants to create, schedule, or add a new event, meeting, or appointment (e.g. "schedule a meeting", "create an event", "add to my calendar")
- **email_read**: User wants to read, check, or browse their emails or inbox (e.g. "check my email", "any new emails", "what's in my inbox")
- **email_send**: User wants to compose and send a new email to someone (e.g. "send an email to X", "email John about Y")
- **email_reply**: User wants to reply to an existing email (e.g. "reply to Sarah's email", "respond to the email about X")
- **blog_write**: User wants to write AND publish (or draft) a blog post for TenantStack (e.g. "write a blog post about X", "post to TenantStack about Y", "draft a blog post on Z")
- **blog_list**: User wants to see recent blog posts on TenantStack (e.g. "show me recent TenantStack posts", "list my blog posts")
- **physicianpad_write**: User wants to write AND publish (or draft) a blog post for PhysicianPad (e.g. "write a PhysicianPad post about X", "post to PhysicianPad about Y", "blog post for physicians about Z")
- **physicianpad_list**: User wants to see recent PhysicianPad blog posts (e.g. "show PhysicianPad posts", "list PhysicianPad blog")

Rules:
1. If the question mentions "latest", "current", "today", "recently", "news", "price", "score" → lean toward perplexity
2. If the question involves a visual file or Google services → lean toward gemini
3. If the question is creative or conversational → lean toward gpt4
4. If the question involves calendar, schedule, events, meetings → calendar_read or calendar_create
5. If the question involves email, inbox, sending messages → email_read, email_send, or email_reply
6. If the question involves writing or posting a blog post for TenantStack → blog_write or blog_list
7. If the question involves writing or posting a blog post for PhysicianPad, physicians, medical scribing → physicianpad_write or physicianpad_list
8. Default to claude for complex analytical tasks

Respond ONLY with valid JSON, no markdown, no explanation outside the JSON:
{{"llm": "perplexity|gemini|gpt4|claude|calendar_read|calendar_create|email_read|email_send|email_reply|blog_write|blog_list|physicianpad_write|physicianpad_list", "reason": "one sentence explanation"}}

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
        valid = ["perplexity", "gemini", "gpt4", "claude",
                 "calendar_read", "calendar_create",
                 "email_read", "email_send", "email_reply",
                 "blog_write", "blog_list",
                 "physicianpad_write", "physicianpad_list"]
        if llm not in valid:
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
    "calendar_read": "Calendar",
    "calendar_create": "Calendar",
    "email_read": "Gmail",
    "email_send": "Gmail",
    "email_reply": "Gmail",
}

LLM_COLORS = {
    "claude": "#D97757",
    "gpt4": "#10a37f",
    "gemini": "#4285F4",
    "perplexity": "#7c3aed",
    "calendar_read": "#0F9D58",
    "calendar_create": "#0F9D58",
    "email_read": "#EA4335",
    "email_send": "#EA4335",
    "email_reply": "#EA4335",
    "blog_write": "#6366f1",
    "blog_list":  "#6366f1",
}

LLM_DISPLAY_NAMES.update({
    "blog_write":        "TenantStack Blog",
    "blog_list":         "TenantStack Blog",
    "physicianpad_write": "PhysicianPad Blog",
    "physicianpad_list":  "PhysicianPad Blog",
})

LLM_COLORS.update({
    "physicianpad_write": "#0ea5e9",
    "physicianpad_list":  "#0ea5e9",
})


# ─────────────────────────────────────────────
# GOOGLE CREDENTIALS
# ─────────────────────────────────────────────

def get_google_credentials():
    """Build a Google Credentials object from stored refresh token."""
    if not all([GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, GOOGLE_REFRESH_TOKEN]):
        return None
    try:
        from google.oauth2.credentials import Credentials
        from google.auth.transport.requests import Request as GoogleRequest
        creds = Credentials(
            token=None,
            refresh_token=GOOGLE_REFRESH_TOKEN,
            token_uri="https://oauth2.googleapis.com/token",
            client_id=GOOGLE_CLIENT_ID,
            client_secret=GOOGLE_CLIENT_SECRET,
            scopes=[
                "https://www.googleapis.com/auth/calendar",
                "https://www.googleapis.com/auth/gmail.modify",
            ],
        )
        creds.refresh(GoogleRequest())
        return creds
    except Exception as e:
        print(f"Google credentials error: {e}")
        return None


def google_not_configured_msg() -> str:
    return (
        "Google Calendar and Gmail aren't connected yet. "
        "To set that up, run the `get_google_token.py` script included in your project, "
        "then add `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET`, and `GOOGLE_REFRESH_TOKEN` "
        "to your environment variables."
    )


# ─────────────────────────────────────────────
# CALENDAR FUNCTIONS
# ─────────────────────────────────────────────

def get_calendar_events(days: int = 7) -> list:
    """Fetch upcoming calendar events for the next N days."""
    creds = get_google_credentials()
    if not creds:
        return []
    try:
        from googleapiclient.discovery import build
        service = build("calendar", "v3", credentials=creds)
        now = datetime.now(timezone.utc)
        time_max = now + timedelta(days=days)
        result = service.events().list(
            calendarId="primary",
            timeMin=now.isoformat(),
            timeMax=time_max.isoformat(),
            maxResults=15,
            singleEvents=True,
            orderBy="startTime",
        ).execute()
        return result.get("items", [])
    except Exception as e:
        print(f"Calendar fetch error: {e}")
        return []


def create_calendar_event(summary: str, start_dt: str, end_dt: str,
                          attendees: list = None, description: str = "",
                          location: str = "") -> dict:
    """Create a new Google Calendar event."""
    creds = get_google_credentials()
    if not creds:
        raise ValueError("Google credentials not configured")
    from googleapiclient.discovery import build
    service = build("calendar", "v3", credentials=creds)
    event_body = {
        "summary": summary,
        "description": description,
        "location": location,
        "start": {"dateTime": start_dt, "timeZone": "America/New_York"},
        "end":   {"dateTime": end_dt,   "timeZone": "America/New_York"},
    }
    if attendees:
        event_body["attendees"] = [{"email": e} for e in attendees]
    send_updates = "all" if attendees else "none"
    created = service.events().insert(
        calendarId="primary", body=event_body, sendUpdates=send_updates
    ).execute()
    return created


def format_events(events: list) -> str:
    if not events:
        return "Your calendar is clear for the next 7 days. 📅"
    lines = ["Here's what's coming up:\n"]
    for e in events:
        start = e["start"].get("dateTime", e["start"].get("date", ""))
        try:
            dt = datetime.fromisoformat(start.replace("Z", "+00:00"))
            formatted = dt.strftime("%a, %b %-d at %-I:%M %p")
        except Exception:
            formatted = start
        lines.append(f"**{e.get('summary', '(no title)')}**")
        attendees = e.get("attendees", [])
        if attendees:
            names = [a.get("displayName", a.get("email", "")) for a in attendees[:3]]
            lines.append(f"🗓 {formatted}  ·  with {', '.join(names)}")
        else:
            lines.append(f"🗓 {formatted}")
        if e.get("location"):
            lines.append(f"📍 {e['location']}")
        lines.append("")
    return "\n".join(lines).strip()


def extract_event_params(question: str) -> dict:
    """Use Claude Haiku to parse natural language into structured event fields."""
    now = datetime.now()
    prompt = f"""Extract calendar event details from this request.
Today is {now.strftime('%A, %B %d, %Y')} and the current time is {now.strftime('%I:%M %p')}.

Request: {question}

Return ONLY valid JSON (no markdown fences):
{{
  "summary": "event title",
  "start_dt": "YYYY-MM-DDTHH:MM:SS",
  "end_dt": "YYYY-MM-DDTHH:MM:SS",
  "attendees": [],
  "description": "",
  "location": ""
}}

Rules:
- If no end time is specified, assume 1 hour after start.
- If no time is specified, use 9:00 AM.
- attendees should be email addresses only; omit names without emails.
- Use New York timezone context for relative times."""

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=400,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = message.content[0].text.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else raw
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


# ─────────────────────────────────────────────
# GMAIL FUNCTIONS
# ─────────────────────────────────────────────

def get_recent_emails(count: int = 5, query: str = "in:inbox is:unread") -> list:
    """Fetch recent emails from Gmail."""
    creds = get_google_credentials()
    if not creds:
        return []
    try:
        from googleapiclient.discovery import build
        service = build("gmail", "v1", credentials=creds)
        result = service.users().messages().list(
            userId="me", q=query, maxResults=count
        ).execute()
        messages = result.get("messages", [])
        details = []
        for msg in messages:
            msg_data = service.users().messages().get(
                userId="me", id=msg["id"], format="full"
            ).execute()
            headers = {h["name"]: h["value"] for h in msg_data["payload"]["headers"]}
            # Extract plain-text body
            body = ""
            payload = msg_data["payload"]
            if "parts" in payload:
                for part in payload["parts"]:
                    if part["mimeType"] == "text/plain" and "data" in part.get("body", {}):
                        body = base64.urlsafe_b64decode(
                            part["body"]["data"]
                        ).decode("utf-8", errors="replace")
                        break
            elif "body" in payload and "data" in payload["body"]:
                body = base64.urlsafe_b64decode(
                    payload["body"]["data"]
                ).decode("utf-8", errors="replace")
            details.append({
                "id": msg_data["id"],
                "threadId": msg_data["threadId"],
                "subject": headers.get("Subject", "(no subject)"),
                "from": headers.get("From", ""),
                "to": headers.get("To", ""),
                "date": headers.get("Date", ""),
                "snippet": msg_data.get("snippet", ""),
                "body": body[:2000],
            })
        return details
    except Exception as e:
        print(f"Gmail fetch error: {e}")
        return []


def send_email_message(to: str, subject: str, body: str) -> dict:
    """Send a new email via Gmail."""
    creds = get_google_credentials()
    if not creds:
        raise ValueError("Google credentials not configured")
    from googleapiclient.discovery import build
    service = build("gmail", "v1", credentials=creds)
    msg = MIMEText(body)
    msg["to"] = to
    msg["subject"] = subject
    raw = base64.urlsafe_b64encode(msg.as_bytes()).decode("utf-8")
    return service.users().messages().send(userId="me", body={"raw": raw}).execute()


def reply_to_email_message(thread_id: str, to: str, subject: str, body: str) -> dict:
    """Reply to an existing Gmail thread."""
    creds = get_google_credentials()
    if not creds:
        raise ValueError("Google credentials not configured")
    from googleapiclient.discovery import build
    service = build("gmail", "v1", credentials=creds)
    reply_subject = subject if subject.startswith("Re:") else f"Re: {subject}"
    msg = MIMEText(body)
    msg["to"] = to
    msg["subject"] = reply_subject
    raw = base64.urlsafe_b64encode(msg.as_bytes()).decode("utf-8")
    return service.users().messages().send(
        userId="me", body={"raw": raw, "threadId": thread_id}
    ).execute()


def format_emails(emails: list) -> str:
    if not emails:
        return "No unread emails in your inbox. 📬"
    lines = [f"You have **{len(emails)} unread email(s)**:\n"]
    for i, e in enumerate(emails, 1):
        lines.append(f"**{i}. {e['subject']}**")
        lines.append(f"From: {e['from']}")
        snippet = e["snippet"][:180]
        if snippet:
            lines.append(f"_{snippet}..._")
        lines.append("")
    return "\n".join(lines).strip()


def extract_email_send_params(question: str) -> dict:
    """Use Claude Haiku to parse natural language into email fields."""
    prompt = f"""Extract email details from this request.

Request: {question}

Return ONLY valid JSON (no markdown fences):
{{
  "to": "recipient@example.com",
  "subject": "email subject",
  "body": "full email body text"
}}"""
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=600,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = message.content[0].text.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else raw
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


def extract_email_reply_params(question: str, recent_emails: list) -> dict:
    """Use Claude to identify which email to reply to and draft the reply."""
    emails_summary = json.dumps([{
        "index": i,
        "id": e["id"],
        "threadId": e["threadId"],
        "from": e["from"],
        "to": e["to"],
        "subject": e["subject"],
        "snippet": e["snippet"][:150],
    } for i, e in enumerate(recent_emails)], indent=2)

    prompt = f"""Given these recent emails and the user's request, identify which email to reply to and write the reply.

Recent emails:
{emails_summary}

User request: {question}

Return ONLY valid JSON (no markdown fences):
{{
  "email_index": 0,
  "reply_body": "the full reply text to send"
}}"""
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    message = client.messages.create(
        model="claude-opus-4-5-20251101",
        max_tokens=600,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = message.content[0].text.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else raw
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


# ─────────────────────────────────────────────
# GOOGLE ACTION HANDLERS
# ─────────────────────────────────────────────

def handle_calendar_read() -> str:
    if not all([GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, GOOGLE_REFRESH_TOKEN]):
        return google_not_configured_msg()
    events = get_calendar_events(days=7)
    return format_events(events)


def handle_calendar_create(question: str) -> str:
    if not all([GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, GOOGLE_REFRESH_TOKEN]):
        return google_not_configured_msg()
    try:
        params = extract_event_params(question)
        event = create_calendar_event(
            summary=params.get("summary", "New Event"),
            start_dt=params.get("start_dt", ""),
            end_dt=params.get("end_dt", ""),
            attendees=params.get("attendees") or [],
            description=params.get("description", ""),
            location=params.get("location", ""),
        )
        start_raw = params.get("start_dt", "")
        try:
            dt = datetime.fromisoformat(start_raw)
            formatted_time = dt.strftime("%a, %b %-d at %-I:%M %p")
        except Exception:
            formatted_time = start_raw
        result = f"✅ **Event created:** {params.get('summary')}\n🗓 {formatted_time}"
        if params.get("location"):
            result += f"\n📍 {params['location']}"
        if params.get("attendees"):
            result += f"\n👥 Invites sent to: {', '.join(params['attendees'])}"
        link = event.get("htmlLink", "")
        if link:
            result += f"\n\n[View in Google Calendar]({link})"
        return result
    except Exception as e:
        return f"Sorry, I couldn't create the event: {e}"


def handle_email_read() -> str:
    if not all([GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, GOOGLE_REFRESH_TOKEN]):
        return google_not_configured_msg()
    emails = get_recent_emails(count=5)
    return format_emails(emails)


def handle_email_send(question: str) -> str:
    if not all([GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, GOOGLE_REFRESH_TOKEN]):
        return google_not_configured_msg()
    try:
        params = extract_email_send_params(question)
        send_email_message(
            to=params["to"],
            subject=params["subject"],
            body=params["body"],
        )
        return f"✅ **Email sent** to {params['to']}\n**Subject:** {params['subject']}"
    except Exception as e:
        return f"Sorry, I couldn't send the email: {e}"


def handle_email_reply(question: str) -> str:
    if not all([GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, GOOGLE_REFRESH_TOKEN]):
        return google_not_configured_msg()
    try:
        # Fetch recent emails to provide context
        emails = get_recent_emails(count=10, query="in:inbox")
        if not emails:
            return "No recent emails found to reply to."
        params = extract_email_reply_params(question, emails)
        idx = params.get("email_index", 0)
        if idx >= len(emails):
            idx = 0
        target = emails[idx]
        # Extract sender's email address (strip display name if present)
        from_raw = target["from"]
        reply_to = from_raw
        if "<" in from_raw and ">" in from_raw:
            reply_to = from_raw.split("<")[1].rstrip(">")
        reply_to_email_message(
            thread_id=target["threadId"],
            to=reply_to,
            subject=target["subject"],
            body=params["reply_body"],
        )
        return (
            f"✅ **Reply sent** to {reply_to}\n"
            f"**Re:** {target['subject']}\n\n"
            f"_{params['reply_body'][:200]}_"
        )
    except Exception as e:
        return f"Sorry, I couldn't send the reply: {e}"


# ─────────────────────────────────────────────
# TENANTSTACK BLOG FUNCTIONS
# ─────────────────────────────────────────────

TENANTSTACK_WRITER_PROMPT = """You are the lead content writer for TenantStack, a modern property management software platform.

AUDIENCE:
- Property management companies (small to enterprise)
- Real estate investors with multiple properties
- Property investment companies
- Landlords who own 2+ rental properties

TONE: Professional, engaging, and with a dry wit. You write like the smartest, most experienced person at a property management industry conference — someone who clearly knows their stuff but isn't afraid to crack a joke about a nightmare tenant story. Never stuffy. Never boring. No corporate buzzword soup.

STYLE RULES:
- Open with a hook — a surprising stat, a relatable pain point, or a sharp one-liner
- Use "you" and "your" to speak directly to the reader
- Short paragraphs (2-4 sentences max)
- Subheadings should be specific and benefit-driven (not just "Introduction")
- Include at least one piece of practical, immediately actionable advice per section
- Occasional humor is welcome but never forced
- End with a clear takeaway or call-to-action related to TenantStack

SEO — CRITICAL:
You MUST naturally integrate relevant keywords from the TARGET KEYWORDS list below into every post. Do not stuff them — weave them in where they fit naturally, like a skilled writer would. The goal is to rank on Google for these terms.

TARGET KEYWORDS (use a mix of 6-10 per post, based on the topic):

PRIMARY (highest search intent — use at least 2-3 per post):
- property management software
- landlord software
- rental property management
- tenant screening
- property management tips
- how to manage rental properties
- property management for landlords
- rent collection software

LONG-TAIL / LOW COMPETITION (use 3-5 per post — these rank faster):
- property management software for small landlords
- how to screen tenants without violating fair housing laws
- how to collect rent online as a landlord
- property management software for 1-10 units
- how to handle late rent payments
- how to raise rent without losing tenants
- how to reduce tenant turnover
- rental property accounting and expense tracking
- what software do landlords use to manage tenants
- how to manage rental properties without a property manager
- landlord tenant screening checklist
- how to handle difficult tenants legally
- property management app for independent landlords
- AI property management tools for landlords
- how to scale a rental portfolio

KEYWORD PLACEMENT RULES:
- Include the primary keyword or a close variant in the H1 title
- Use at least one long-tail keyword in an H2 subheading
- Weave keywords naturally into the first 100 words of the intro
- Use keywords in the excerpt/meta description
- Do NOT repeat the exact same phrase more than twice in one post

SEO STRUCTURE:
- Title: specific, benefit-driven, includes a target keyword, under 65 characters if possible
- Excerpt: 2-sentence meta description containing a primary keyword (under 160 chars)
- Aim for 900-1200 words of content
- Use FAQ-style subheadings when appropriate (e.g. "How do landlords screen tenants?") — these rank for featured snippets

OUTPUT FORMAT — respond ONLY with valid JSON, no markdown fences:
{
  "title": "Post title here",
  "slug": "post-title-here",
  "excerpt": "2-sentence compelling summary for previews and SEO (under 160 chars)",
  "category_slug": "one of: tips, guides, industry-news, technology, finance, tenant-management, maintenance",
  "content": "<full HTML content using <h2>, <h3>, <p>, <ul>, <li>, <strong>, <em> tags>"
}"""


def slugify(text: str) -> str:
    """Convert a title to a URL-friendly slug."""
    import re
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_-]+", "-", text)
    text = re.sub(r"^-+|-+$", "", text)
    return text[:80]


def write_blog_post(topic: str, status: str = "published") -> dict:
    """Use Claude to write a TenantStack blog post and post it via the API."""
    if not TENANTSTACK_BLOG_API_KEY:
        raise ValueError("TENANTSTACK_BLOG_API_KEY not configured")

    # Step 1 — Write the post with Claude
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    user_prompt = f"Write a blog post about: {topic}\n\nRemember to follow all tone, style, and format rules."

    message = client.messages.create(
        model="claude-opus-4-5-20251101",
        max_tokens=4000,
        system=TENANTSTACK_WRITER_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )

    raw = message.content[0].text.strip()
    # Strip any accidental markdown fences
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else raw
        if raw.startswith("json"):
            raw = raw[4:]
    post_data = json.loads(raw.strip())

    # Ensure slug is clean
    if not post_data.get("slug"):
        post_data["slug"] = slugify(post_data.get("title", topic))

    # Step 2 — Post to the TenantStack Blog API
    headers = {
        "Content-Type": "application/json",
        "x-api-key": TENANTSTACK_BLOG_API_KEY,
    }
    payload = {
        "title":         post_data["title"],
        "slug":          post_data["slug"],
        "excerpt":       post_data.get("excerpt", ""),
        "content":       post_data["content"],
        "category_slug": post_data.get("category_slug", "tips"),
        "author":        "Jarvis",
        "status":        status,
    }
    response = requests.post(TENANTSTACK_BLOG_URL, json=payload, headers=headers, timeout=30)
    response.raise_for_status()
    result = response.json()

    return {
        "title":    post_data["title"],
        "slug":     post_data["slug"],
        "excerpt":  post_data.get("excerpt", ""),
        "category": post_data.get("category_slug", "tips"),
        "status":   status,
        "api_result": result,
    }


def list_blog_posts(status: str = "published") -> list:
    """Fetch recent posts from the TenantStack Blog API."""
    if not TENANTSTACK_BLOG_API_KEY:
        return []
    headers = {"x-api-key": TENANTSTACK_BLOG_API_KEY}
    response = requests.get(
        TENANTSTACK_BLOG_URL,
        params={"status": status},
        headers=headers,
        timeout=15,
    )
    response.raise_for_status()
    data = response.json()
    # API may return a list directly or {"posts": [...]}
    if isinstance(data, list):
        return data
    return data.get("posts", data.get("data", []))


def handle_blog_write(question: str) -> str:
    """Detect draft vs publish intent, write the post, and return a confirmation."""
    if not TENANTSTACK_BLOG_API_KEY:
        return (
            "The TenantStack Blog API key isn't configured yet. "
            "Add `TENANTSTACK_BLOG_API_KEY` to your `~/.llm-router.env` file."
        )
    # Detect if user wants a draft
    q_lower = question.lower()
    status = "draft" if any(w in q_lower for w in ["draft", "save as draft", "don't publish", "do not publish"]) else "published"

    try:
        result = write_blog_post(question, status=status)
        status_label = "📝 Saved as draft" if status == "draft" else "🚀 Published live"
        return (
            f"{status_label} on **blog.tenantstack.com**\n\n"
            f"**{result['title']}**\n"
            f"_{result['excerpt']}_\n\n"
            f"Category: `{result['category']}`\n"
            f"Slug: `{result['slug']}`"
        )
    except Exception as e:
        return f"Sorry, I couldn't post to the TenantStack blog: {e}"


def handle_blog_list(_question: str) -> str:
    """Return a formatted list of recent TenantStack blog posts."""
    if not TENANTSTACK_BLOG_API_KEY:
        return "The TenantStack Blog API key isn't configured yet."
    try:
        posts = list_blog_posts()
        if not posts:
            return "No published posts found on the TenantStack blog yet."
        lines = [f"**{len(posts)} post(s) on blog.tenantstack.com:**\n"]
        for i, p in enumerate(posts[:10], 1):
            title   = p.get("title", "(no title)")
            slug    = p.get("slug", "")
            excerpt = p.get("excerpt", "")[:100]
            lines.append(f"**{i}. {title}**")
            if excerpt:
                lines.append(f"_{excerpt}_")
            if slug:
                lines.append(f"🔗 blog.tenantstack.com/{slug}")
            lines.append("")
        return "\n".join(lines).strip()
    except Exception as e:
        return f"Couldn't fetch blog posts: {e}"


# ─────────────────────────────────────────────
# PHYSICIANPAD BLOG FUNCTIONS
# ─────────────────────────────────────────────

PHYSICIANPAD_WRITER_PROMPT = """You are the lead content writer for PhysicianPad, an AI-powered medical scribing software that helps clinicians spend less time on documentation and more time with patients.

AUDIENCE:
- Primary care physicians (family medicine, internal medicine)
- Therapists and mental health counselors
- Psychiatrists
- Chiropractors
- Dentists and dental practitioners

TONE: Professional, engaging, and with a touch of dry humor. You write like a brilliant colleague who has survived the EHR wars and lived to tell the tale. You understand clinical workflows intimately — the frustration of charting at midnight, the joy of leaving the office on time for once. You're never condescending, always practical, and occasionally funny in the way only someone who's sat in a clinical setting can be. No hollow buzzwords. No preachy wellness talk.

STYLE RULES:
- Open with a hook that resonates with clinician pain (documentation burden, burnout, lost time)
- Speak directly to the reader using "you" and "your practice"
- Short paragraphs — clinicians are busy people, they skim
- Subheadings that are specific and benefit-driven
- Concrete, actionable takeaways in every section
- Occasional clinical humor is welcome (think: "Yes, SOAP notes at 11pm again")
- End with a natural call-to-action toward PhysicianPad

TOPICS TO DRAW FROM:
- AI medical scribing and ambient documentation
- Reducing physician burnout and documentation burden
- EHR efficiency tips (Epic, Athena, Cerner, etc.)
- Patient-physician interaction improvements
- Billing and coding accuracy
- Clinical workflow optimization
- Telehealth documentation
- Mental health documentation challenges
- Specialty-specific documentation tips (chiropractic SOAP notes, dental charting, therapy progress notes)

SEO — CRITICAL:
You MUST naturally integrate relevant keywords from the TARGET KEYWORDS list below into every post. Do not stuff them — weave them in where they fit naturally. The goal is to rank on Google for these specific terms, especially the long-tail specialty ones where competition is very low.

TARGET KEYWORDS (use a mix of 6-10 per post, based on the topic):

PRIMARY (highest search volume — use at least 2-3 per post):
- AI medical scribe
- AI scribe
- medical documentation software
- ambient documentation
- physician burnout
- clinical documentation
- EHR efficiency
- reduce charting time

LONG-TAIL / LOW COMPETITION — SPECIALTY (use 3-5 per post — these rank fastest):
- AI scribe for chiropractors
- AI medical scribe for therapists
- AI SOAP note generator for chiropractors
- ambient AI documentation for mental health
- AI scribe for small medical practices
- how to reduce after-hours charting
- best AI scribe for primary care physicians
- telehealth documentation tips for clinicians
- reduce physician burnout with AI documentation
- AI medical scribe for dentists
- how to speed up clinical notes
- AI scribe for therapy progress notes
- how long should clinical notes take
- medical scribing software for psychiatrists
- ambient AI vs traditional transcription

KEYWORD PLACEMENT RULES:
- Include the primary keyword or a close variant in the H1 title
- Use at least one specialty-specific long-tail keyword in an H2 subheading
- Weave keywords naturally into the first 100 words of the intro
- Use keywords in the excerpt/meta description
- Do NOT repeat the exact same phrase more than twice in one post

SEO STRUCTURE:
- Title: specific, benefit-driven, includes a target keyword, under 65 characters if possible
- Excerpt: 2-sentence meta description containing a primary keyword (under 160 chars)
- Aim for 900-1200 words of content
- Use FAQ-style subheadings when appropriate (e.g. "Does AI scribing work for chiropractors?") — these rank for featured snippets and AI search results

OUTPUT FORMAT — respond ONLY with valid JSON, no markdown fences:
{
  "title": "Post title here",
  "slug": "post-title-here",
  "excerpt": "2-sentence compelling summary (under 160 chars)",
  "category": "one of: Efficiency, AI & Technology, Burnout & Wellness, Billing & Coding, Clinical Tips, Telehealth, Mental Health, Specialty Care",
  "authorName": "Dr. Sarah Chen",
  "authorRole": "Chief Medical Officer",
  "content": "<full HTML content using <h2>, <h3>, <p>, <ul>, <li>, <strong>, <em> tags>"
}"""


def write_physicianpad_post(topic: str, status: str = "published") -> dict:
    """Use Claude to write a PhysicianPad blog post and publish via the API."""
    if not PHYSICIANPAD_BLOG_API_KEY:
        raise ValueError("PHYSICIANPAD_BLOG_API_KEY not configured")

    # Step 1 — Write the post with Claude
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    user_prompt = f"Write a blog post about: {topic}\n\nFollow all tone, style, and format rules."

    message = client.messages.create(
        model="claude-opus-4-5-20251101",
        max_tokens=4000,
        system=PHYSICIANPAD_WRITER_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )

    raw = message.content[0].text.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else raw
        if raw.startswith("json"):
            raw = raw[4:]
    post_data = json.loads(raw.strip())

    if not post_data.get("slug"):
        post_data["slug"] = slugify(post_data.get("title", topic))

    # Estimate read time (avg 200 words/min)
    word_count = len(post_data.get("content", "").split())
    read_time = f"{max(1, round(word_count / 200))} min read"

    # Step 2 — Post to PhysicianPad Blog API
    headers = {
        "Content-Type": "application/json",
        "X-Api-Key": PHYSICIANPAD_BLOG_API_KEY,
    }
    payload = {
        "title":      post_data["title"],
        "slug":       post_data["slug"],
        "excerpt":    post_data.get("excerpt", ""),
        "content":    post_data["content"],
        "category":   post_data.get("category", "Efficiency"),
        "status":     status,
        "authorName": post_data.get("authorName", "Dr. Sarah Chen"),
        "authorRole": post_data.get("authorRole", "Chief Medical Officer"),
        "readTime":   read_time,
        "featured":   False,
    }
    response = requests.post(PHYSICIANPAD_BLOG_URL, json=payload, headers=headers, timeout=30)
    response.raise_for_status()

    return {
        "title":    post_data["title"],
        "slug":     post_data["slug"],
        "excerpt":  post_data.get("excerpt", ""),
        "category": post_data.get("category", "Efficiency"),
        "readTime": read_time,
        "status":   status,
    }


def list_physicianpad_posts() -> list:
    """Fetch recent PhysicianPad blog posts."""
    if not PHYSICIANPAD_BLOG_API_KEY:
        return []
    headers = {"X-Api-Key": PHYSICIANPAD_BLOG_API_KEY}
    response = requests.get(PHYSICIANPAD_BLOG_URL, headers=headers, timeout=15)
    response.raise_for_status()
    data = response.json()
    if isinstance(data, list):
        return data
    return data.get("posts", data.get("data", []))


def handle_physicianpad_write(question: str) -> str:
    if not PHYSICIANPAD_BLOG_API_KEY:
        return "The PhysicianPad Blog API key isn't configured. Add `PHYSICIANPAD_BLOG_API_KEY` to your `~/.llm-router.env` file."
    q_lower = question.lower()
    status = "draft" if any(w in q_lower for w in ["draft", "save as draft", "don't publish", "do not publish"]) else "published"
    try:
        result = write_physicianpad_post(question, status=status)
        status_label = "📝 Saved as draft" if status == "draft" else "🚀 Published live"
        return (
            f"{status_label} on **blog.physicianpad.com**\n\n"
            f"**{result['title']}**\n"
            f"_{result['excerpt']}_\n\n"
            f"Category: `{result['category']}`  ·  {result['readTime']}\n"
            f"Slug: `{result['slug']}`"
        )
    except Exception as e:
        return f"Sorry, I couldn't post to the PhysicianPad blog: {e}"


def handle_physicianpad_list(_question: str) -> str:
    if not PHYSICIANPAD_BLOG_API_KEY:
        return "The PhysicianPad Blog API key isn't configured."
    try:
        posts = list_physicianpad_posts()
        if not posts:
            return "No posts found on the PhysicianPad blog yet."
        lines = [f"**{len(posts)} post(s) on blog.physicianpad.com:**\n"]
        for i, p in enumerate(posts[:10], 1):
            title   = p.get("title", "(no title)")
            slug    = p.get("slug", "")
            excerpt = p.get("excerpt", "")[:100]
            category = p.get("category", "")
            lines.append(f"**{i}. {title}**")
            if category:
                lines.append(f"_{category}_")
            if excerpt:
                lines.append(f"{excerpt}...")
            if slug:
                lines.append(f"🔗 blog.physicianpad.com/{slug}")
            lines.append("")
        return "\n".join(lines).strip()
    except Exception as e:
        return f"Couldn't fetch PhysicianPad posts: {e}"


GOOGLE_ACTION_HANDLERS = {
    "calendar_read":      lambda q: handle_calendar_read(),
    "calendar_create":    handle_calendar_create,
    "email_read":         lambda q: handle_email_read(),
    "email_send":         handle_email_send,
    "email_reply":        handle_email_reply,
    "blog_write":         handle_blog_write,
    "blog_list":          handle_blog_list,
    "physicianpad_write": handle_physicianpad_write,
    "physicianpad_list":  handle_physicianpad_list,
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

    if override_llm and (override_llm in LLM_CALLERS or override_llm in GOOGLE_ACTION_HANDLERS):
        chosen_llm = override_llm
        routing_reason = "Manually selected"
    else:
        chosen_llm, routing_reason = route_question(question)

    # Handle Google Calendar / Gmail actions
    if chosen_llm in GOOGLE_ACTION_HANDLERS:
        try:
            answer = GOOGLE_ACTION_HANDLERS[chosen_llm](question)
        except Exception as e:
            answer = f"Something went wrong: {e}"
        return JSONResponse({
            "answer": answer,
            "llm": chosen_llm,
            "llm_display": LLM_DISPLAY_NAMES[chosen_llm],
            "llm_color": LLM_COLORS[chosen_llm],
            "routing_reason": routing_reason,
        })

    # Handle LLM routing
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

    # Strip markdown formatting before sending to TTS
    import re
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)   # bold
    text = re.sub(r'\*(.*?)\*', r'\1', text)         # italic
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)  # links
    text = re.sub(r'#{1,6}\s', '', text)              # headers

    # Trim very long responses for TTS — ~800 chars is natural for speech
    if len(text) > 800:
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
    google_configured = all([GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, GOOGLE_REFRESH_TOKEN])
    return {
        "status": "ok",
        "api_keys_configured": {
            "anthropic":   bool(ANTHROPIC_API_KEY),
            "openai":      bool(OPENAI_API_KEY),
            "google":      bool(GOOGLE_API_KEY),
            "perplexity":  bool(PERPLEXITY_API_KEY),
            "elevenlabs":  bool(ELEVENLABS_API_KEY),
            "google_calendar_gmail": google_configured,
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

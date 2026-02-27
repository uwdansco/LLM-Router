#!/usr/bin/env python3
"""
auto_blog_post.py — Daily automated blog poster for Jarvis.

Runs Monday–Friday at 8am PST.
- Picks a fresh topic for each blog using Claude
- Writes a full blog post in each brand's voice
- Publishes directly to TenantStack and PhysicianPad

Reads API keys from ~/.llm-router.env
"""

import os
import json
import re
import sys
import requests
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
import anthropic

# ── Load environment ──────────────────────────────────────────────────────────
load_dotenv()
load_dotenv(Path.home() / ".llm-router.env")

ANTHROPIC_API_KEY         = os.getenv("ANTHROPIC_API_KEY", "")
TENANTSTACK_BLOG_API_KEY  = os.getenv("TENANTSTACK_BLOG_API_KEY", "")
TENANTSTACK_BLOG_URL      = "https://tsikzygmwawvxheisdhc.supabase.co/functions/v1/blog-api"
PHYSICIANPAD_BLOG_API_KEY = os.getenv("PHYSICIANPAD_BLOG_API_KEY", "")
PHYSICIANPAD_BLOG_URL     = "https://blog.physicianpad.com/api/admin/posts"

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
today  = datetime.now().strftime("%A, %B %d, %Y")


# ── Helpers ───────────────────────────────────────────────────────────────────

def log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_-]+", "-", text)
    text = re.sub(r"^-+|-+$", "", text)
    return text[:80]


def parse_json(raw: str) -> dict:
    """Strip markdown fences and parse JSON."""
    raw = raw.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else raw
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


# ── Topic picker ──────────────────────────────────────────────────────────────

def pick_topic(blog: str, topic_pool: list[str]) -> str:
    """Ask Claude to pick the most interesting topic for today."""
    pool_str = "\n".join(f"- {t}" for t in topic_pool)
    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=100,
        messages=[{"role": "user", "content":
            f"Today is {today}. From the list below, pick the single most timely and "
            f"interesting topic for a {blog} blog post. Return ONLY the topic text, nothing else.\n\n{pool_str}"
        }],
    )
    return msg.content[0].text.strip()


# ── TenantStack ───────────────────────────────────────────────────────────────

TENANTSTACK_TOPICS = [
    "How to screen tenants without violating fair housing laws",
    "5 lease clauses every landlord should stop ignoring",
    "The real cost of a bad tenant (and how to avoid them)",
    "How to raise rent without losing good tenants",
    "Property management software: what to look for in 2025",
    "Short-term vs long-term rentals: which is right for your portfolio",
    "How to handle maintenance requests before they become emergencies",
    "Security deposit mistakes landlords keep making",
    "Building a tenant retention strategy that actually works",
    "How to scale from 1 property to 10 without losing your mind",
    "Late rent: when to call, when to text, when to serve notice",
    "The landlord's guide to property inspections",
    "Self-managing vs hiring a property manager: a real breakdown",
    "How to write a rental listing that attracts quality tenants",
    "Eviction prevention: the early warning signs every landlord should know",
    "Tax deductions rental property owners consistently miss",
    "How AI is changing property management in 2025",
    "What tenants actually want (and why it matters to your bottom line)",
    "Rent collection best practices for multi-property landlords",
    "How to handle difficult tenants professionally and legally",
]

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

SEO:
- Naturally weave in property management keywords
- Title should be specific, benefit-driven, and under 65 characters if possible
- Aim for 900-1200 words of content

OUTPUT FORMAT — respond ONLY with valid JSON, no markdown fences:
{
  "title": "Post title here",
  "slug": "post-title-here",
  "excerpt": "2-sentence compelling summary for previews and SEO (under 160 chars)",
  "category_slug": "one of: tips, guides, industry-news, technology, finance, tenant-management, maintenance",
  "content": "<full HTML content using <h2>, <h3>, <p>, <ul>, <li>, <strong>, <em> tags>"
}"""


def post_to_tenantstack(topic: str) -> dict:
    log(f"TenantStack: writing post on '{topic}'...")
    msg = client.messages.create(
        model="claude-opus-4-5-20251101",
        max_tokens=4000,
        system=TENANTSTACK_WRITER_PROMPT,
        messages=[{"role": "user", "content":
            f"Today is {today}. Write a blog post about: {topic}\n\nFollow all tone, style, and format rules."
        }],
    )
    post = parse_json(msg.content[0].text)
    if not post.get("slug"):
        post["slug"] = slugify(post.get("title", topic))

    headers = {"Content-Type": "application/json", "x-api-key": TENANTSTACK_BLOG_API_KEY}
    payload = {
        "title":         post["title"],
        "slug":          post["slug"],
        "excerpt":       post.get("excerpt", ""),
        "content":       post["content"],
        "category_slug": post.get("category_slug", "tips"),
        "author":        "Jarvis",
        "status":        "published",
    }
    resp = requests.post(TENANTSTACK_BLOG_URL, json=payload, headers=headers, timeout=60)
    resp.raise_for_status()
    log(f"TenantStack: ✅ Published '{post['title']}'")
    return post


# ── PhysicianPad ──────────────────────────────────────────────────────────────

PHYSICIANPAD_TOPICS = [
    "How AI scribing is giving physicians 2 hours back every day",
    "The hidden cost of manual charting (it's more than you think)",
    "SOAP notes for chiropractors: a faster, smarter workflow",
    "How to cut therapy progress note time in half",
    "Dental charting in the AI era: what's changing",
    "Physician burnout and documentation: breaking the midnight charting cycle",
    "How ambient AI documentation works — and why it's different",
    "EHR efficiency tips every primary care physician needs",
    "Telehealth documentation: common mistakes and how to fix them",
    "How psychiatrists can streamline session notes without losing detail",
    "AI scribing vs traditional transcription: a real comparison",
    "The future of clinical documentation: where we're headed by 2026",
    "How to improve patient engagement by spending less time on charts",
    "Billing accuracy and AI scribing: what the data shows",
    "Documentation compliance for mental health practitioners",
    "What primary care physicians wish their EHR could do",
    "How to onboard your practice to AI scribing without disruption",
    "The ROI of AI medical scribing for small practices",
    "Reducing after-hours charting: strategies that actually work",
    "How AI is closing the documentation gap in underserved communities",
]

PHYSICIANPAD_WRITER_PROMPT = """You are the lead content writer for PhysicianPad, an AI-powered medical scribing software that helps clinicians spend less time on documentation and more time with patients.

AUDIENCE:
- Primary care physicians (family medicine, internal medicine)
- Therapists and mental health counselors
- Psychiatrists
- Chiropractors
- Dentists and dental practitioners

TONE: Professional, engaging, and with a touch of dry humor. You write like a brilliant colleague who has survived the EHR wars and lived to tell the tale. You understand clinical workflows intimately — the frustration of charting at midnight, the joy of leaving the office on time for once. Never condescending, always practical, occasionally funny in the way only someone who's sat in a clinical setting can be. No hollow buzzwords.

STYLE RULES:
- Open with a hook that resonates with clinician pain (documentation burden, burnout, lost time)
- Speak directly to the reader using "you" and "your practice"
- Short paragraphs — clinicians are busy people, they skim
- Subheadings that are specific and benefit-driven
- Concrete, actionable takeaways in every section
- Occasional clinical humor is welcome (think: "Yes, SOAP notes at 11pm again")
- End with a natural call-to-action toward PhysicianPad

SEO:
- Weave in keywords naturally: AI scribe, medical documentation, physician burnout, EHR, clinical notes
- Title under 65 characters if possible
- Aim for 900-1200 words

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


def post_to_physicianpad(topic: str) -> dict:
    log(f"PhysicianPad: writing post on '{topic}'...")
    msg = client.messages.create(
        model="claude-opus-4-5-20251101",
        max_tokens=4000,
        system=PHYSICIANPAD_WRITER_PROMPT,
        messages=[{"role": "user", "content":
            f"Today is {today}. Write a blog post about: {topic}\n\nFollow all tone, style, and format rules."
        }],
    )
    post = parse_json(msg.content[0].text)
    if not post.get("slug"):
        post["slug"] = slugify(post.get("title", topic))

    word_count = len(post.get("content", "").split())
    read_time  = f"{max(1, round(word_count / 200))} min read"

    headers = {"Content-Type": "application/json", "X-Api-Key": PHYSICIANPAD_BLOG_API_KEY}
    payload = {
        "title":      post["title"],
        "slug":       post["slug"],
        "excerpt":    post.get("excerpt", ""),
        "content":    post["content"],
        "category":   post.get("category", "Efficiency"),
        "status":     "published",
        "authorName": post.get("authorName", "Dr. Sarah Chen"),
        "authorRole": post.get("authorRole", "Chief Medical Officer"),
        "readTime":   read_time,
        "featured":   False,
    }
    resp = requests.post(PHYSICIANPAD_BLOG_URL, json=payload, headers=headers, timeout=60)
    resp.raise_for_status()
    log(f"PhysicianPad: ✅ Published '{post['title']}'")
    return post


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    log(f"=== Daily Blog Post — {today} ===")
    errors = []

    # TenantStack
    if TENANTSTACK_BLOG_API_KEY:
        try:
            topic = pick_topic("TenantStack property management", TENANTSTACK_TOPICS)
            post_to_tenantstack(topic)
        except Exception as e:
            log(f"TenantStack: ❌ Error — {e}")
            errors.append(f"TenantStack: {e}")
    else:
        log("TenantStack: ⚠️  TENANTSTACK_BLOG_API_KEY not set — skipping")

    # PhysicianPad
    if PHYSICIANPAD_BLOG_API_KEY:
        try:
            topic = pick_topic("PhysicianPad medical scribing", PHYSICIANPAD_TOPICS)
            post_to_physicianpad(topic)
        except Exception as e:
            log(f"PhysicianPad: ❌ Error — {e}")
            errors.append(f"PhysicianPad: {e}")
    else:
        log("PhysicianPad: ⚠️  PHYSICIANPAD_BLOG_API_KEY not set — skipping")

    if errors:
        log(f"Completed with {len(errors)} error(s).")
        sys.exit(1)
    else:
        log("All posts published successfully. ✅")


if __name__ == "__main__":
    main()

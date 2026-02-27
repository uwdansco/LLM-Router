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
    # Long-tail, SEO-optimized topic angles — each maps to a target keyword cluster
    "How to screen tenants without violating fair housing laws",
    "Property management software for small landlords: what to look for",
    "How to collect rent online as a landlord (and actually get paid on time)",
    "How to raise rent without losing good tenants",
    "How to manage rental properties without a property manager",
    "How to handle difficult tenants professionally and legally",
    "How to handle maintenance requests before they become emergencies",
    "Security deposit mistakes landlords keep making",
    "How to reduce tenant turnover: a landlord retention strategy that works",
    "How to scale a rental portfolio from 1 property to 10",
    "Late rent payments: when to call, when to text, when to serve notice",
    "How to track rental income and expenses without a spreadsheet",
    "Self-managing vs hiring a property manager: a real cost breakdown",
    "How to write a rental listing that attracts quality tenants",
    "Eviction prevention: early warning signs every landlord should know",
    "Tax deductions rental property owners consistently miss",
    "How AI property management tools are changing the landlord game in 2025",
    "What tenants actually want (and why it matters to your bottom line)",
    "Landlord tenant screening checklist: what to check before signing a lease",
    "Best property management app for independent landlords with 1-10 units",
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
    # Long-tail, SEO-optimized topic angles — each maps to a target keyword cluster
    "AI scribe for chiropractors: faster SOAP notes without the midnight catch-up",
    "How AI scribing is giving physicians 2 hours back every day",
    "The hidden cost of manual charting (it's more than you think)",
    "AI scribe for therapists: how to cut therapy progress note time in half",
    "AI medical scribe for dentists: what's changing in dental documentation",
    "Physician burnout and documentation: breaking the midnight charting cycle",
    "How ambient AI documentation works — and why it's different from transcription",
    "EHR efficiency tips every primary care physician needs in 2025",
    "Telehealth documentation tips: common mistakes and how to fix them",
    "How psychiatrists can streamline session notes with AI scribing",
    "AI medical scribe vs traditional transcription: a real comparison",
    "How to reduce after-hours charting as a primary care physician",
    "How to improve patient engagement by spending less time on charts",
    "Billing accuracy and AI scribing: what the data shows",
    "Ambient AI documentation for mental health practitioners",
    "The ROI of AI medical scribing for small practices",
    "How to onboard your practice to an AI scribe without disruption",
    "Best AI scribe for small medical practices: what to look for",
    "How long should clinical notes take? A benchmark for busy clinicians",
    "How AI is reducing physician burnout through smarter documentation",
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

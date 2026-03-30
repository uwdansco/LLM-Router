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
load_dotenv(override=True)
load_dotenv(Path.home() / ".llm-router.env", override=False)

ANTHROPIC_API_KEY             = os.getenv("ANTHROPIC_API_KEY", "")
TENANTSTACK_BLOG_API_KEY      = os.getenv("TENANTSTACK_BLOG_API_KEY", "")
TENANTSTACK_SUPABASE_ANON_KEY = os.getenv("TENANTSTACK_SUPABASE_ANON_KEY", "")
TENANTSTACK_BLOG_URL          = "https://tsikzygmwawvxheisdhc.supabase.co/functions/v1/blog-api"
PHYSICIANPAD_BLOG_API_KEY = os.getenv("PHYSICIANPAD_BLOG_API_KEY", "")
PHYSICIANPAD_BLOG_URL     = "https://blog.physicianpad.com/api/admin/posts"

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
today  = datetime.now().strftime("%A, %B %d, %Y")


# ── Helpers ───────────────────────────────────────────────────────────────────

def log(msg: str):
    # Strip non-ASCII to avoid Windows cp1252 encoding crashes
    safe = msg.encode("ascii", "replace").decode("ascii")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {safe}", flush=True)


def slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_-]+", "-", text)
    text = re.sub(r"^-+|-+$", "", text)
    return text[:80]


def safe_json_parse(raw: str) -> dict:
    """Parse JSON from Claude, with fallback for unescaped HTML double quotes."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Fallback: extract fields via regex (handles unescaped quotes in HTML content)
    result = {}
    for key in ["title", "meta_title", "slug", "excerpt", "category", "category_slug", "authorName", "authorRole"]:
        m = re.search(rf'"{re.escape(key)}"\s*:\s*"((?:[^"\\]|\\.)*)"', raw)
        if m:
            result[key] = m.group(1)

    # Extract content field (largest string value, often contains unescaped HTML)
    m = re.search(r'"content"\s*:\s*"([\s\S]*?)"\s*,\s*"faq"', raw)
    if not m:
        m = re.search(r'"content"\s*:\s*"([\s\S]*?)"\s*[}\n]', raw)
    if m:
        result["content"] = m.group(1).replace('\\"', '"')
    else:
        idx = raw.find('"content"')
        if idx >= 0:
            colon = raw.find(':', idx + 9)
            quote = raw.find('"', colon + 1)
            if quote >= 0:
                # Find the end — look for ,"faq" or closing brace
                faq_idx = raw.find('"faq"', quote)
                if faq_idx > 0:
                    last_quote = raw.rfind('"', quote + 1, faq_idx)
                else:
                    last_quote = raw.rfind('"')
                result["content"] = raw[quote + 1:last_quote]

    # Extract FAQ array — try json.loads on just the faq portion
    faq_match = re.search(r'"faq"\s*:\s*(\[[\s\S]*?\])\s*}', raw)
    if faq_match:
        try:
            result["faq"] = json.loads(faq_match.group(1))
        except json.JSONDecodeError:
            result["faq"] = []

    if not result.get("title") or not result.get("content"):
        raise ValueError(f"Could not parse blog post JSON: {raw[:200]}")
    return result


def parse_json(raw: str) -> dict:
    """Strip markdown fences and parse JSON safely."""
    raw = raw.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else raw
        if raw.startswith("json"):
            raw = raw[4:]
    return safe_json_parse(raw.strip())


# ── Topic picker (with history to avoid duplicates) ──────────────────────────

TOPIC_HISTORY_FILE = Path(__file__).parent / "topic_history.json"


def load_topic_history() -> dict:
    if TOPIC_HISTORY_FILE.exists():
        try:
            return json.loads(TOPIC_HISTORY_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def save_topic_history(history: dict):
    TOPIC_HISTORY_FILE.write_text(json.dumps(history, indent=2), encoding="utf-8")


def pick_topic(blog: str, topic_pool: list[str]) -> str:
    """Ask Claude to pick the most interesting topic for today, avoiding recent topics."""
    history = load_topic_history()
    used = set(history.get(blog, []))
    available = [t for t in topic_pool if t not in used]

    # If all topics used, reset history for this blog
    if not available:
        log(f"{blog}: All {len(topic_pool)} topics used — resetting history.")
        history[blog] = []
        available = topic_pool
        save_topic_history(history)

    pool_str = "\n".join(f"- {t}" for t in available)
    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=100,
        messages=[{"role": "user", "content":
            f"Today is {today}. From the list below, pick the single most timely and "
            f"interesting topic for a {blog} blog post. Return ONLY the topic text, nothing else.\n\n{pool_str}"
        }],
    )
    chosen = msg.content[0].text.strip()

    # Save to history
    if blog not in history:
        history[blog] = []
    history[blog].append(chosen)
    save_topic_history(history)
    log(f"{blog}: Picked topic ({len(history[blog])}/{len(topic_pool)} used)")

    return chosen


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
Blog URL: https://blog.tenantstack.com

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
- Aim for 1200-1800 words of content (longer posts rank higher)
- Use FAQ-style subheadings when appropriate (e.g. "How do landlords screen tenants?") — these rank for featured snippets

ADVANCED SEO — MUST INCLUDE ALL OF THESE:

1. TABLE OF CONTENTS: Start the content with a clickable table of contents using anchor links:
   <nav><h2>Table of Contents</h2><ul><li><a href='#section-slug'>Section Title</a></li>...</ul></nav>
   Then use matching id attributes on each <h2>: <h2 id='section-slug'>Section Title</h2>

2. INTERNAL LINKS: Include 2-3 internal links to related TenantStack blog topics using keyword-rich anchor text. Use the format:
   <a href='https://blog.tenantstack.com/SLUG'>anchor text with keyword</a>
   Related slugs to link to (pick 2-3 that are relevant to the topic):
   - how-to-screen-tenants-without-violating-fair-housing-laws
   - how-to-collect-rent-online-as-a-landlord
   - how-to-raise-rent-without-losing-good-tenants
   - how-to-manage-rental-properties-without-a-property-manager
   - how-to-handle-difficult-tenants-professionally-and-legally
   - how-to-reduce-tenant-turnover
   - how-to-scale-a-rental-portfolio-from-1-property-to-10
   - how-to-track-rental-income-and-expenses
   - landlord-tenant-screening-checklist
   - security-deposit-mistakes-landlords-keep-making
   - tax-deductions-rental-property-owners-miss
   - eviction-prevention-early-warning-signs

3. FAQ SCHEMA: Include 3-5 FAQ items in the "faq" field (see output format). These MUST be real questions people search for on Google, with concise 2-3 sentence answers. Google displays these as rich results.

4. META TITLE: Provide a separate "meta_title" optimized for search (can differ slightly from the display title). Max 60 characters. Front-load the primary keyword.

5. SEMANTIC HTML: Wrap the full content in <article> tags. Use <section> for each major section. Use <strong> to bold key phrases that contain target keywords (signals relevance to Google).

6. IMAGE ALT TEXT PLACEHOLDERS: Include 2-3 image placeholders with SEO-optimized alt text:
   <img src='/images/DESCRIPTIVE-NAME.webp' alt='keyword-rich description of image' loading='lazy' />

7. CALL-TO-ACTION: End with a compelling CTA section linking to TenantStack:
   <section id='cta'><h2>Ready to Simplify Your Property Management?</h2><p>...mention TenantStack features relevant to the topic...</p><a href='https://tenantstack.com'>Try TenantStack Free</a></section>

OUTPUT FORMAT — respond ONLY with valid JSON, no markdown fences:
{
  "title": "Display title here",
  "meta_title": "SEO-optimized title under 60 chars with primary keyword first",
  "slug": "post-title-here",
  "excerpt": "2-sentence compelling summary for previews and SEO (under 160 chars)",
  "category_slug": "one of: tips, guides, industry-news, technology, finance, tenant-management, maintenance",
  "content": "<article>...<full HTML content>...</article>",
  "faq": [
    {"question": "Real question people Google?", "answer": "Concise 2-3 sentence answer."},
    {"question": "Another common search query?", "answer": "Helpful direct answer."}
  ]
}

CRITICAL JSON RULE: Inside the content HTML, use single quotes for ALL HTML attributes (e.g. href='url', class='name'). Never use double quotes inside HTML attribute values — they will break JSON encoding."""


def build_faq_schema(faq_items: list, page_url: str) -> str:
    """Generate FAQ JSON-LD schema markup for Google rich results."""
    if not faq_items:
        return ""
    entities = []
    for item in faq_items:
        entities.append({
            "@type": "Question",
            "name": item.get("question", ""),
            "acceptedAnswer": {
                "@type": "Answer",
                "text": item.get("answer", ""),
            },
        })
    schema = {
        "@context": "https://schema.org",
        "@type": "FAQPage",
        "mainEntity": entities,
    }
    return f"<script type='application/ld+json'>{json.dumps(schema)}</script>"


def build_article_schema(post: dict, page_url: str, publisher: str, logo_url: str) -> str:
    """Generate Article JSON-LD schema markup for Google rich results."""
    schema = {
        "@context": "https://schema.org",
        "@type": "Article",
        "headline": post.get("meta_title", post.get("title", "")),
        "description": post.get("excerpt", ""),
        "author": {
            "@type": "Organization" if publisher == "TenantStack" else "Person",
            "name": post.get("authorName", publisher),
        },
        "publisher": {
            "@type": "Organization",
            "name": publisher,
            "logo": {"@type": "ImageObject", "url": logo_url},
        },
        "url": page_url,
        "datePublished": datetime.now().strftime("%Y-%m-%dT%H:%M:%S+00:00"),
        "dateModified": datetime.now().strftime("%Y-%m-%dT%H:%M:%S+00:00"),
        "mainEntityOfPage": {"@type": "WebPage", "@id": page_url},
    }
    return f"<script type='application/ld+json'>{json.dumps(schema)}</script>"


def post_to_tenantstack(topic: str) -> dict:
    log(f"TenantStack: writing post on '{topic}'...")
    msg = client.messages.create(
        model="claude-opus-4-5-20251101",
        max_tokens=6000,
        system=TENANTSTACK_WRITER_PROMPT,
        messages=[{"role": "user", "content":
            f"Today is {today}. Write a blog post about: {topic}\n\nFollow all tone, style, SEO, and format rules."
        }],
    )
    post = parse_json(msg.content[0].text)
    if not post.get("slug"):
        post["slug"] = slugify(post.get("title", topic))

    # Append date to slug to avoid duplicates
    date_suffix = datetime.now().strftime("-%Y%m%d")
    if not post["slug"].endswith(date_suffix):
        post["slug"] += date_suffix

    # Inject structured data (FAQ + Article schema) into content
    page_url = f"https://blog.tenantstack.com/{post['slug']}"
    faq_schema = build_faq_schema(post.get("faq", []), page_url)
    article_schema = build_article_schema(post, page_url, "TenantStack", "https://tenantstack.com/logo.png")
    content = post["content"] + "\n" + faq_schema + "\n" + article_schema

    headers = {
        "Content-Type": "application/json",
        "x-api-key": TENANTSTACK_BLOG_API_KEY,
        "Authorization": f"Bearer {TENANTSTACK_SUPABASE_ANON_KEY}",
    }
    payload = {
        "title":         post.get("meta_title", post["title"]),
        "slug":          post["slug"],
        "excerpt":       post.get("excerpt", ""),
        "content":       content,
        "category_slug": post.get("category_slug", "tips"),
        "author":        "Jarvis",
        "status":        "published",
    }
    resp = requests.post(TENANTSTACK_BLOG_URL, json=payload, headers=headers, timeout=60)
    resp.raise_for_status()
    faq_count = len(post.get("faq", []))
    log(f"TenantStack: Published '{post['title']}' (FAQ schema: {faq_count} items, article schema: yes)")
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
Blog URL: https://blog.physicianpad.com

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
- Aim for 1200-1800 words of content (longer posts rank higher)
- Use FAQ-style subheadings when appropriate (e.g. "Does AI scribing work for chiropractors?") — these rank for featured snippets and AI search results

ADVANCED SEO — MUST INCLUDE ALL OF THESE:

1. TABLE OF CONTENTS: Start the content with a clickable table of contents using anchor links:
   <nav><h2>Table of Contents</h2><ul><li><a href='#section-slug'>Section Title</a></li>...</ul></nav>
   Then use matching id attributes on each <h2>: <h2 id='section-slug'>Section Title</h2>

2. INTERNAL LINKS: Include 2-3 internal links to related PhysicianPad blog topics using keyword-rich anchor text. Use the format:
   <a href='https://blog.physicianpad.com/SLUG'>anchor text with keyword</a>
   Related slugs to link to (pick 2-3 that are relevant to the topic):
   - ai-scribe-for-chiropractors
   - how-ai-scribing-gives-physicians-2-hours-back
   - hidden-cost-of-manual-charting
   - ai-scribe-for-therapists
   - ai-medical-scribe-for-dentists
   - physician-burnout-and-documentation
   - how-ambient-ai-documentation-works
   - ehr-efficiency-tips-primary-care
   - telehealth-documentation-tips
   - ai-medical-scribe-vs-traditional-transcription
   - reduce-after-hours-charting
   - roi-of-ai-medical-scribing-small-practices
   - best-ai-scribe-for-small-medical-practices

3. FAQ SCHEMA: Include 3-5 FAQ items in the "faq" field (see output format). These MUST be real questions people search for on Google, with concise 2-3 sentence answers. Google displays these as rich results.

4. META TITLE: Provide a separate "meta_title" optimized for search (can differ slightly from the display title). Max 60 characters. Front-load the primary keyword.

5. SEMANTIC HTML: Wrap the full content in <article> tags. Use <section> for each major section. Use <strong> to bold key phrases that contain target keywords (signals relevance to Google).

6. IMAGE ALT TEXT PLACEHOLDERS: Include 2-3 image placeholders with SEO-optimized alt text:
   <img src='/images/DESCRIPTIVE-NAME.webp' alt='keyword-rich description of image' loading='lazy' />

7. CALL-TO-ACTION: End with a compelling CTA section linking to PhysicianPad:
   <section id='cta'><h2>Spend Less Time Charting, More Time With Patients</h2><p>...mention PhysicianPad features relevant to the topic...</p><a href='https://physicianpad.com'>Try PhysicianPad Free</a></section>

OUTPUT FORMAT — respond ONLY with valid JSON, no markdown fences:
{
  "title": "Display title here",
  "meta_title": "SEO-optimized title under 60 chars with primary keyword first",
  "slug": "post-title-here",
  "excerpt": "2-sentence compelling summary (under 160 chars)",
  "category": "one of: Efficiency, AI & Technology, Burnout & Wellness, Billing & Coding, Clinical Tips, Telehealth, Mental Health, Specialty Care",
  "authorName": "Dr. Sarah Chen",
  "authorRole": "Chief Medical Officer",
  "content": "<article>...<full HTML content>...</article>",
  "faq": [
    {"question": "Real question people Google?", "answer": "Concise 2-3 sentence answer."},
    {"question": "Another common search query?", "answer": "Helpful direct answer."}
  ]
}

CRITICAL JSON RULE: Inside the content HTML, use single quotes for ALL HTML attributes (e.g. href='url', class='name'). Never use double quotes inside HTML attribute values — they will break JSON encoding."""


def post_to_physicianpad(topic: str) -> dict:
    log(f"PhysicianPad: writing post on '{topic}'...")
    msg = client.messages.create(
        model="claude-opus-4-5-20251101",
        max_tokens=6000,
        system=PHYSICIANPAD_WRITER_PROMPT,
        messages=[{"role": "user", "content":
            f"Today is {today}. Write a blog post about: {topic}\n\nFollow all tone, style, SEO, and format rules."
        }],
    )
    post = parse_json(msg.content[0].text)
    if not post.get("slug"):
        post["slug"] = slugify(post.get("title", topic))

    # Append date to slug to avoid duplicates
    date_suffix = datetime.now().strftime("-%Y%m%d")
    if not post["slug"].endswith(date_suffix):
        post["slug"] += date_suffix

    # Inject structured data (FAQ + Article schema) into content
    page_url = f"https://blog.physicianpad.com/{post['slug']}"
    faq_schema = build_faq_schema(post.get("faq", []), page_url)
    article_schema = build_article_schema(
        post, page_url, "PhysicianPad", "https://physicianpad.com/logo.png"
    )
    content = post["content"] + "\n" + faq_schema + "\n" + article_schema

    word_count = len(content.split())
    read_time  = f"{max(1, round(word_count / 200))} min read"

    headers = {"Content-Type": "application/json", "X-Api-Key": PHYSICIANPAD_BLOG_API_KEY}
    payload = {
        "title":      post.get("meta_title", post["title"]),
        "slug":       post["slug"],
        "excerpt":    post.get("excerpt", ""),
        "content":    content,
        "category":   post.get("category", "Efficiency"),
        "status":     "published",
        "authorName": post.get("authorName", "Dr. Sarah Chen"),
        "authorRole": post.get("authorRole", "Chief Medical Officer"),
        "readTime":   read_time,
        "featured":   False,
    }
    resp = requests.post(PHYSICIANPAD_BLOG_URL, json=payload, headers=headers, timeout=60)
    resp.raise_for_status()
    faq_count = len(post.get("faq", []))
    log(f"PhysicianPad: Published '{post['title']}' (FAQ schema: {faq_count} items, article schema: yes)")
    return post


# ── Sitemap & Search Engine Indexing ─────────────────────────────────────────
#
# Both platforms have DYNAMIC sitemaps that auto-generate from the database:
#   TenantStack:  https://tsikzygmwawvxheisdhc.supabase.co/functions/v1/sitemap
#   PhysicianPad: https://blog.physicianpad.com/sitemap.xml
#
# After publishing, we:
#   1. Ping Google & Bing with the sitemap URLs
#   2. Submit each new post URL via IndexNow for instant indexing
#   3. Submit individual URLs to Google via the Indexing API (if configured)

# Live sitemap URLs (auto-generated by each platform from the database)
TENANTSTACK_SITEMAP_URL  = "https://tsikzygmwawvxheisdhc.supabase.co/functions/v1/sitemap"
PHYSICIANPAD_SITEMAP_URL = "https://blog.physicianpad.com/sitemap.xml"


def submit_urls_indexnow(post_url: str, sitemap_url: str, site_host: str, brand: str):
    """Submit new post URL to IndexNow for rapid Bing/Yandex/DuckDuckGo indexing.

    NOTE: IndexNow requires ALL URLs in urlList to match the host field.
    The sitemap URL may be on a different domain (e.g. Supabase), so we only
    submit the post URL itself.
    """
    key = "jarvisindexkey"
    try:
        payload = {
            "host": site_host,
            "key": key,
            "keyLocation": f"https://{site_host}/{key}.txt",
            "urlList": [post_url],
        }
        r = requests.post(
            "https://api.indexnow.org/indexnow",
            json=payload,
            headers={"Content-Type": "application/json; charset=utf-8"},
            timeout=10,
        )
        log(f"IndexNow ({brand}): Submitted {post_url} ({r.status_code})")
    except Exception as e:
        log(f"IndexNow ({brand}): Failed - {e}")


def submit_url_google(post_url: str, brand: str):
    """Submit a URL to Google for indexing via the Web Search Indexing API.
    Requires GOOGLE_INDEXING_KEY_FILE env var pointing to a service account JSON.
    Set up at: https://console.cloud.google.com > APIs & Services > Credentials"""
    key_file = os.getenv("GOOGLE_INDEXING_KEY_FILE", "")
    if not key_file:
        return  # Not configured - skip silently
    try:
        from google.oauth2 import service_account
        from googleapiclient.discovery import build

        credentials = service_account.Credentials.from_service_account_file(
            key_file, scopes=["https://www.googleapis.com/auth/indexing"]
        )
        service = build("indexing", "v3", credentials=credentials)
        service.urlNotifications().publish(
            body={"url": post_url, "type": "URL_UPDATED"}
        ).execute()
        log(f"Google Indexing ({brand}): Submitted {post_url}")
    except ImportError:
        log(f"Google Indexing ({brand}): google-api-python-client not installed - skipping")
    except Exception as e:
        log(f"Google Indexing ({brand}): Error - {e}")


def notify_search_engines(post: dict, brand: str, blog_url: str, sitemap_url: str):
    """After publishing a post, notify all search engines for fast indexing."""
    post_url = f"{blog_url}/{post.get('slug', '')}"
    site_host = blog_url.replace("https://", "").replace("http://", "").split("/")[0]

    log(f"Search ({brand}): Notifying search engines of new post...")

    # 1. Submit new post URL via IndexNow (Bing, Yandex, DuckDuckGo, Naver, Seznam)
    submit_urls_indexnow(post_url, sitemap_url, site_host, brand)

    # 2. Submit to Google Indexing API (if configured)
    submit_url_google(post_url, brand)

    log(f"Search ({brand}): Done - {post_url}")


# ── Social Media Sharing ─────────────────────────────────────────────────────

TWITTER_API_KEY        = os.getenv("TWITTER_API_KEY", "")
TWITTER_API_SECRET     = os.getenv("TWITTER_API_SECRET", "")
TWITTER_ACCESS_TOKEN   = os.getenv("TWITTER_ACCESS_TOKEN", "")
TWITTER_ACCESS_SECRET  = os.getenv("TWITTER_ACCESS_SECRET", "")
LINKEDIN_ACCESS_TOKEN  = os.getenv("LINKEDIN_ACCESS_TOKEN", "")
LINKEDIN_PERSON_URN    = os.getenv("LINKEDIN_PERSON_URN", "")
FACEBOOK_PAGE_TOKEN    = os.getenv("FACEBOOK_PAGE_TOKEN", "")
FACEBOOK_PAGE_ID       = os.getenv("FACEBOOK_PAGE_ID", "")


def generate_social_posts(post: dict, brand: str, blog_url: str) -> dict:
    """Use Claude to generate platform-specific social media posts."""
    post_url = f"{blog_url}/{post.get('slug', '')}"
    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1000,
        messages=[{"role": "user", "content":
            f"""Generate social media posts to promote this {brand} blog post.

Title: {post.get('title', '')}
Excerpt: {post.get('excerpt', '')}
URL: {post_url}

Generate posts for each platform. Be engaging, use the brand voice, and include relevant hashtags.

Respond ONLY with valid JSON, no markdown fences:
{{
  "twitter": "Tweet text (under 280 chars, include URL and 3-5 hashtags)",
  "linkedin": "LinkedIn post (2-3 paragraphs, professional tone, include URL, 3-5 hashtags at end)",
  "facebook": "Facebook post (conversational, 2-3 sentences, include URL, 1-2 hashtags)"
}}"""
        }],
    )
    return parse_json(msg.content[0].text)


def post_to_twitter(text: str) -> bool:
    """Post a tweet using Twitter API v2."""
    if not all([TWITTER_API_KEY, TWITTER_API_SECRET, TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_SECRET]):
        return False
    try:
        from requests_oauthlib import OAuth1
        auth = OAuth1(TWITTER_API_KEY, TWITTER_API_SECRET, TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_SECRET)
        resp = requests.post(
            "https://api.twitter.com/2/tweets",
            json={"text": text},
            auth=auth,
            timeout=15,
        )
        if resp.ok:
            log(f"Twitter/X: Posted successfully")
            return True
        else:
            log(f"Twitter/X: Failed ({resp.status_code}) - {resp.text[:200]}")
            return False
    except ImportError:
        log("Twitter/X: requests_oauthlib not installed - skipping")
        return False
    except Exception as e:
        log(f"Twitter/X: Error - {e}")
        return False


def post_to_linkedin(text: str) -> bool:
    """Post to LinkedIn using the API."""
    if not LINKEDIN_ACCESS_TOKEN or not LINKEDIN_PERSON_URN:
        return False
    try:
        headers = {
            "Authorization": f"Bearer {LINKEDIN_ACCESS_TOKEN}",
            "Content-Type": "application/json",
            "X-Restli-Protocol-Version": "2.0.0",
        }
        payload = {
            "author": LINKEDIN_PERSON_URN,
            "lifecycleState": "PUBLISHED",
            "specificContent": {
                "com.linkedin.ugc.ShareContent": {
                    "shareCommentary": {"text": text},
                    "shareMediaCategory": "NONE",
                }
            },
            "visibility": {"com.linkedin.ugc.MemberNetworkVisibility": "PUBLIC"},
        }
        resp = requests.post(
            "https://api.linkedin.com/v2/ugcPosts",
            json=payload,
            headers=headers,
            timeout=15,
        )
        if resp.ok or resp.status_code == 201:
            log(f"LinkedIn: Posted successfully")
            return True
        else:
            log(f"LinkedIn: Failed ({resp.status_code}) - {resp.text[:200]}")
            return False
    except Exception as e:
        log(f"LinkedIn: Error - {e}")
        return False


def post_to_facebook(text: str) -> bool:
    """Post to a Facebook Page using the Graph API."""
    if not FACEBOOK_PAGE_TOKEN or not FACEBOOK_PAGE_ID:
        return False
    try:
        resp = requests.post(
            f"https://graph.facebook.com/v18.0/{FACEBOOK_PAGE_ID}/feed",
            data={"message": text, "access_token": FACEBOOK_PAGE_TOKEN},
            timeout=15,
        )
        if resp.ok:
            log(f"Facebook: Posted successfully")
            return True
        else:
            log(f"Facebook: Failed ({resp.status_code}) - {resp.text[:200]}")
            return False
    except Exception as e:
        log(f"Facebook: Error - {e}")
        return False


def share_to_social_media(post: dict, brand: str, blog_url: str):
    """Generate and post social media content for a published blog post."""
    has_any = any([
        TWITTER_API_KEY,
        LINKEDIN_ACCESS_TOKEN,
        FACEBOOK_PAGE_TOKEN,
    ])

    log(f"Social: Generating {brand} social media posts...")
    try:
        social = generate_social_posts(post, brand, blog_url)
    except Exception as e:
        log(f"Social: Failed to generate posts - {e}")
        return

    # Save social posts to file (always — even if APIs aren't configured)
    social_dir = Path(__file__).parent / "social_posts"
    social_dir.mkdir(exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d")
    brand_slug = brand.lower().replace(" ", "")
    social_file = social_dir / f"{brand_slug}-{date_str}.json"
    social_data = {
        "brand": brand,
        "blog_title": post.get("title", ""),
        "blog_url": f"{blog_url}/{post.get('slug', '')}",
        "generated_at": datetime.now().isoformat(),
        "posts": social,
    }
    social_file.write_text(json.dumps(social_data, indent=2), encoding="utf-8")
    log(f"Social: Saved to {social_file}")

    # Post to platforms if API keys are configured
    if social.get("twitter"):
        if TWITTER_API_KEY:
            post_to_twitter(social["twitter"])
        else:
            log(f"Social: Twitter/X post ready (no API key - saved to file)")

    if social.get("linkedin"):
        if LINKEDIN_ACCESS_TOKEN:
            post_to_linkedin(social["linkedin"])
        else:
            log(f"Social: LinkedIn post ready (no API key - saved to file)")

    if social.get("facebook"):
        if FACEBOOK_PAGE_TOKEN:
            post_to_facebook(social["facebook"])
        else:
            log(f"Social: Facebook post ready (no API key - saved to file)")

    if not has_any:
        log(f"Social: No social API keys configured. Posts saved to {social_file} for manual posting.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    log(f"=== Daily Blog Post + Social + Sitemap -- {today} ===")
    errors = []
    published_posts = []

    # TenantStack
    if TENANTSTACK_BLOG_API_KEY:
        try:
            topic = pick_topic("TenantStack property management", TENANTSTACK_TOPICS)
            post = post_to_tenantstack(topic)
            published_posts.append(("TenantStack", post, "https://blog.tenantstack.com"))
        except Exception as e:
            log(f"TenantStack: Error - {e}")
            errors.append(f"TenantStack: {e}")
    else:
        log("TenantStack: TENANTSTACK_BLOG_API_KEY not set - skipping")

    # PhysicianPad
    if PHYSICIANPAD_BLOG_API_KEY:
        try:
            topic = pick_topic("PhysicianPad medical scribing", PHYSICIANPAD_TOPICS)
            post = post_to_physicianpad(topic)
            published_posts.append(("PhysicianPad", post, "https://blog.physicianpad.com"))
        except Exception as e:
            log(f"PhysicianPad: Error - {e}")
            errors.append(f"PhysicianPad: {e}")
    else:
        log("PhysicianPad: PHYSICIANPAD_BLOG_API_KEY not set - skipping")

    # Social media sharing + search engine indexing for all published posts
    sitemap_map = {
        "TenantStack": TENANTSTACK_SITEMAP_URL,
        "PhysicianPad": PHYSICIANPAD_SITEMAP_URL,
    }
    for brand, post, blog_url in published_posts:
        # Social media
        try:
            share_to_social_media(post, brand, blog_url)
        except Exception as e:
            log(f"Social ({brand}): Error - {e}")
            errors.append(f"Social ({brand}): {e}")

        # Search engine notification (sitemap ping + IndexNow + Google Indexing API)
        try:
            notify_search_engines(post, brand, blog_url, sitemap_map.get(brand, ""))
        except Exception as e:
            log(f"Search Indexing ({brand}): Error - {e}")
            errors.append(f"Search Indexing ({brand}): {e}")

    if errors:
        log(f"Completed with {len(errors)} error(s).")
        sys.exit(1)
    else:
        log("All posts published, social shared, search engines notified.")


if __name__ == "__main__":
    main()

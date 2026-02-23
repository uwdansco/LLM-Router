#!/usr/bin/env python3
"""
ONE-TIME SETUP: Get your Google OAuth refresh token for Jarvis.

Run this script ONCE locally (on your Mac) to authorize Jarvis to access
your Google Calendar and Gmail. It will open a browser window for you to
sign in, then print the values you need to add to your .env / Railway.

Prerequisites:
  1. Go to https://console.cloud.google.com
  2. Create a new project (or select an existing one)
  3. Enable these APIs:
       - Google Calendar API
       - Gmail API
  4. Go to "APIs & Services" → "Credentials"
  5. Click "Create Credentials" → "OAuth client ID"
  6. Application type: Desktop app  (NOT web application)
  7. Download the JSON and save it as client_secret.json in this folder

Then run:
  pip install google-auth-oauthlib
  python get_google_token.py
"""

import json
from pathlib import Path
from google_auth_oauthlib.flow import InstalledAppFlow

SCOPES = [
    "https://www.googleapis.com/auth/calendar",
    "https://www.googleapis.com/auth/gmail.modify",
]

SECRET_FILE = Path(__file__).parent / "client_secret.json"

if not SECRET_FILE.exists():
    print("\n❌  client_secret.json not found!")
    print("   Download it from Google Cloud Console → Credentials → your OAuth client")
    print(f"   Save it to: {SECRET_FILE}\n")
    raise SystemExit(1)

print("\n🌐  Opening browser for Google sign-in...")
print("   Sign in with the Google account you want Jarvis to use.\n")

flow = InstalledAppFlow.from_client_secrets_file(str(SECRET_FILE), SCOPES)
creds = flow.run_local_server(port=0)

# Also read client ID/secret from the file
with open(SECRET_FILE) as f:
    secret_data = json.load(f)
client_info = secret_data.get("installed") or secret_data.get("web", {})

print("\n" + "=" * 60)
print("✅  Success! Add these to your .env or Railway environment:")
print("=" * 60)
print(f"GOOGLE_CLIENT_ID={client_info.get('client_id', creds.client_id)}")
print(f"GOOGLE_CLIENT_SECRET={client_info.get('client_secret', creds.client_secret)}")
print(f"GOOGLE_REFRESH_TOKEN={creds.refresh_token}")
print("=" * 60)
print("\n⚠️   Keep these values private — they grant access to your Google account.")
print("    For Railway: add them in your service's 'Variables' tab.")
print("    For local:  paste them into your ~/.llm-router.env file.\n")

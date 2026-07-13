"""
One-time setup script: creates initial installer and manager accounts.

Usage:
    .venv\Scripts\python.exe scripts\create_users.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.database import init_db
from backend.auth import register_user

init_db()

USERS = [
    ("installer", "change-me", "installer"),
    ("manager",   "change-me",   "manager"),
]

for username, password, role in USERS:
    ok = register_user(username, password, role)
    if ok:
        print(f"Created {role}: {username}")
    else:
        print(f"Already exists: {username} (skipped)")

print("\nDone. Change these passwords before deploying to production.")

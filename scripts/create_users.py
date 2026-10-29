"""
One-time setup script: creates initial installer and manager accounts.

Usage:
    .venv\Scripts\python.exe -m scripts.create_users
"""

from backend.auth import register_user
from backend.database import init_db

init_db()

USERS = [
    ("installer", "change-me", "installer"),
    ("manager", "change-me", "manager"),
]

for username, password, role in USERS:
    ok = register_user(username, password, role)
    if ok:
        print(f"Created {role}: {username}")
    else:
        print(f"Already exists: {username} (skipped)")

print("\nDone. Change these passwords before deploying to production.")

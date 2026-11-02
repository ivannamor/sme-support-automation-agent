"""
One-time setup script: creates initial installer and manager accounts, each
with a random password shown only once.

Usage:
    .venv\\Scripts\\python.exe -m scripts.create_users
"""

import secrets

from backend.auth import register_user
from backend.database import init_db

USERS = [
    ("installer", "installer"),
    ("manager", "manager"),
]


def main() -> None:
    init_db()
    for username, role in USERS:
        password = secrets.token_urlsafe(16)
        if register_user(username, password, role):
            print(f"Created {role}: {username}  password: {password}")
        else:
            print(f"Already exists: {username} (skipped)")

    print("\nDone. Save these passwords now: they won't be shown again.")


if __name__ == "__main__":
    main()

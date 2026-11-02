"""
Changes the password of an existing installer or manager account.

Usage:
    .venv\\Scripts\\python.exe -m scripts.set_password <username>
"""

import argparse
import getpass
import sys

from backend.auth import set_password
from backend.database import init_db


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Change an account's password.")
    parser.add_argument("username")
    args = parser.parse_args(argv)

    password = getpass.getpass("New password: ")
    if password != getpass.getpass("Repeat it: "):
        print("The two passwords don't match.")
        return 1

    init_db()
    try:
        changed = set_password(args.username, password)
    except ValueError as exc:
        print(exc)
        return 1
    if not changed:
        print(f"No account named {args.username}.")
        return 1
    print(f"Password changed for {args.username}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

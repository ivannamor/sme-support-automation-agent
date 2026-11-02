"""Initializes the database and prints a sample: python -m backend.database"""

from backend.database.inventory import get_inventory
from backend.database.schema import init_db

if __name__ == "__main__":
    init_db(reset=False)

    print("\nSample inventory rows:")
    for item in get_inventory()[:10]:
        print(item)

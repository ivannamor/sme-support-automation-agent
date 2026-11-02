"""Paths, settings and the logger shared by every tool module."""

import logging
from pathlib import Path

from backend.secrets_provider import get_secret


def find_project_root(start_path: Path) -> Path:
    """
    Finds the project root by walking upward until it finds known project folders.
    This makes the file robust whether tools.py is in the root or a subfolder.
    """

    current = start_path.resolve()

    for candidate in [current] + list(current.parents):
        if (candidate / "rag").exists() and (candidate / "db").exists():
            return candidate

    # Fallback: current working directory
    return Path.cwd().resolve()


PROJECT_ROOT = find_project_root(Path(__file__).parent)
SECRETS_DIR = PROJECT_ROOT / "secrets"
CREDENTIALS_PATH = SECRETS_DIR / "credentials.json"
TOKEN_PATH = SECRETS_DIR / "token.json"

API_URL = get_secret("API_URL", "http://127.0.0.1:8000")
INSTALLER_BASE_LOCATION = get_secret("INSTALLER_BASE_LOCATION", "Milan, Italy")
INSTALLER_BASE_LAT = get_secret("INSTALLER_BASE_LAT")
INSTALLER_BASE_LON = get_secret("INSTALLER_BASE_LON")
INSTALLER_BASE_LABEL = get_secret("INSTALLER_BASE_LABEL", INSTALLER_BASE_LOCATION)
GEOCODING_COUNTRY_CODES = get_secret("GEOCODING_COUNTRY_CODES", "it")

# Setup tool-specific logger
logger = logging.getLogger("DoorTools")

"""
Centralized secrets/config loading for the app's runtime (Streamlit UI,
FastAPI inventory service, and the tools/assistant they call into).

Every secret is read through get_secret()/require_secret() instead of each
module independently calling load_dotenv() + os.getenv() with its own
defaults. os.environ is the single source of truth:

- Local dev: .env is loaded once here (via python-dotenv) to populate it,
  same as before — nothing changes about the local workflow.
- Docker / hosted deployments: don't ship .env in the image. Inject real
  environment variables at container start instead — e.g. `docker run -e`,
  an --env-file, a Kubernetes/ECS Secret, or a secret manager's own Docker
  integration (Doppler: `doppler run -- <start command>`; AWS: task-definition
  secrets from Secrets Manager/SSM injected as env vars). This module doesn't
  care which — it only ever reads os.environ, so no code changes are needed
  when you swap the secret source later.

Rotation note: this reads secrets once at process start (like the app always
has). Picking up a rotated value means restarting the process, which the
secret manager's deploy hook or orchestrator typically does anyway. True
zero-downtime rotation (re-fetching without a restart) is future work, not
implemented here.

Dev/offline scripts (ingestion, evaluation) are intentionally out of scope —
they're run locally by a developer with their own .env and aren't part of
the deployed app's runtime footprint.
"""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")  # local convenience only; no-op if the file doesn't exist

logger = logging.getLogger("Secrets")

# Secrets the app cannot run without. Checked via check_required_secrets(),
# e.g. at startup, so misconfiguration fails loudly instead of surfacing as
# a confusing downstream error.
REQUIRED_SECRETS = (
    "OPENAI_API_KEY",
    "OPENROUTER_API_KEY",
)


def get_secret(key: str, default: str = "") -> str:
    """Reads a secret/config value from the environment."""
    return os.environ.get(key, default)


def require_secret(key: str) -> str:
    """Reads a secret/config value, raising clearly if it's missing."""
    value = get_secret(key)
    if not value:
        raise RuntimeError(
            f"Missing required secret '{key}'. Set it as an environment variable "
            "(locally via .env; in Docker/hosted environments, inject it via your "
            "deployment platform or secret manager instead of baking it into the image)."
        )
    return value


def check_required_secrets() -> list[str]:
    """Returns the subset of REQUIRED_SECRETS that are currently missing."""
    return [key for key in REQUIRED_SECRETS if not get_secret(key)]

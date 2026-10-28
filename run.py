import os
import secrets
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

API_SERVER_PATH = PROJECT_ROOT / "app" / "api_server.py"
STREAMLIT_APP_PATH = PROJECT_ROOT / "app" / "app.py"

API_HEALTH_URL = "http://127.0.0.1:8000/health"
API_STARTUP_TIMEOUT = 20  # seconds


def ensure_inventory_api_key() -> None:
    """
    The inventory API refuses to run without a key. If none is configured,
    generate one for this run: both child processes inherit it through the
    environment, so local dev needs no setup and the API is never open.
    """
    from backend.secrets_provider import get_secret  # also loads .env

    if not get_secret("INVENTORY_API_KEY"):
        os.environ["INVENTORY_API_KEY"] = secrets.token_urlsafe(32)
        print("🔑 INVENTORY_API_KEY not set — generated a key for this session.")


def wait_for_api(process: subprocess.Popen) -> bool:
    deadline = time.time() + API_STARTUP_TIMEOUT
    while time.time() < deadline:
        if process.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(API_HEALTH_URL, timeout=1) as response:
                if response.status == 200:
                    return True
        except OSError:
            pass
        time.sleep(0.5)
    return False


def start_services():
    sys.path.insert(0, str(PROJECT_ROOT))
    ensure_inventory_api_key()

    print("🚀 Starting Inventory API...")

    api_process = subprocess.Popen(
        [sys.executable, str(API_SERVER_PATH)],
        cwd=str(PROJECT_ROOT),
    )

    if not wait_for_api(api_process):
        print("⚠️  Inventory API did not become healthy — inventory tools will report errors.")

    print("🎨 Starting Streamlit Interface...")

    try:
        subprocess.run(
            ["streamlit", "run", str(STREAMLIT_APP_PATH), "--server.runOnSave=true"],
            cwd=str(PROJECT_ROOT),
        )
    except KeyboardInterrupt:
        print("\n🛑 Shutting down services...")
    finally:
        api_process.terminate()
        api_process.wait()


if __name__ == "__main__":
    start_services()

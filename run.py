import os
import secrets
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

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


def stop_on_sigterm(_signum, _frame) -> None:
    """
    `docker stop` sends SIGTERM. In a container this script is PID 1, which
    ignores that signal unless it has a handler, so Docker would wait 10 seconds
    and then kill everything. Handle it like Ctrl+C instead.
    """
    raise KeyboardInterrupt


def start_services():
    from backend.database import init_db

    ensure_inventory_api_key()
    signal.signal(signal.SIGTERM, stop_on_sigterm)

    # Before the API starts: its health check reads the inventory table.
    init_db()

    processes = []
    try:
        print("🚀 Starting Inventory API...")

        api_process = subprocess.Popen(
            [sys.executable, "-m", "app.api_server"],
            cwd=str(PROJECT_ROOT),
        )
        processes.append(api_process)

        if not wait_for_api(api_process):
            print("⚠️  Inventory API did not become healthy — inventory tools will report errors.")

        print("🎨 Starting Streamlit Interface...")

        streamlit_process = subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run", str(STREAMLIT_APP_PATH)],
            cwd=str(PROJECT_ROOT),
        )
        processes.append(streamlit_process)
        streamlit_process.wait()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down services...")
    finally:
        for process in processes:
            process.terminate()
        for process in processes:
            process.wait()


if __name__ == "__main__":
    start_services()

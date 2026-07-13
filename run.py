import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

API_SERVER_PATH = PROJECT_ROOT / "app" / "api_server.py"
STREAMLIT_APP_PATH = PROJECT_ROOT / "app" / "app.py"


def start_services():
    print("🚀 Starting Inventory API...")

    api_process = subprocess.Popen(
        [sys.executable, str(API_SERVER_PATH)],
        cwd=str(PROJECT_ROOT),
    )

    time.sleep(3)

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
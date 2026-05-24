import subprocess
import sys
import os

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))

    app_path = os.path.join(current_dir, "app.py")

    print(f"Uruchamianie serwera Streamlit z pliku: {app_path}")
    print("Przeglądarka otworzy się automatycznie.")

    subprocess.run([sys.executable, "-m", "streamlit", "run", app_path])
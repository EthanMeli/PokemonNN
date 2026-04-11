# launch_showdown.py
import subprocess
import time
from pathlib import Path
import os
import socket

SHOWDOWN_PATH = Path(r"C:\Users\elite\Documents\pokemon-showdown")
NUM_INSTANCES = 16
BASE_PORT = 8000


def wait_for_port(port: int, timeout: float = 30.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("localhost", port), timeout=0.5):
                return True
        except (ConnectionRefusedError, socket.timeout, OSError):
            time.sleep(0.2)
    return False

def main():
    
    processes = []
    for i in range(NUM_INSTANCES):
        port = BASE_PORT + i
        print(f"Starting Showdown on port {port}")
        p = subprocess.Popen(
            ["node", "pokemon-showdown", "start", "--no-security", str(port)],
            cwd=SHOWDOWN_PATH,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            import ctypes
            handle = ctypes.windll.kernel32.OpenProcess(0x1F0FFF, False, p.pid)
            mask = 1 << (i % os.cpu_count())
            ctypes.windll.kernel32.SetProcessAffinityMask(handle, mask)
            ctypes.windll.kernel32.CloseHandle(handle)
        except Exception as e:
            print(f"Could not set affinity for port {port}: {e}")
            
        if wait_for_port(port, timeout=30.0):
            print("ready")
            processes.append(p)
        else:
            print("FAILED to become ready in 30s")
            p.terminate()

    print(f"Launched {NUM_INSTANCES} instances. Ctrl+C to stop all.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping all instances...")
        for p in processes:
            p.terminate()
        for p in processes:
            p.wait()

if __name__ == "__main__":
    main()
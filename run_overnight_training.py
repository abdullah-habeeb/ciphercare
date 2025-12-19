"""
Robust Overnight FL Training Orchestrator
Handles process cleanup, server readiness check, and client orchestration.
"""

import subprocess
import sys
import time
import os
import socket
from datetime import datetime
import signal

def cleanup_ports(port=8081):
    """Attempt to kill processes using the FL port."""
    print(f"Cleaning up port {port}...")
    try:
        # Windows-specific port cleanup
        subprocess.run(f"for /f \"tokens=5\" %a in ('netstat -aon ^| find \":{port}\"') do taskkill /f /pid %a", 
                       shell=True, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
    except Exception:
        pass
    
    # Also blindly kill any python processes running our scripts to be safe
    # Be careful not to kill THIS script (though we usually won't match)
    scripts = ["fl_server_enhanced.py", "run_hospital_"]
    # Removed blanket taskkill to avoid self-termination


def wait_for_server(host='127.0.0.1', port=8081, timeout=30):
    """Wait until the server port is listening."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            with socket.create_connection((host, port), timeout=1):
                return True
        except (ConnectionRefusedError, OSError):
            time.sleep(1)
            print(".", end="", flush=True)
    return False

def main():
    print("="*60)
    print(f"STARTING ROBUST FL TRAINING: {datetime.now().strftime('%H:%M:%S')}")
    print("="*60)

    # 1. Cleanup
    cleanup_ports()
    time.sleep(2)

    # 2. Start Server
    print("\n[1/3] Starting Server...")
    server_log = open("fl_server_output.log", "w")
    server_proc = subprocess.Popen(
        [sys.executable, "fl_server_enhanced.py"],
        stdout=server_log,
        stderr=subprocess.STDOUT,
        text=True
    )
    
    print(f"      Server PID: {server_proc.pid}")
    print("      Waiting for server to be ready...", end="")
    
    if wait_for_server(port=8081):
        print(" Connected!")
    else:
        print("\n\n❌ Server failed to start or is not listening on 8081.")
        print("Check fl_server_output.log")
        server_proc.terminate()
        return

    # 3. Start Clients
    clients = [
        ("A", "run_hospital_a_client_enhanced.py"),
        ("B", "run_hospital_b_client_enhanced.py"),
        ("C", "run_hospital_c_client_enhanced.py"),
        ("D", "run_hospital_d_client_enhanced.py"),
        ("E", "run_hospital_e_client_enhanced.py")
    ]
    
    print("\n[2/3] Starting 5 Clients...")
    client_procs = []
    
    for hid, script in clients:
        if not os.path.exists(script):
            print(f"⚠️  Missing script for {hid}: {script}")
            continue
            
        print(f"      Starting Hospital {hid}...", end="")
        log = open(f"hospital_{hid}_output.log", "w")
        proc = subprocess.Popen(
            [sys.executable, script],
            stdout=log,
            stderr=subprocess.STDOUT
        )
        client_procs.append((proc, log))
        print(f" PID: {proc.pid}")
        time.sleep(3) # Stagger to prevent CPU spike / race conditions

    print("\n[3/3] Training Running! (Monitoring...)")
    print("      Press Ctrl+C to stop manually.")
    
    try:
        # Wait for server to finish (it exits after rounds complete)
        # We poll periodically instead of blocking wait to allow for timeout logic if needed
        while server_proc.poll() is None:
            time.sleep(5)
            # Optional: Check if clients are still alive. 
            # If server dies, we should kill clients.
        
        print("\n✅ Server finished successfully.")
        
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
    finally:
        print("Shutting down processes...")
        server_proc.terminate()
        for p, l in client_procs:
            p.terminate()
            l.close()
        server_log.close()
        print("Done.")

if __name__ == "__main__":
    main()

"""
Test script to verify federated learning setup.
Connects Hospital A and Hospital D clients to the FL server.
"""
import subprocess
import time
import sys

def start_server():
    """Start the FL server in background."""
    print("Starting FL Server...")
    server_proc = subprocess.Popen(
        [sys.executable, "fl_server.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    time.sleep(3)  # Wait for server to start
    return server_proc

def start_client(hospital_name, client_script, server_address="127.0.0.1:8080"):
    """Start a hospital client."""
    print(f"Starting {hospital_name} Client...")
    client_proc = subprocess.Popen(
        [sys.executable, client_script, "--server", server_address],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    return client_proc

def main():
    print("="*60)
    print("Federated Learning Test")
    print("="*60)
    
    # Start server
    server = start_server()
    
    # Start clients
    hospital_a = start_client("Hospital A", "src/hospital_a/federated_client.py")
    time.sleep(2)
    hospital_d = start_client("Hospital D", "src/hospital_d/federated/federated_client.py")
    
    print("\n✓ Server and clients started")
    print("Monitoring FL rounds...")
    print("="*60)
    
    # Monitor server output
    try:
        while True:
            line = server.stdout.readline()
            if line:
                print(f"[SERVER] {line.strip()}")
            if "FL finished" in line or not line:
                break
    except KeyboardInterrupt:
        print("\nStopping FL test...")
    finally:
        # Cleanup
        hospital_a.terminate()
        hospital_d.terminate()
        server.terminate()
        print("✓ All processes terminated")

if __name__ == "__main__":
    main()

import subprocess
import time
import sys
import json
import os
from pathlib import Path

# --- Configuration ---
FL_ROUNDS = 3
SUBSET_FRACTION = 1.0 # Use FULL real data for convergencefast FL
SERVER_PORT = 8081  # Use same port to avoid conflicts
PYTHON_EXE = sys.executable

def run_command(cmd, desc, wait=True):
    print(f"\n[EXEC] {desc}...")
    print(f"       Command: {cmd}")
    if wait:
        result = subprocess.run(cmd, shell=True)
        if result.returncode != 0:
            print(f"[FAIL] Error in step: {desc}")
            return False
        return True
    else:
        return subprocess.Popen(cmd, shell=True)

def stage_1_baselines():
    print(f"\n{'#'*60}")
    print("STAGE 1: GENERATING LOCAL BASELINES (real local-only training, no FL)")
    print(f"{'#'*60}")

    # run_local_baselines.py trains each hospital's model on ONLY that
    # hospital's own real data (no federation) and writes real measured
    # AUROC to fl_results/metrics/before_fl.json. This used to be a hardcoded
    # dict ({"A": 0.65, "B": 0.82, ...}) that never trained anything.
    result = subprocess.run([PYTHON_EXE, "run_local_baselines.py"])
    if result.returncode != 0:
        print("Error computing local baselines.")
    else:
        print("Baselines generated from real local-only training.")
    
def stage_2_fast_fl():
    print(f"\n{'#'*60}")
    print(f"STAGE 2: FAST FL SIMULATION (Subset={SUBSET_FRACTION})")
    print(f"{'#'*60}")
    
    # 1. Kill old processes on port
    if os.name == 'nt':
        subprocess.run(f"netstat -ano | findstr :{SERVER_PORT} | findstr LISTENING > tmp_pid.txt", shell=True)
        try:
            with open("tmp_pid.txt", "r") as f:
                line = f.read().strip()
                if line:
                    pid = line.split()[-1]
                    subprocess.run(f"taskkill /F /PID {pid}", shell=True)
                    print(f"+ Killed old process on port {SERVER_PORT}")
        except:
            pass
    
    # 2. Cleanup old results
    import glob
    for f in glob.glob("fl_results/round_*.json"):
        try:
            os.remove(f)
        except:
            pass
    print("+ Cleaned up old FL round logs.")

    # 3. Start Server
    server_cmd = f"{PYTHON_EXE} fl_server_enhanced.py"
    server_proc = run_command(server_cmd, "Starting FL Server", wait=False)
    time.sleep(5) # Wait for server
    
    # 3. Start Clients (A-E)
    client_procs = []
    hospitals = ['a', 'b', 'c', 'd', 'e']
    
    first_run = True
    for h in hospitals:
        cmd = f"{PYTHON_EXE} run_hospital_{h}_client_enhanced.py --subset {SUBSET_FRACTION} > client_{h}.log 2>&1"
        proc = run_command(cmd, f"Starting Hospital {h.upper()}", wait=False)
        client_procs.append(proc)
        if first_run:
             time.sleep(5) # Give A some time to init model params?
             first_run = False
        else:
             time.sleep(2) # Stagger start
        
    print(f"\n+ FL System Active. Running for {FL_ROUNDS} rounds...")
    
    # Monitor indefinitely until server finishes (it writes 'FL Training Complete' to log)
    # For this script, we'll wait a fixed buffer or check log file modifications.
    # Since run_overnight takes ~15 mins, this subset run should take ~6 mins.
    
    start_time = time.time()
    while True:
        elapsed = time.time() - start_time
        if elapsed > 1800: # 30 min timeout
            print("timeout reached")
            break
        
        # Check logs for completion
        if os.path.exists(f"fl_results/round_{FL_ROUNDS}_aggregation.json"):
            print(f"\n+ Round {FL_ROUNDS} Aggregation detected! Finishing up...")
            time.sleep(10) # Let clients finish
            break
        time.sleep(10)
        print(f"   Waiting for Round {FL_ROUNDS}... ({int(elapsed)}s elapsed)")
        
    # Kill all
    server_proc.kill()
    for p in client_procs:
        p.kill()
    
    # Extract FL Metrics
    fl_metrics = {}
    try:
        with open(f"fl_results/round_{FL_ROUNDS}_aggregation.json") as f:
            data = json.load(f)
            for c in data.get("clients", []):
                fl_metrics[c["id"]] = c["auroc"]
    except:
        print("[WARN] Could not read Round 5 metrics.")
    
    from fl_utils.simulation_utils import save_simulation_metrics
    save_simulation_metrics("after_fl", fl_metrics)


def stage_3_personalization():
    print(f"\n{'#'*60}")
    print("STAGE 3: PERSONALIZATION (real fine-tuning of the saved global model)")
    print(f"{'#'*60}")

    # run_personalization.py loads the actual checkpoint fl_server_enhanced.py
    # now saves (fl_results/checkpoints/global_model_latest.pth), freezes the
    # shared encoder, and really fine-tunes each hospital's head on its own
    # real data. This used to be `improved = min(0.99, base + 0.035)` --
    # a fabricated number that never touched a model.
    result = subprocess.run([PYTHON_EXE, "run_personalization.py"])
    if result.returncode != 0:
        print("Error running personalization.")
    else:
        print("Personalization complete (real fine-tuning).")

def main():
    stage_1_baselines()
    stage_2_fast_fl()
    stage_3_personalization()
    
    from fl_utils.simulation_utils import generate_comparison_table
    generate_comparison_table()

if __name__ == "__main__":
    main()

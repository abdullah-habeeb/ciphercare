import json
import os
from pathlib import Path

# Config
FL_RESULTS_DIR = Path(r"c:\Users\aishw\codered5\fl_results")
CHAIN_PATH = FL_RESULTS_DIR / "blockchain_audit" / "audit_chain.json"
TARGET_ROUND = 3

def rewind_blockchain():
    # 1. Load Chain
    if not CHAIN_PATH.exists():
        print("Error: Chain not found")
        return

    with open(CHAIN_PATH, 'r') as f:
        chain = json.load(f)

    # 2. Find cut-off point
    # We want to keep up to Round 3.
    # Typically genesis=0, DP=1, R1=2, R2=3, R3=4.
    # So we want to keep blocks where 'round_number' <= 3 (or type GENESIS/DP)
    
    new_chain = []
    for block in chain:
        if block.get('block_type') == 'FL_ROUND':
            if block.get('round_number', 999) <= TARGET_ROUND:
                new_chain.append(block)
        else:
            # Keep Genesis and DP blocks
            new_chain.append(block)
    
    # 3. Save truncated chain
    with open(CHAIN_PATH, 'w') as f:
        json.dump(new_chain, f, indent=2)
    
    print(f"Chain truncated. New length: {len(new_chain)} blocks.")

    # 4. Delete future aggregation files
    # Delete round_4.json, round_5.json, etc.
    # We don't know exactly how many, just glob them and check number
    for file in FL_RESULTS_DIR.glob("round_*_aggregation.json"):
        try:
            # Extra number from filename "round_X_aggregation.json"
            parts = file.name.split('_')
            # parts[0] is 'round', parts[1] is number
            round_num = int(parts[1])
            
            if round_num > TARGET_ROUND:
                print(f"Deleting future round file: {file.name}")
                os.remove(file)
        except Exception as e:
            print(f"Skipping file {file.name}: {e}")

    print("Rewind complete.")

if __name__ == "__main__":
    rewind_blockchain()

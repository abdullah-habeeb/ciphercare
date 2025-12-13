import json
import os
from pathlib import Path

BASE_DIR = Path(r"c:\Users\aishw\codered5\fl_results")
BLOCKCHAIN_PATH = BASE_DIR / "blockchain_audit/audit_chain.json"

def sync_results_with_blockchain():
    print(f"Reading {BLOCKCHAIN_PATH}...")
    try:
        with open(BLOCKCHAIN_PATH, 'r') as f:
            chain = json.load(f)
    except Exception as e:
        print(f"Error reading blockchain: {e}")
        return

    # 1. Identify valid rounds from the chain
    valid_rounds = set()
    latest_round_data = {}
    
    print(f"Total blocks found: {len(chain)}")
    
    for block in chain:
        if block.get('block_type') == 'FL_ROUND':
            r_num = block.get('round_number') or block.get('data', {}).get('round')
            if r_num:
                valid_rounds.add(r_num)
                # Store data to regenerate file if needed
                latest_round_data[r_num] = {
                    "round": r_num,
                    "timestamp": block['timestamp'],
                    "num_clients": block['data'].get('num_clients', 5),
                    "total_samples": block['data'].get('total_samples', 0),
                    "clients": block['data'].get('client_weights', [])
                }
    
    print(f"Valid rounds found in chain: {sorted(list(valid_rounds))}")
    
    # 2. Delete extraneous files
    all_files = list(BASE_DIR.glob("round_*_aggregation.json"))
    for f in all_files:
        try:
            # Extract round number from filename "round_X_aggregation.json"
            part = f.name.split('_')[1]
            r_num = int(part)
            
            if r_num not in valid_rounds:
                print(f"DELETING extraneous file: {f.name}")
                os.remove(f)
            else:
                # Optional: Ensure content matches?
                # For now, we assume if it exists it's okay, but maybe we should overwrite to be safe.
                pass
        except:
            pass
            
    # 3. Regenerate missing valid files
    for r_num in valid_rounds:
        target_file = BASE_DIR / f"round_{r_num}_aggregation.json"
        
        # We REWRITE it to ensure it matches the blockchain exactly, 
        # as the user said "refer to it" (refer to the chain).
        data = latest_round_data[r_num]
        
        # fix client structure if needed
        for c in data['clients']:
            if 'raw_weight' not in c:
                c['raw_weight'] = 0.0
                
        with open(target_file, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Synced file: {target_file.name}")

if __name__ == "__main__":
    sync_results_with_blockchain()

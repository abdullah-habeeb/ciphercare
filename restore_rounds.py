import json
from pathlib import Path

# Paths
BASE_DIR = Path(r"c:\Users\aishw\codered5\fl_results")
BLOCKCHAIN_PATH = BASE_DIR / "blockchain_audit/audit_chain.json"

def restore_rounds_from_blockchain():
    print(f"Reading blockchain from {BLOCKCHAIN_PATH}...")
    
    with open(BLOCKCHAIN_PATH, 'r') as f:
        chain = json.load(f)
    
    restored_count = 0
    
    # Iterate through blocks to find FL_ROUND blocks
    for block in chain:
        if block['block_type'] == 'FL_ROUND':
            round_num = block.get('round_number') or block['data'].get('round')
            
            if not round_num:
                continue
                
            # Construct round data structure
            round_data = {
                "round": round_num,
                "timestamp": block['timestamp'],
                "num_clients": block['data'].get('num_clients', 5),
                "total_samples": block['data'].get('total_samples', 0),
                "clients": block['data'].get('client_weights', [])
            }
            
            # Map client_weights to clients format if needed (they look similar in the block)
            # The block data has 'client_weights' which contains id, auroc, samples, normalized_weight
            # The round file expects 'clients' with id, auroc, samples, raw_weight, normalized_weight
            
            for client in round_data['clients']:
                if 'raw_weight' not in client:
                    client['raw_weight'] = 0.0 # Missing in block, defaulting
            
            # Write to file
            target_file = BASE_DIR / f"round_{round_num}_aggregation.json"
            if not target_file.exists():
                print(f"Restoring missing file: {target_file.name}")
                with open(target_file, 'w') as f:
                    json.dump(round_data, f, indent=2)
                restored_count += 1
            else:
                print(f"File exists, skipping: {target_file.name}")
                
    print(f"Recovery complete. Restored {restored_count} round files.")

if __name__ == "__main__":
    restore_rounds_from_blockchain()

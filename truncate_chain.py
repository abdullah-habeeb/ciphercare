import json
import os
from pathlib import Path

BASE_DIR = Path(r"c:\Users\aishw\codered5\fl_results")
BLOCKCHAIN_PATH = BASE_DIR / "blockchain_audit/audit_chain.json"

def truncate_blockchain():
    print(f"Reading {BLOCKCHAIN_PATH}...")
    with open(BLOCKCHAIN_PATH, 'r') as f:
        chain = json.load(f)
    
    # Find the last "Round 1"
    last_start_index = -1
    for i, block in enumerate(chain):
        if block.get('block_type') == 'FL_ROUND' and (block.get('round_number') == 1 or block.get('data', {}).get('round') == 1):
            last_start_index = i
            
    if last_start_index == -1:
        print("No Round 1 found. Cannot identify session start.")
        return

    print(f"Latest session starts at block index {last_start_index}")
    
    # Keep Genesis + DP + The new session (up to 7 rounds)
    # We always iterate from Start Index
    new_chain = chain[:2] # Keep Genesis (0) and DP (1) usually
    
    # Actually, we should just keep the Start Block and the next 6 rounds
    # But we need to preserve the hash link? 
    # If we truncate, valid hashes might break if we don't recompute, but for simulation demo, consistency of data is key.
    # User said "revert back to this", implying the file content *is* the truth.
    # We will slice the rounds 1..7 from that session.
    
    session_blocks = []
    rounds_found = 0
    
    for i in range(last_start_index, len(chain)):
        block = chain[i]
        if block.get('block_type') == 'FL_ROUND':
            rounds_found += 1
            session_blocks.append(block)
            if rounds_found >= 7:
                break
        else:
            # Keep non-round blocks (updates etc) if they are within the range
            session_blocks.append(block)
            
    # Combine
    # If the session didn't start at 2, we might lose the 'genesis' context but 
    # essentially we want [Genesis, DP, ... Session Rounds 1-7]
    
    # Let's verify if Block 0 and 1 are Genesis/DP
    genesis_blocks = chain[:last_start_index]
    # Filter genesis to only keep essential setup blocks if needed, or just keep all history up to start?
    # No, user wants "only 7 rounds". Usually implies "Delete old history".
    
    # Hard prune: Genesis + Session Rounds 1-7
    final_chain = [chain[0]] # Genesis
    if chain[1]['block_type'] == 'DP_GUARANTEE':
        final_chain.append(chain[1])
        
    final_chain.extend(session_blocks)
    
    # Re-index blocks
    for idx, block in enumerate(final_chain):
        block['block_index'] = idx
        
    print(f"Truncated chain to {len(final_chain)} blocks (Rounds 1-{rounds_found})")
    
    with open(BLOCKCHAIN_PATH, 'w') as f:
        json.dump(final_chain, f, indent=2)
        
    # Now Sync Files
    valid_rounds = set(range(1, 8)) # 1 to 7
    all_files = list(BASE_DIR.glob("round_*_aggregation.json"))
    
    for f in all_files:
        try:
            r_num = int(f.name.split('_')[1])
            if r_num > 7:
                print(f"Deleting {f.name}")
                os.remove(f)
        except:
            pass
            
    # Regenerate 1-7 from the extracted blocks
    for block in session_blocks:
        if block['block_type'] == 'FL_ROUND':
            r_num = block.get('round_number') or block['data'].get('round')
            if r_num:
                target_file = BASE_DIR / f"round_{r_num}_aggregation.json"
                # Ensure it exists
                round_data = {
                    "round": r_num,
                    "timestamp": block['timestamp'],
                    "num_clients": block['data'].get('num_clients', 5),
                    "total_samples": block['data'].get('total_samples', 0),
                    "clients": block['data'].get('client_weights', [])
                }
                # Fix weights
                for c in round_data['clients']:
                    if 'raw_weight' not in c: c['raw_weight'] = 0.0
                    
                with open(target_file, 'w') as f:
                    json.dump(round_data, f, indent=2)

if __name__ == "__main__":
    truncate_blockchain()

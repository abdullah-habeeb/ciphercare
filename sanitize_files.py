import os

files_to_fix = [
    "fl_server_enhanced.py",
    "fl_utils/domain_relevance.py",
    "fl_utils/blockchain_audit.py",
    "fl_utils/dp_utils.py",
    "run_hospital_a_client_enhanced.py",
    "run_hospital_b_client_enhanced.py",
    "run_hospital_c_client_enhanced.py",
    "run_hospital_d_client_enhanced.py",
    "run_hospital_e_client_enhanced.py",
    "process_dp_update.py"
]

print("Sanitizing files...")
for filename in files_to_fix:
    if not os.path.exists(filename):
        print(f"Skipping (not found): {filename}")
        continue
        
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Replace Checkmark
        new_content = content.replace('\u2713', '+')
        # Replace Epsilon
        new_content = new_content.replace('\u03b5', 'epsilon')
        # Replace Delta
        new_content = new_content.replace('\u03b4', 'delta')
        
        if content != new_content:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print(f"Fixed: {filename}")
        else:
            print(f"Clean: {filename}")
            
    except Exception as e:
        print(f"Error processing {filename}: {e}")

print("Done.")

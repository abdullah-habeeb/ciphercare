
try:
    with open('fl_server_output.log', 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()
        if "Traceback" in content:
            idx = content.find("Traceback")
            print("FOUND TRACEBACK:")
            print(content[idx:idx+2000])
        else:
            print("NO TRACEBACK FOUND in full file.")
            print("Last 1000 chars:")
            print(content[-1000:])
except Exception as e:
    print(e)

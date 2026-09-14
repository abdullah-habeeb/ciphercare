import argparse
import os

parser = argparse.ArgumentParser(description="Check whether a path (and its parent) exists, and list contents if so.")
parser.add_argument("path", help="Path to probe.")
args = parser.parse_args()

path = args.path
print(f"Path exists: {os.path.exists(path)}")
if os.path.exists(path):
    print("Listing:")
    try:
        print(os.listdir(path))
    except Exception as e:
        print(e)
else:
    print("Path does not exist. Parent:")
    parent = os.path.dirname(path.rstrip("/\\"))
    print(f"Parent exists: {os.path.exists(parent)}")
    if os.path.exists(parent):
        print(os.listdir(parent))

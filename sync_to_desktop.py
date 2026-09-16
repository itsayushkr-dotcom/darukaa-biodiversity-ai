"""Utility script to sync project files to Desktop."""
import os
import shutil

SRC = os.path.dirname(os.path.abspath(__file__))
DST = r"C:\Users\itsay\Desktop\Darukaa-Biodiversity-AI"

def sync():
    os.makedirs(DST, exist_ok=True)
    count = 0
    for root, dirs, files in os.walk(SRC):
        if ".git" in root or "__pycache__" in root:
            continue
        rel_path = os.path.relpath(root, SRC)
        target_dir = os.path.join(DST, rel_path) if rel_path != '.' else DST
        os.makedirs(target_dir, exist_ok=True)
        for f in files:
            if f.endswith(".pyc"):
                continue
            s_file = os.path.join(root, f)
            d_file = os.path.join(target_dir, f)
            shutil.copy2(s_file, d_file)
            count += 1
    print(f"Synced {count} clean files to {DST}")

if __name__ == "__main__":
    sync()

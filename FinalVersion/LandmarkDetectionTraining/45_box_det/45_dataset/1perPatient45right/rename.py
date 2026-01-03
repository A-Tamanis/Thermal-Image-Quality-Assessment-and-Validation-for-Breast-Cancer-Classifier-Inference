import os
import re

folder = r"."

# regex to find the first number sequence after 'T'
pattern = re.compile(r"T0*([0-9]+)")

for filename in os.listdir(folder):
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    match = pattern.search(filename)
    if not match:
        print("No patient number found in:", filename)
        continue

    patient_num = int(match.group(1))   # e.g. "0038" → 38
    new_name = f"p{patient_num:03}.jpg"  # zero padded: p038.jpg

    old_path = os.path.join(folder, filename)
    new_path = os.path.join(folder, new_name)

    os.rename(old_path, new_path)
    print(f"Renamed {filename} → {new_name}")

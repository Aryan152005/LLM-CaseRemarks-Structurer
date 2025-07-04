import json
import os

# Load your ground truth JSON data
try:
    with open("test_data/english/cleaned_new_40_eng.json", "r", encoding="utf-8") as f:
        ground_truth_data = json.load(f)
except FileNotFoundError:
    print("Error: ground_truth_50.json not found. Please save the JSON data into this file.")
    exit()

output_dir = "test_data/english/eng_test_text_files_for_evaluation"
os.makedirs(output_dir, exist_ok=True)

for item in ground_truth_data:
    file_name = item["file_name"]
    event_info_text = item["event_info_text"]

    # Ensure file_name ends with .txt only once
    if not file_name.endswith(".txt"):
        file_name += ".txt"
    
    file_path = os.path.join(output_dir, file_name)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(event_info_text)
    print(f"Created {file_path}")

print(f"\nGenerated {len(ground_truth_data)} .txt files in the '{output_dir}' directory.")
print("This JSON file contains the ground truth for your evaluation.")

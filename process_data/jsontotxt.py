import json
import os

# Your provided ground truth data (paste the full JSON array content here)
# Make sure to save the full JSON array above into a file named 'ground_truth_50.json' first.
# Then, you can load it as follows:
try:
    with open("ground_50_truth_geminiGenerated.json", "r", encoding="utf-8") as f:
        ground_truth_data = json.load(f)
except FileNotFoundError:
    print("Error: ground_truth_50.json not found. Please save the JSON data into this file.")
    exit()


output_dir = "test_text_files_for_evaluation" # Directory where you want to save the .txt files
os.makedirs(output_dir, exist_ok=True) # Create the directory if it doesn't exist

for item in ground_truth_data:
    file_name = item["file_name"]
    event_info_text = item["event_info_text"]
    
    file_path = os.path.join(output_dir, f"{file_name}.txt")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(event_info_text)
    print(f"Created {file_path}")

print(f"\nGenerated {len(ground_truth_data)} .txt files in the '{output_dir}' directory.")
print("This JSON file contains the ground truth for your evaluation.")
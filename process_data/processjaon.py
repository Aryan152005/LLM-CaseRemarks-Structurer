import json
import sys
import os

def normalize_value(value):
    """Normalize values like 'no', 'none', 'not applicable' to 'not_specified'."""
    if isinstance(value, str) and value.strip().lower() in {'no', 'none', 'not applicable', 'not_specified'}:
        return "not_specified"
    return value

def clean_entry(entry):
    """Cleans a single dictionary entry."""
    cleaned = {}
    for key, value in entry.items():
        if isinstance(value, str):
            cleaned[key] = normalize_value(value)
        else:
            cleaned[key] = value

    # Ensure 'generated_event_sub_type_detail' exists
    if 'generated_event_sub_type_detail' not in cleaned:
        cleaned['generated_event_sub_type_detail'] = "not specified"
    
    return cleaned

def process_json_file(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    cleaned_data = [clean_entry(item) for item in data]

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(cleaned_data, f, ensure_ascii=False, indent=2)

    print(f"Processed JSON saved to: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python process_erss_json.py <input_json_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    if not os.path.isfile(input_file):
        print(f"Error: File '{input_file}' does not exist.")
        sys.exit(1)

    output_file = f"cleaned_{os.path.basename(input_file)}"
    process_json_file(input_file, output_file)

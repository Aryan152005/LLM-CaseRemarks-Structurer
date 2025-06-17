import pandas as pd
from ollama import chat
import json
import sys
import os
import time
import re # For regex parsing of field_name: value output
from transformers import AutoTokenizer # Assuming this is used for token counting
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import defaultdict
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import itertools

# For BLEU, ROUGE, Jaccard
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer

# Ensure NLTK data for tokenization
try:
    import nltk
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')


# --- IMPORTANT: Importing schema.py and its components ---
# These files (schema.py and few_shot_examples.py) are expected to be
# in the SAME directory as this script (new_main_for_updated_schema.py).
# The sys.path.append line has been removed to simplify direct imports
# when all files are in the same location.
try:
    from schema import FIELD_VALUE_SCHEMA, ALL_EVENT_SUB_TYPES, ALL_CLASSIFICATION_FIELDS
except ImportError:
    print("Error: schema.py not found. Please ensure it's in the SAME directory as new_main_for_updated_schema.py.")
    sys.exit(1)

# Import few-shot examples
try:
    from few_shot_examples import get_few_shot_examples_str
except ImportError:
    print("Error: few_shot_examples.py not found. Please ensure it's in the SAME directory as new_main_for_updated_schema.py.")
    sys.exit(1)

# --- Function to derive event_type from event_sub_type (defined locally, as in original v2_improved_compare_model.py) ---
# THIS FUNCTION IS DEFINED GLOBALLY AND SHOULD BE ACCESSIBLE.
def derive_event_type(sub_type: str) -> str:
    sub_type_upper = sub_type.upper().strip()
    
    # First, try to match against specific sub-types in the schema, case-insensitively
    # This logic matches the provided v2_improved_compare_model.py snippet's derive_event_type
    for event_type_key, sub_types_list in FIELD_VALUE_SCHEMA['event_sub_type'].items():
        for canonical_sub_type in sub_types_list:
            if sub_type_upper == canonical_sub_type.upper().strip():
                return event_type_key
    
    # If not a specific type, check if it's explicitly 'OTHERS' or starts with 'OTHERS:'
    if sub_type_upper == 'OTHERS' or sub_type_upper.startswith('OTHERS:'):
        return 'OTHERS'
    
    # If it's neither a specific type (even case-insensitively) nor OTHERS, default to OTHERS
    return 'OTHERS'

# --- Configuration ---
CLASSIFICATION_MODEL_NAME = "llama3.1:8b" # Explicitly set the model for single deployment
# CORRECTED: Changed OUTPUT_DIR back to "model_comparison_results" as per user's instruction
OUTPUT_DIR = "model_comparison_results" 
CLASSIFICATION_METRICS_DIR = os.path.join(OUTPUT_DIR, "classification_metrics")
# Using the path from your snippet for ground_truth_annotations.json
GROUND_TRUTH_ANNOTATIONS_FILE = 'data/ground_truth_annotations.json'

# Ensure output directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CLASSIFICATION_METRICS_DIR, exist_ok=True)


# --- Helper Functions (Existing and New) ---

def load_ground_truth_annotations(file_path):
    """Loads ground truth annotations from a JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Ground truth annotations file not found: {file_path}")
        return {}
    except json.JSONDecodeError:
        print(f"Error decoding JSON from ground truth file: {file_path}")
        return {}

def generate_schema_instructions() -> str:
    """
    Generates a formatted string of schema rules and examples,
    to be included in the SYSTEM_PROMPT, based on the FIELD_VALUE_SCHEMA structure.
    This now correctly handles lists, strings, and dicts in FIELD_VALUE_SCHEMA.
    """
    instructions = []
    for field_name in ALL_CLASSIFICATION_FIELDS:
        field_value = FIELD_VALUE_SCHEMA.get(field_name)

        if field_name == "event_type":
            # Event_type is derived, so just list its allowed values if needed.
            # However, the prompt focuses on event_sub_type directly.
            # We can still list its possibilities for completeness in schema rules.
            if isinstance(field_value, list):
                instructions.append(f"- `{field_name}`: Categorical. Derived from `event_sub_type`. Allowed values: {', '.join(field_value)}. Example: '{field_value[0] if field_value else 'N/A'}'.")
        elif field_name == "event_sub_type":
            # Special handling for event_sub_type as its values are mapped to event_types
            # The prompt already lists ALL_EVENT_SUB_TYPES in Rule 1, so we just mention its purpose here.
            instructions.append(f"- `{field_name}`: Event sub-type. Choose the most specific option from the full list provided in Rule 1. Example: '{ALL_EVENT_SUB_TYPES[0] if ALL_EVENT_SUB_TYPES else 'N/A'}'.")
        elif isinstance(field_value, list):
            # Categorical field with allowed values as a list directly (e.g., state_of_victim, victim_gender)
            instructions.append(f"- `{field_name}`: Categorical. Allowed values: {', '.join(field_value)}. Example: '{field_value[0] if field_value else 'N/A'}'.")
        elif isinstance(field_value, str) and field_value == "text_allow_not_specified":
            # Free text field (e.g., specified_matter, incident_location)
            instructions.append(f"- `{field_name}`: Free text. Use 'not specified' if not present. Example: 'Some specific detail'.")
        else:
            # Fallback for any other unexpected structure or complex fields
            instructions.append(f"- `{field_name}`: Undefined schema. Use 'not specified' if not present. Example: 'N/A'.")

    return "\n".join(instructions)

def calculate_jaccard_similarity(predicted_dict, ground_truth_dict):
    """
    Calculates Jaccard similarity for dictionaries of extracted fields.
    Compares the intersection over union of items (key-value pairs) in the dictionaries.
    This is suitable for comparing sets of extracted items.
    """
    if not predicted_dict and not ground_truth_dict:
        return 1.0 # Both empty or perfectly matched if empty
    if not predicted_dict or not ground_truth_dict:
        return 0.0 # One is empty, the other is not

    predicted_items = set(predicted_dict.items())
    ground_truth_items = set(ground_truth_dict.items())

    intersection = len(predicted_items.intersection(ground_truth_items))
    union = len(predicted_items.union(ground_truth_items))

    return intersection / union if union != 0 else 0.0

def calculate_bleu_rouge_scores(reference_text, candidate_text):
    """
    Calculates BLEU and ROUGE scores between a reference and candidate text.
    For structured output, serialize them to a string.
    """
    if not reference_text and not candidate_text:
        return {'bleu': 1.0, 'rouge1_f1': 1.0, 'rouge2_f1': 1.0, 'rougeL_f1': 1.0}
    if not reference_text or not candidate_text:
        return {'bleu': 0.0, 'rouge1_f1': 0.0, 'rouge2_f1': 0.0, 'rougeL_f1': 0.0}

    # Tokenize for BLEU
    reference_tokens = [nltk.word_tokenize(reference_text.lower())]
    candidate_tokens = nltk.word_tokenize(candidate_text.lower())

    # BLEU Score
    try:
        # Using 1-gram for simplicity for structured text comparison
        bleu_score = sentence_bleu(reference_tokens, candidate_tokens, weights=(1, 0, 0, 0))
    except ZeroDivisionError:
        bleu_score = 0.0

    # ROUGE Score
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference_text, candidate_text)

    return {
        'bleu': bleu_score,
        'rouge1_f1': scores['rouge1'].fmeasure,
        'rouge2_f1': scores['rouge2'].fmeasure,
        'rougeL_f1': scores['rougeL'].fmeasure
    }

# --- Core Classification Pipeline ---

# Generate schema instructions once
SCHEMA_INSTRUCTIONS = generate_schema_instructions()

# The exact SYSTEM_PROMPT content provided by the user
SYSTEM_PROMPT_TEMPLATE = f"""EMERGENCY CALL CLASSIFICATION TASK:
You are analyzing 112 emergency call transcripts. Your task is to extract structured information with these STRICT RULES:

1.  **For `event_sub_type`:** You MUST select one of the following exact sub-types:
    {', '.join(ALL_EVENT_SUB_TYPES)}.
    Choose the MOST SPECIFIC sub-type that accurately describes the incident.
    If the incident does not clearly fit any of these sub-types, you MUST choose 'OTHERS'.
    NEVER use 'NULL' or 'not specified' for `event_sub_type` unless 'not specified' was an explicit option provided. Always try to be specific based on the transcript.

2.  **For `specified_matter`:** Summarize the core incident details. If you select 'OTHERS' for `event_sub_type`, you MUST provide a brief explanation of why it is 'OTHERS' and suggest a more specific descriptive phrase for the incident (e.g., "OTHERS: Caller reports unusual animal behavior not covered by existing categories."). If no specific matter, use "not specified".

3.  **For all other fields:** Extract information explicitly stated or clearly implied by the CALLER'S statements. Do not infer. If information is not present or cannot be inferred, use "not specified".

4.  **For categorical fields:** Select one of the provided exact options (case-sensitive as listed in the schema instructions below). If not specified, use "not specified".

5.  **For text fields (e.g., `incident_location`, `suspect_description`):** Extract verbatim when possible. If none, use "not specified".

**SCHEMA RULES (and examples of valid values):**
{SCHEMA_INSTRUCTIONS}

{get_few_shot_examples_str()}

**The Current Input Transcript to Classify:**
\"\"\"{{report_text}}\"\"\"

**Output MUST be in EXACT format (one field per line, `field_name: value`):**
"""

def extract_classification_from_remark(call_remark_text: str, max_retries: int = 3, retry_delay: int = 5) -> dict:
    """
    Sends a call remark to the Ollama model for classification and entity extraction,
    with robust error handling and retries. Parses the 'field_name: value' output format.
    """
    # Populate the prompt with the current report_text
    current_system_prompt = SYSTEM_PROMPT_TEMPLATE.format(report_text=call_remark_text)

    messages = [
        {"role": "user", "content": current_system_prompt}
    ]

    for attempt in range(max_retries):
        try:
            print(f"Attempt {attempt + 1}/{max_retries} for Call Remark: '{call_remark_text[:50]}...'")
            response = chat(
                model=CLASSIFICATION_MODEL_NAME,
                messages=messages,
                # IMPORTANT: 'format' is NOT 'json' as the model output is explicitly 'field_name: value'
                options={'temperature': 0.0, 'num_predict': -1, 'top_k': 1, 'top_p': 0.1}, # Optimize for deterministic output
                stream=False # Ensure the full response is received at once
            )

            response_content = response.get('message', {}).get('content', '').strip()
            if not response_content:
                print(f"[{CLASSIFICATION_MODEL_NAME}] Model returned empty response content for '{call_remark_text[:50]}...'.")
                raise ValueError("Empty model response")

            # --- Robust Parsing of 'field_name: value' output ---
            parsed_data = {}
            lines = response_content.split('\n')
            for line in lines:
                line = line.strip()
                if not line:
                    continue # Skip empty lines

                # Filter out obvious non-data lines (model's conversational filler, markdown fences, extended explanations)
                if any(keyword in line for keyword in ["Based on the input transcript", "```", "Note that", "The rest of the fields"]):
                    continue

                # Use regex to robustly parse "field_name: value"
                # Allows for optional leading bullet points (* or -), optional backticks around field_name, and captures everything after colon as value
                match = re.match(r'^\s*[\*\-]?\s*`?([a-zA-Z_]+)`?:\s*(.*)$', line)
                if match:
                    field_name = match.group(1).strip()
                    value = match.group(2).strip()
                    parsed_data[field_name] = value
                else:
                    print(f"[{CLASSIFICATION_MODEL_NAME}] Warning: Could not parse line '{line}' in expected 'field_name: value' format.")


            # --- Validation and Correction of Parsed Data ---
            final_extracted_data = {}
            extracted_other_fields = {} # To hold all fields EXCEPT event_type and event_sub_type

            # Process event_sub_type first as it's critical for event_type derivation
            model_sub_type_raw = parsed_data.get('event_sub_type', 'not specified')
            # Always ensure event_sub_type is a recognized canonical value or 'OTHERS'
            canonical_sub_type = "OTHERS"
            model_sub_type_upper = model_sub_type_raw.upper().strip()
            for allowed_st in ALL_EVENT_SUB_TYPES:
                if model_sub_type_upper == allowed_st.upper().strip():
                    canonical_sub_type = allowed_st
                    break
            if model_sub_type_upper.startswith("OTHERS"):
                canonical_sub_type = "OTHERS" # Standardize "OTHERS: explanation" to just "OTHERS" for classification

            final_extracted_data['event_sub_type'] = canonical_sub_type
            # Derive event_type based on the *validated* event_sub_type
            final_extracted_data['event_type'] = derive_event_type(final_extracted_data['event_sub_type']) # This is the line that caused the NameError

            # Populate other fields based on ALL_CLASSIFICATION_FIELDS
            for field in ALL_CLASSIFICATION_FIELDS:
                if field in ["event_type", "event_sub_type"]:
                    continue # Already handled as top-level fields

                value = parsed_data.get(field, 'not specified')

                # Handle specified_matter rule: if event_sub_type is OTHERS, specified_matter must have an explanation
                if field == 'specified_matter' and final_extracted_data['event_sub_type'] == 'OTHERS' and value == 'not specified':
                    print(f"[{CLASSIFICATION_MODEL_NAME}] Warning: 'specified_matter' is 'not specified' while event_sub_type is 'OTHERS'. Model should have provided an explanation.")
                    # Keep 'not specified' if model failed, but log warning.
                
                extracted_other_fields[field] = value
            
            final_extracted_data['extracted_fields'] = extracted_other_fields

            # Add operational metadata
            final_extracted_data['original_call_remark'] = call_remark_text
            final_extracted_data['model_response_time_s'] = response.get('total_duration', 0) / 1e9
            final_extracted_data['eval_count'] = response.get('eval_count', 0)
            final_extracted_data['eval_duration_s'] = response.get('eval_duration', 0) / 1e9

            return final_extracted_data

        except Exception as e:
            print(f"[{CLASSIFICATION_MODEL_NAME}] Error classifying remark: {e}. Retrying (Attempt {attempt + 1}/{max_retries})...")
            if attempt < max_retries - 1:
                time.sleep(retry_delay) # Wait before retrying
            else:
                print(f"[{CLASSIFICATION_MODEL_NAME}] Max retries reached for remark: '{call_remark_text[:100]}...'. Returning default error structure.")
                # Return a default/error structure if all retries fail
                return {
                    'event_type': 'OTHERS', # Default event_type
                    'event_sub_type': 'OTHERS',
                    'extracted_fields': {field: "not specified" for field in ALL_CLASSIFICATION_FIELDS if field not in ["event_type", "event_sub_type"]},
                    'original_call_remark': call_remark_text,
                    'classification_error': str(e),
                    'model_response_time_s': 0,
                    'eval_count': 0,
                    'eval_duration_s': 0
                }

def run_classification_pipeline(input_folder: str, batch_save_interval: int = 50):
    """
    Main pipeline to process call remarks, classify them using Ollama,
    and save the results with metrics.
    """
    print(f"Starting classification pipeline with model: {CLASSIFICATION_MODEL_NAME}")
    all_extracted_records = []
    processed_count = 0
    start_time = time.time()

    # Load ground truth annotations at the start
    ground_truth_annotations = load_ground_truth_annotations(GROUND_TRUTH_ANNOTATIONS_FILE)

    # Assuming input_folder contains .txt files, where each file is a call remark
    for filename in os.listdir(input_folder):
        if filename.endswith(".txt"):
            file_path = os.path.join(input_folder, filename)
            call_id = os.path.splitext(filename)[0] # Use filename as call_id

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    call_remark_text = f.read().strip()

                if not call_remark_text:
                    print(f"Skipping empty file: {filename}")
                    continue

                # Classify the remark
                extracted_record = extract_classification_from_remark(call_remark_text)
                extracted_record['call_id'] = call_id # Add call_id for easier matching

                all_extracted_records.append(extracted_record)
                processed_count += 1

                # Batch save results to prevent data loss on large datasets
                if processed_count % batch_save_interval == 0:
                    df_current_output = pd.DataFrame(all_extracted_records)
                    csv_path = os.path.join(OUTPUT_DIR, f"classified_calls_batch_{processed_count}.csv")
                    json_path = os.path.join(OUTPUT_DIR, f"classified_calls_batch_{processed_count}.json")

                    df_current_output.to_csv(csv_path, index=False, encoding='utf-8')
                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump(df_current_output.to_dict(orient='records'), f, indent=4, ensure_ascii=False)
                    print(f"Saved {processed_count} classified calls to CSV and JSON batches.")

            except Exception as e:
                print(f"Failed to process file {filename}: {e}")

    # Save any remaining records after the loop
    if all_extracted_records and processed_count % batch_save_interval != 0:
        df_final_batch = pd.DataFrame(all_extracted_records)
        csv_path_final = os.path.join(OUTPUT_DIR, f"classified_calls_final_batch_{processed_count}.csv")
        json_path_final = os.path.join(OUTPUT_DIR, f"classified_calls_final_batch_{processed_count}.json")
        df_final_batch.to_csv(csv_path_final, index=False, encoding='utf-8')
        with open(json_path_final, 'w', encoding='utf-8') as f:
            json.dump(df_final_batch.to_dict(orient='records'), f, indent=4, ensure_ascii=False)
        print(f"Saved final {processed_count} classified calls to CSV and JSON.")


    end_time = time.time()
    total_duration = end_time - start_time
    print(f"\nProcessing complete. Total files processed: {processed_count}")
    print(f"Total time taken: {total_duration:.2f} seconds")

    # Generate final metrics report using all collected records
    if all_extracted_records:
        metrics_df_final = pd.DataFrame(all_extracted_records)

        generate_single_model_metrics_report(
            CLASSIFICATION_MODEL_NAME,
            metrics_df_final,
            ground_truth_annotations, # Pass the loaded ground truth
            CLASSIFICATION_METRICS_DIR
        )

def generate_single_model_metrics_report(model_name, classified_df, ground_truth_annotations, output_dir):
    """
    Generates a comprehensive metrics report for a single model's classification performance,
    including traditional metrics and new text-based similarity metrics.
    """
    print(f"\nGenerating metrics report for {model_name}...")

    # Initialize metrics storage
    true_labels = []
    predicted_labels = []
    completeness_scores = []
    response_times = []
    eval_counts = []
    eval_durations = []

    # New metrics lists
    all_jaccard_scores = []
    all_bleu_rouge_scores = defaultdict(lambda: {'bleu': [], 'rouge1_f1': [], 'rouge2_f1': [], 'rougeL_f1': []})

    # Prepare data for metrics calculation
    for _, row in classified_df.iterrows():
        call_id = row['call_id']
        predicted_event_sub_type = row.get('event_sub_type', 'OTHERS')
        # Use the 'extracted_fields' dictionary for other predicted fields
        predicted_other_fields = row.get('extracted_fields', {})
        # For evaluation, combine event_type and event_sub_type into the other fields
        predicted_combined_fields = {
            'event_type': row.get('event_type', 'OTHERS'),
            'event_sub_type': predicted_event_sub_type
        }
        predicted_combined_fields.update(predicted_other_fields)


        # Ground truth record, defaulting to empty dict if not found
        ground_truth_record = ground_truth_annotations.get(call_id, {})
        ground_truth_event_sub_type = ground_truth_record.get('event_sub_type', 'OTHERS')
        ground_truth_other_fields = ground_truth_record.get('extracted_fields', {})
        # For evaluation, combine event_type and event_sub_type into the other fields
        ground_truth_combined_fields = {
            'event_type': ground_truth_record.get('event_type', 'OTHERS'), # Assume GT has event_type
            'event_sub_type': ground_truth_event_sub_type
        }
        ground_truth_combined_fields.update(ground_truth_other_fields)


        true_labels.append(ground_truth_event_sub_type.upper()) # Ensure case consistency for comparison
        predicted_labels.append(predicted_event_sub_type.upper()) # Ensure case consistency for comparison

        # Completeness (how many ground truth fields were extracted by the model)
        # Count fields that are NOT 'not specified' or empty in predicted
        extracted_fields_count = sum(1 for field_name, value in predicted_combined_fields.items() if value not in ["", "not specified"])
        # Count fields that are NOT 'not specified' or empty in ground truth
        ground_truth_fields_count = sum(1 for field_name, value in ground_truth_combined_fields.items() if value not in ["", "not specified"])

        if ground_truth_fields_count > 0:
            completeness_scores.append(extracted_fields_count / ground_truth_fields_count)
        else:
            completeness_scores.append(1.0) # If no ground truth fields to extract, consider complete

        # Response times and evaluation stats
        response_times.append(row.get('model_response_time_s', 0))
        eval_counts.append(row.get('eval_count', 0))
        eval_durations.append(row.get('eval_duration_s', 0))

        # Calculate Jaccard Similarity for all extracted fields (event_type, event_sub_type, and others)
        jaccard = calculate_jaccard_similarity(predicted_combined_fields, ground_truth_combined_fields)
        all_jaccard_scores.append(jaccard)

        # For BLEU/ROUGE, serialize the ground truth and predicted records into canonical strings
        # Adjusting serialization to match the "field_name: value" output for consistency in evaluation.
        # This will be a multi-line string, ensuring order is consistent with ALL_CLASSIFICATION_FIELDS.
        gt_serialized_parts = []
        pred_serialized_parts = []
        for field in ALL_CLASSIFICATION_FIELDS:
            gt_val = ground_truth_combined_fields.get(field, 'not specified')
            pred_val = predicted_combined_fields.get(field, 'not specified')
            gt_serialized_parts.append(f"{field}: {gt_val}")
            pred_serialized_parts.append(f"{field}: {pred_val}")

        gt_serialized = "\n".join(gt_serialized_parts)
        pred_serialized = "\n".join(pred_serialized_parts)

        bleu_rouge_results = calculate_bleu_rouge_scores(gt_serialized, pred_serialized)
        all_bleu_rouge_scores['bleu'].append(bleu_rouge_results['bleu'])
        all_bleu_rouge_scores['rouge1_f1'].append(bleu_rouge_results['rouge1_f1'])
        all_bleu_rouge_scores['rouge2_f1'].append(bleu_rouge_results['rouge2_f1'])
        all_bleu_rouge_scores['rougeL_f1'].append(bleu_rouge_results['rougeL_f1'])


    # Calculate overall metrics
    overall_accuracy = np.mean([1 if p == t else 0 for p, t in zip(predicted_labels, true_labels)]) * 100
    avg_completeness = np.mean(completeness_scores) * 100

    # Handle cases where true_labels or predicted_labels might be empty
    if not true_labels:
        precision, recall, f1_score = 0.0, 0.0, 0.0
    else:
        all_unique_labels = list(set(true_labels + predicted_labels))
        # Ensure 'labels' argument is provided to avoid issues with missing labels
        precision, recall, f1_score, _ = precision_recall_fscore_support(
            true_labels, predicted_labels, average='weighted', labels=all_unique_labels, zero_division=0
        )

    avg_response_time = np.mean(response_times)
    avg_tokens_per_second = np.sum(eval_counts) / np.sum(eval_durations) if np.sum(eval_durations) > 0 else 0

    # Average new metrics
    avg_jaccard = np.mean(all_jaccard_scores)
    avg_bleu = np.mean(all_bleu_rouge_scores['bleu'])
    avg_rouge1_f1 = np.mean(all_bleu_rouge_scores['rouge1_f1'])
    avg_rouge2_f1 = np.mean(all_bleu_rouge_scores['rouge2_f1'])
    avg_rougeL_f1 = np.mean(all_bleu_rouge_scores['rougeL_f1'])


    # Generate confusion matrix
    if true_labels: # Only generate if there's data
        cm = confusion_matrix(true_labels, predicted_labels, labels=sorted(list(set(true_labels + predicted_labels))))
        cm_df = pd.DataFrame(cm, index=sorted(list(set(true_labels + predicted_labels))), columns=sorted(list(set(true_labels + predicted_labels))))
        cm_path = os.path.join(output_dir, f"{model_name}_confusion_matrix.csv")
        cm_df.to_csv(cm_path)
        print(f"Confusion matrix saved to {cm_path}")
    else:
        print("No data to generate confusion matrix.")


    # Save metrics summary
    metrics_summary_path = os.path.join(output_dir, f"{model_name}_metrics_summary.txt")
    with open(metrics_summary_path, 'w') as f:
        f.write(f"--- Metrics Report for {model_name} ---\n")
        f.write(f"Overall Classification Accuracy (event_sub_type): {overall_accuracy:.2f}%\n")
        f.write(f"Overall Completeness (Extracted Fields): {avg_completeness:.2f}%\n")
        f.write(f"Weighted Precision (event_sub_type): {precision:.4f}\n")
        f.write(f"Weighted Recall (event_sub_type): {recall:.4f}\n")
        f.write(f"Weighted F1-Score (event_sub_type): {f1_score:.4f}\n")
        f.write(f"Average Model Response Time (s): {avg_response_time:.4f}\n")
        f.write(f"Average Tokens/Second (during eval): {avg_tokens_per_second:.2f}\n")
        f.write("\n--- Additional Text-Based Metrics ---\n")
        f.write(f"Average Jaccard Similarity (All Extracted Fields): {avg_jaccard:.4f}\n")
        f.write(f"Average BLEU Score (Serialized Output): {avg_bleu:.4f}\n")
        f.write(f"Average ROUGE-1 F1 Score (Serialized Output): {avg_rouge1_f1:.4f}\n")
        f.write(f"Average ROUGE-2 F1 Score (Serialized Output): {avg_rouge2_f1:.4f}\n")
        f.write(f"Average ROUGE-L F1 Score (Serialized Output): {avg_rougeL_f1:.4f}\n")

    print(f"Metrics report saved to: {metrics_summary_path}")

    # --- Plotting (for single model, similar to v2_improved_compare_model.py's individual plots) ---

    # Overall Accuracy
    plt.figure(figsize=(8, 6))
    sns.barplot(x=[model_name], y=[overall_accuracy], palette='viridis')
    plt.title(f'Overall Classification Accuracy for {model_name}')
    plt.ylabel('Accuracy (%)')
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{model_name}_overall_accuracy.png'))
    plt.close()

    # Overall Completeness
    plt.figure(figsize=(8, 6))
    sns.barplot(x=[model_name], y=[avg_completeness], palette='plasma')
    plt.title(f'Overall Completeness for {model_name}')
    plt.ylabel('Completeness (%)')
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{model_name}_overall_completeness.png'))
    plt.close()

    # Average Tokens/Second
    plt.figure(figsize=(8, 6))
    sns.barplot(x=[model_name], y=[avg_tokens_per_second], palette='cividis')
    plt.title(f'Average Tokens per Second for {model_name}')
    plt.ylabel('Tokens/Second')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{model_name}_avg_tokens_per_second.png'))
    plt.close()

    print(f"\nPerformance plots generated in '{output_dir}' directory for {model_name}.")


if __name__ == "__main__":
    # --- Option to run the standalone classification pipeline for a single model ---
    # Specify your input folder for classification. Example: 'data/ground_truth_eng'
    # Ensure 'data/ground_truth_eng' exists and contains your .txt call remark files.
    # Also ensure 'data/ground_truth_annotations.json' contains the ground truth for these files.
    # The batch_save_interval specifies how often results are saved to CSV/JSON during processing.
    run_classification_pipeline('data/ground_truth_eng', batch_save_interval=50)

    print("\nLLM Classification Evaluation Script Finished.")